"""DZ_7, кейс 1: ИИ-агент поддержки на LangGraph (RAG + тикет + контрольный слой).

Сценарий: текстовый запрос пользователя → поиск релевантной информации в
базе знаний (FAQ, векторный поиск Qdrant) → генерация проверяемого ответа
со ссылками на источники → при среднем риске — создание тикета во внешней
тикет-системе (инструмент create_ticket, function calling) и эскалация.
Оркестрация — LangGraph `StateGraph` (10 рабочих узлов + 4 терминала,
5 точек ветвления). Контрольный слой (метрики/лог/бюджет/проверка ответа)
— порт DZ_6.

Граф (порядок и семантика — порт DZ_6 + узел tool_create_ticket):
    check_memory → classify → retrieve → check_relevance → generate
    → check_answer → validate → check_validation → save → finish
плюс ветви:
    check_memory: хит в Q&A-памяти ────────→ finish_cached (без LLM-вызовов)
    classify: risk=high / parse_error ─────→ escalate
    check_relevance: контекст нерелевантен → refuse
    check_answer: пуст/длинный/чужие id ───→ check_validation (ретрай)
    check_validation: не grounded ─────────→ generate (ретрай) / escalate
    check_validation: риск medium ─────────→ tool_create_ticket → save
    любой узел: исключение ────────────────→ escalate (step_error / budget_exceeded)

Используемые метрики (взяты из DZ_6):
    - RunMetrics — метрики прогона: успех, длительность, LLM-вызовы, токены,
      стоимость в рублях (cost_rub), ретраи, модель, вызовы инструментов;
    - MeteredChat — метрирующая обёртка LLM: токены (usage или оценка len//4),
      retry с backoff на сетевые ошибки; две модели (CHEAP/ADVANCED) делят
      один бюджет (BudgetGuard);
    - BudgetGuard — предохранитель MAX_LLM_CALLS_PER_RUN: превышение →
      эскалация budget_exceeded, а не падение;
    - RunLog — лог выполнения logs/runs.jsonl (JSONL, ротация, --report, алерт);
    - check_answer — детерминированная проверка ответа без LLM.

Инструмент (порт DZ_2): create_ticket → HTTP-сервис ticket_server.py
(stdlib http.server, data/tickets.json). Аргументы LLM валидируются по
tool_schema.json (JSON Schema draft-07 — единый источник); невалидные
аргументы → повтор с текстом ошибок до MAX_VALIDATION_RETRIES; сбой
сервиса → эскалация tool_error, тикет считается несозданным.

Контекст — векторный поиск по базе знаний в Qdrant (Docker, паттерн DZ_4);
память — граф прошлых Q&A (memory/qa.json). LLM — OpenAI-совместимый
сервер (LM Studio); для selftest — заглушка FakeLLM (режимы
default/risky/medium/ungrounded/loop/flaky/badformat/tool_bad), мок Qdrant
с TF-IDF-эмбеддером и in-process мок тикет-сервиса, поэтому самодиагностика
работает без LLM, без Qdrant и без сети. Если Qdrant или LLM недоступны в
живом режиме — агент не падает: поиск деградирует до мок-памяти, на LLM —
эскалация с причиной, а не traceback.

Запуск:
    .venv/bin/python agent.py                 # интерактив
    .venv/bin/python agent.py "вопрос"         # один вопрос (+ строка метрик)
    .venv/bin/python agent.py --demo           # 4 прогона, все ветки + сводка
    .venv/bin/python agent.py --selftest       # 12 проверок без LLM
    .venv/bin/python agent.py --report         # агрегаты по logs/runs.jsonl
    .venv/bin/python agent.py --show-trace "вопрос"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable, Optional, TypedDict

import jsonschema
import openai
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

# --------------------------------------------------------------------------- #
# Конфигурация — читается из .env / переменных окружения
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "lm-studio"
# Два типа моделей (порт DZ_1): дешёвая — классификация/валидатор/типовые
# ответы, продвинутая — сложная генерация и перегенерация после провала.
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemma-4-12b-qat")
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "google/gemma-4-12b-qat")
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))

KB_FILE = os.getenv("KB_FILE", "context/kb.json")
QA_MEMORY_FILE = os.getenv("QA_MEMORY_FILE", "memory/qa.json")

# Векторный поиск (Qdrant, паттерн DZ_4)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dz7_case1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Порог релевантности — косинусное сходство (0..1) найденного контекста.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
MEMORY_HIT_THRESHOLD = float(os.getenv("MEMORY_HIT_THRESHOLD", "1.5"))
MAX_VALIDATION_RETRIES = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))

# --- Тикет-сервис (инструмент create_ticket) -------------------------------- #

TICKET_API_URL = os.getenv("TICKET_API_URL", "http://127.0.0.1:8766")
TICKET_TIMEOUT_SECONDS = float(os.getenv("TICKET_TIMEOUT_SECONDS", "5"))
# Жёсткий лимит LLM-вызовов на формирование аргументов create_ticket.
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "4"))

# --- Контрольный слой (порт DZ_6) ------------------------------------------- #

# Предохранитель: максимум LLM-вызовов за один прогон (каждая попытка —
# включая повторные после сетевой ошибки — учитывается в бюджете).
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "6"))
# Retry с backoff (0.5с/1.5с) только на транзиентные сетевые ошибки LLM.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Лимит длины ответа для детерминированной проверки check_answer.
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "2000"))
# Лог выполнения (JSONL, одна строка на прогон) — часть сдачи.
RUNS_LOG_FILE = os.getenv("RUNS_LOG_FILE", "logs/runs.jsonl")
# Максимум записей в логе: при переполнении файл ротируется в <файл>.1
# (хранится один архив). 0 — ротация выключена.
RUNS_LOG_MAX_RECORDS = int(os.getenv("RUNS_LOG_MAX_RECORDS", "1000"))
# Тариф за 1 000 000 токенов в рублях — для метрики cost_rub.
LLM_COST_INPUT_PER_1M_RUB = float(os.getenv("LLM_COST_INPUT_PER_1M_RUB", "15"))
LLM_COST_OUTPUT_PER_1M_RUB = float(os.getenv("LLM_COST_OUTPUT_PER_1M_RUB", "60"))
# Порог доли эскалаций (0..1): выше — алерт.
ERROR_ALERT_THRESHOLD = float(os.getenv("ERROR_ALERT_THRESHOLD", "0.2"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR")
logger = logging.getLogger("dz7.case1")

# Путь к схеме аргументов create_ticket (draft-07) — единый источник:
# передаётся LLM в tools=[...] и используется validate_tool_call().
TOOL_SCHEMA_PATH = os.path.join(BASE_DIR, "tool_schema.json")


def _resolve(path: str) -> str:
    """Относительный путь — от корня проекта."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# --------------------------------------------------------------------------- #
# Модели данных
# --------------------------------------------------------------------------- #

class Outcome(StrEnum):
    ANSWERED = auto()
    ANSWERED_CACHED = auto()
    REFUSED = auto()
    ESCALATED = auto()


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    document: Document
    score: float


@dataclass(frozen=True)
class Node:
    """Узел графа-памяти: документ БЗ (doc) или обработанный вопрос (qa)."""
    id: str
    label: str
    type: str  # "doc" | "qa"
    text: str


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    relation: str


@dataclass(frozen=True)
class ChatResponse:
    """Ответ LLM: текст + usage (None, если сервер usage не возвращает)."""
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class AgentResult:
    outcome: Outcome
    message: str
    sources: list[str] = field(default_factory=list)
    escalated_reason: str | None = None
    memory_saved: bool = False
    trace: list[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    """Метрики одного прогона: успех, длительность, стоимость, ретраи.

    Стоимость — в рублях (cost_rub): входные и выходные токены тарифицируются
    отдельно по тарифам за 1M токенов (LLM_COST_*_PER_1M_RUB).
    """
    duration_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    outcome: Optional[Outcome] = None
    budget_violated: bool = False
    model_used: str = ""          # CHEAP / ADVANCED (какую модель тянул прогон)
    tool_calls: int = 0           # успешных вызовов инструмента create_ticket
    tool_retries: int = 0         # сетевых повторов HTTP-вызова тикет-сервиса

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_rub(self) -> float:
        """Стоимость прогона в рублях: входные и выходные токены по
        отдельным тарифам за 1 000 000 токенов."""
        return round(
            self.prompt_tokens * LLM_COST_INPUT_PER_1M_RUB / 1_000_000
            + self.completion_tokens * LLM_COST_OUTPUT_PER_1M_RUB / 1_000_000,
            6,
        )

    @property
    def success(self) -> bool:
        """ANSWERED / ANSWERED_CACHED — успех; REFUSED — нормальный отказ
        (считается отдельно); ESCALATED — ошибка (включая тикет-эскалации)."""
        return self.outcome in (Outcome.ANSWERED, Outcome.ANSWERED_CACHED)


@dataclass(frozen=True)
class ScenarioConfig:
    relevance_threshold: float = RELEVANCE_THRESHOLD
    memory_hit_threshold: float = MEMORY_HIT_THRESHOLD
    max_validation_retries: int = MAX_VALIDATION_RETRIES
    max_steps: int = MAX_STEPS
    max_answer_chars: int = MAX_ANSWER_CHARS
    max_tool_steps: int = MAX_TOOL_STEPS


class AgentState(TypedDict, total=False):
    """Состояние графа LangGraph (аналог WorkflowContext из DZ_6).

    Узлы возвращают dict-дельту — только изменившиеся ключи.
    """
    query: str
    classification: dict                 # category, risk, complexity, language
    memory_hit: Node | None              # найденный qa-узел
    docs: list[RetrievedDoc]             # top-k из векторного поиска
    relevance_threshold: float           # эффективный порог (узлу retrieve)
    relevance_ok: bool
    answer: str
    validation: dict                     # {grounded: bool, reason: str}
    validation_source: str               # "check_answer" | "" (LLM-валидатор)
    can_retry: bool
    retry_count: int                     # инкремент — в теле check_validation
    model_used: str                      # "CHEAP" | "ADVANCED" (узел generate)
    ticket: dict | None                  # результат create_ticket
    tool_retries: int
    memory_saved: bool
    escalate_reason: str
    error: str | None                    # fatal: budget_exceeded / step_error:*
    trace: list[str]                     # путь по узлам (свой счётчик шагов)
    result: dict                         # итог терминального узла


# --------------------------------------------------------------------------- #
# Keyword-скоринг (порт DZ_1/DZ_3/DZ_6)
# --------------------------------------------------------------------------- #

_STOP_WORDS = frozenset({
    "и", "в", "во", "на", "по", "с", "со", "из", "за", "от", "о", "об", "а", "но",
    "или", "же", "бы", "не", "да", "что", "как", "кто", "где", "когда", "почему",
    "зачем", "куда", "откуда", "можно", "нужно", "надо", "для", "до", "у", "то",
    "так", "какой", "какая", "какое", "какие", "это", "этот", "эта", "эти",
    "такой", "такая", "такое", "такие", "про", "через",
})


def keyword_score(query: str, text: str) -> float:
    """Скор keyword-совпадений запроса с текстом.

    Точное совпадение = 2, префикс (первые 4 символа) = 1, подстрока = 1 —
    префикс/подстрока только для токенов >= 4 с обеих сторон (иначе «с», «по»
    дают ложные хиты). Итог: сумма / число токенов запроса.
    """
    query_words = [w for w in query.lower().split() if len(w) > 2 and w not in _STOP_WORDS]
    if not query_words:
        query_words = query.lower().split()
    text_words = text.lower().split()
    score = 0
    for q in query_words:
        for t in text_words:
            if q == t:
                score += 2
                break
            if len(q) >= 4 and len(t) >= 4 and q[:4] == t[:4]:
                score += 1
                break
            if len(q) >= 4 and len(t) >= 4 and q in t:
                score += 1
                break
    return score / max(len(query_words), 1)


# --------------------------------------------------------------------------- #
# Граф-память Q&A (порт DZ_4/DZ_6)
# --------------------------------------------------------------------------- #

class GraphMemory:
    """Хранит узлы и рёбра (adjacency list, обе направленности)."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.outgoing.setdefault(node.id, [])
        self.incoming.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        self.outgoing.setdefault(edge.from_id, []).append(edge)
        self.incoming.setdefault(edge.to_id, []).append(edge)


def load_documents(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(d["doc_id"], d["title"], d["text"]) for d in data["documents"]]


def build_runtime_memory(kb: list[Document], qa_path: str) -> GraphMemory:
    """Граф рантайма: документы БЗ (kb.json) + прошлые Q&A (qa.json).

    Документы перечитываются при каждом старте, поэтому в файл памяти
    сохраняются только qa-узлы (см. save_qa_graph).
    """
    graph = GraphMemory()
    for doc in kb:
        graph.add_node(Node(doc.doc_id, doc.title, "doc", doc.text))
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for nd in data.get("nodes", []):
            graph.add_node(Node(nd["id"], nd["label"], nd["type"], nd["text"]))
        for eg in data.get("edges", []):
            graph.add_edge(Edge(eg["from"], eg["to"], eg["relation"]))
    return graph


def _next_qa_id(graph: GraphMemory) -> str:
    max_n = 0
    for nid in graph.nodes:
        if nid.startswith("qa-") and nid[3:].isdigit():
            max_n = max(max_n, int(nid[3:]))
    return f"qa-{max_n + 1}"


def save_qa_graph(graph: GraphMemory, path: str) -> None:
    """Сохраняет граф в файл: qa-узлы + все рёбра. Атомарно через .tmp."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    nodes = [
        {"id": n.id, "label": n.label, "type": n.type, "text": n.text}
        for n in graph.nodes.values()
        if n.type == "qa"
    ]
    edges = [
        {"from": e.from_id, "to": e.to_id, "relation": e.relation}
        for edges_from in graph.outgoing.values()
        for e in edges_from
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Векторная память: эмбеддинги + Qdrant (порт DZ_4/DZ_6)
# --------------------------------------------------------------------------- #

def embed_with_llm(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Получает векторы эмбеддингов через LLM-сервер."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [[float(v) for v in d.embedding] for d in resp.data]


def _random_embed(text: str, dim: int) -> list[float]:
    """Детерминированный псевдо-вектор (хэш текста → RNG) — фолбэк без LLM."""
    import random as _r
    h = sum(ord(c) * (i + 1) for i, c in enumerate(text)) & 0xFFFFFFFF
    rng = _r.Random(h)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    """Токенизация: нижний регистр, только слова, без стоп-слов, len >= 2."""
    tokens = re.split(r"[^\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) >= 2]


class TfidfEmbedder:
    """Детерминированный TF-IDF «эмбеддер» для selftest (без сети).

    Строит IDF по корпусу, затем выдаёт TF-IDF векторы по общему словарю —
    в отличие от псевдо-векторов, поиск на нём остаётся осмысленным.
    """

    def __init__(self, corpus: list[str], dim: int = 512):
        self.dim = dim
        n_docs = len(corpus) or 1
        doc_freq: Counter = Counter()
        for text in corpus:
            for t in set(_tokenize(text)):
                doc_freq[t] += 1
        self.vocabulary = [w for w, _ in doc_freq.most_common(dim)]
        self.idf: dict[str, float] = {}
        for w in self.vocabulary:
            df = doc_freq.get(w, 0) or 1
            self.idf[w] = math.log((n_docs + 1) / (df + 1)) + 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            freq: Counter = Counter(_tokenize(text))
            total = len(freq) or 1
            vec = [
                (freq.get(w, 0) / total) * self.idf.get(w, 1.0)
                for w in self.vocabulary
            ]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class QdrantMemory:
    """Векторная память на Qdrant: upsert документов + cosine-поиск."""

    def __init__(self, url: str, collection: str,
                 embed_fn: Callable[[list[str]], list[list[float]]]):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embed_fn = embed_fn
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию (cosine), если её ещё нет."""
        from qdrant_client.http.models import Distance, VectorParams
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert_documents(self, docs: list[Document]) -> None:
        texts = [f"{d.title} {d.text}" for d in docs]
        vectors = self.embed_fn(texts)
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"doc_id": d.doc_id, "title": d.title, "text": d.text},
            )
            for i, (d, vec) in enumerate(zip(docs, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        """Возвращает [(doc_id, cosine_score, payload), ...]."""
        vec = self.embed_fn([query])[0]
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
        )
        return [(p.payload["doc_id"], p.score, p.payload) for p in resp.points]

    def close(self) -> None:
        self.client.close()


class MockQdrantMemory:
    """Заглушка Qdrant (selftest и фолбэк): тот же интерфейс, косинус в памяти."""

    def __init__(self, embed_fn: Callable[[list[str]], list[list[float]]]):
        self.embed_fn = embed_fn
        self.points: list[dict] = []

    def upsert_documents(self, docs: list[Document]) -> None:
        texts = [f"{d.title} {d.text}" for d in docs]
        for d, vec in zip(docs, self.embed_fn(texts)):
            self.points.append({
                "doc_id": d.doc_id,
                "vector": vec,
                "payload": {"doc_id": d.doc_id, "title": d.title, "text": d.text},
            })

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        qvec = self.embed_fn([query])[0]
        scored = [
            (p["doc_id"], cosine_similarity(qvec, p["vector"]), p["payload"])
            for p in self.points
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def close(self) -> None:
        pass


def make_vector_memory(client: openai.OpenAI, kb: list[Document]):
    """Векторная память: Qdrant, если доступен, иначе мок. Никогда не падает.

    LLM (embedding-модель) недоступен → детерминированные псевдо-векторы
    (поиск деградирует до «почти нет сходства» → ветка refuse, не crash).
    Qdrant недоступен → MockQdrantMemory с теми же эмбеддингами.
    """
    embed_fn: Callable[[list[str]], list[list[float]]]
    try:
        embed_fn = lambda texts: embed_with_llm(client, texts)
        embed_fn(["selftest"])  # проверка, что сервер отдаёт эмбеддинги
        print(f"Эмбеддинги: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    except Exception as e:
        print(f"[ошибка LLM] эмбеддинги недоступны ({e.__class__.__name__}). "
              f"Использую детерминированные псевдо-векторы.")
        embed_fn = lambda texts: [_random_embed(t, EMBEDDING_DIM) for t in texts]
    try:
        qmem = QdrantMemory(QDRANT_URL, QDRANT_COLLECTION, embed_fn)
        qmem.upsert_documents(kb)
        print(f"Qdrant: {QDRANT_COLLECTION} загружено ({len(kb)} документов)")
    except Exception as e:
        print(f"[ошибка Qdrant] {e.__class__.__name__}: {str(e)[:200]}. Использую мок.")
        qmem = MockQdrantMemory(embed_fn)
        qmem.upsert_documents(kb)
    return qmem


# --------------------------------------------------------------------------- #
# Контрольный слой: метрики, бюджет, retry (порт DZ_6)
# --------------------------------------------------------------------------- #

RawChatFn = Callable[[str, str], ChatResponse]  # (system, user) -> ответ
ChatFn = Callable[[str, str], str]  # (system, user) -> текст (после обёртки)


def _usage_tokens(resp: ChatResponse, system: str, user: str) -> tuple[int, int]:
    """Токены: из usage, если сервер его отдаёт; иначе оценка len(text)//4.

    Одна и та же формула для реального клиента и FakeLLM — детерминизм selftest.
    """
    prompt = resp.prompt_tokens
    completion = resp.completion_tokens
    if prompt is None:
        prompt = (len(system) + len(user)) // 4
    if completion is None:
        completion = len(resp.text) // 4
    return prompt, completion


class BudgetExceeded(Exception):
    """Бюджет LLM-вызовов за прогон исчерпан (предохранитель сработал)."""


class BudgetGuard:
    """Предохранитель: максимум LLM-вызовов за один прогон.

    Перед каждым вызовом (каждая попытка, включая повторные после сетевой
    ошибки) проверяется llm_calls < max_calls; превышение — BudgetExceeded,
    который граф переводит в эскалацию, а не падение. Две модели (CHEAP и
    ADVANCED) делят ОДИН guard — бюджет общий на прогон.
    """

    def __init__(self, max_calls: int = MAX_LLM_CALLS_PER_RUN) -> None:
        self.max_calls = max_calls
        self.metrics: RunMetrics = RunMetrics()

    def before_call(self) -> None:
        if self.metrics.llm_calls >= self.max_calls:
            raise BudgetExceeded(
                f"бюджет LLM-вызовов исчерпан: {self.metrics.llm_calls} >= {self.max_calls}"
            )


class MeteredChat:
    """Метрирующая обёртка над LLM: счётчик вызовов, токены, retry, бюджет.

    - каждый вызов (попытка) учитывается в бюджете и метриках прогона;
    - токены — из resp.usage, фолбэк len(text)//4 (см. _usage_tokens);
    - retry с backoff (0.5с, 1.5с) только на транзиентные сетевые ошибки
      (APIConnectionError/APITimeoutError), до max_retries; 4xx и прочие
      ошибки не ретраются — пробрасываются вызывающему;
    - guard может быть общим (две модели делят один бюджет и метрики).
    """

    def __init__(self, raw: RawChatFn, max_retries: int = LLM_MAX_RETRIES,
                 guard: BudgetGuard | None = None) -> None:
        self._raw = raw
        self.max_retries = max_retries
        self.backoff_s = (0.5, 1.5)
        self.guard = guard or BudgetGuard()

    def new_run(self) -> RunMetrics:
        """Новый прогон: свежие метрики и сброс счётчика бюджета."""
        self.guard.metrics = RunMetrics()
        return self.guard.metrics

    def __call__(self, system: str, user: str) -> str:
        m = self.guard.metrics
        for attempt in range(self.max_retries + 1):
            self.guard.before_call()
            m.llm_calls += 1
            try:
                resp = self._raw(system, user)
            except (openai.APIConnectionError, openai.APITimeoutError):
                # Только транзиентные сетевые ошибки ретраем; остальные — пробрасываем.
                if attempt >= self.max_retries:
                    raise
                m.retries += 1
                time.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
                continue
            p, c = _usage_tokens(resp, system, user)
            m.prompt_tokens += p
            m.completion_tokens += c
            return resp.text
        raise AssertionError("недостижимо: цикл retry завершился без ответа")


# --------------------------------------------------------------------------- #
# LLM-вызовы
# --------------------------------------------------------------------------- #

def make_openai_chat(client: openai.OpenAI, model: str) -> RawChatFn:
    """Реальный LLM: OpenAI-совместимый сервер (LM Studio). Модель — параметр."""

    def raw_chat(system: str, user: str) -> ChatResponse:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    return raw_chat


def _strip_code_fences(text: str) -> str:
    """Убирает markdown-обрамление ```json ... ``` вокруг JSON-ответа LLM."""
    raw = text.strip()
    return raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _chat_json(chat: ChatFn, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Вызывает модель и парсит JSON-ответ; при сбое парсинга не роняем граф."""
    raw = _strip_code_fences(chat(system_prompt, user_prompt))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Модель не всегда строго следует формату — деградируем в безопасную
        # сторону (ветку выбирает маршрутизация узла-вызывающего).
        return {"_parse_error": True, "_raw": raw}


def _fake_network_error() -> openai.APIConnectionError:
    """Имитация сетевой ошибки LLM (selftest: режим flaky)."""
    import httpx
    return openai.APIConnectionError(request=httpx.Request("POST", "http://fake.local/v1/chat"))


class FakeLLM:
    """Детерминированная LLM-заглушка для selftest (без сети).

    Режимы:
      default    — низкий риск, валидатор принимает ответ, аргументы тикета
                   валидны;
      risky      — классификатор возвращает risk="high" (escalate);
      medium     — классификатор возвращает risk="medium" (ветка create_ticket);
      ungrounded — валидатор всегда отвергает ответ;
      loop       — как ungrounded (для проверки гарда по шагам);
      flaky      — первые 2 вызова «падают» сетевой ошибкой (проверка retry);
      badformat  — генерация ссылается на несуществующий doc-fake
                   (проверка check_answer);
      tool_bad   — первый вызов create_ticket возвращает невалидные аргументы
                   (priority="urgent"), второй — валидные.
    """

    def __init__(self, mode: str = "default") -> None:
        self.mode = mode
        self.calls: list[tuple[str, str]] = []
        self._ticket_done = False
        self._classify_calls = 0

    def __call__(self, system: str, user: str) -> ChatResponse:
        self.calls.append((system, user))
        if self.mode == "flaky" and "классификатор" in system:
            # Первые 2 вызова классификатора «падают» сетевой ошибкой —
            # проверка retry MeteredChat на одном узле (не жжёт бюджет
            # повторными падениями обеих моделей).
            self._classify_calls += 1
            if self._classify_calls <= 2:
                raise _fake_network_error()
        if "классификатор" in system:
            if self.mode == "risky":
                risk, category = "high", "complaint"
            elif self.mode in ("medium", "tool_bad"):
                risk, category = "medium", "complaint"
            else:
                risk, category = "low", "faq"
            return ChatResponse(json.dumps({
                "category": category,
                "risk": risk,
                "complexity": "simple",
                "language": "ru",
            }, ensure_ascii=False))
        if "валидатор" in system:
            grounded = self.mode not in ("ungrounded", "loop")
            return ChatResponse(json.dumps({
                "grounded": grounded,
                "reason": "все утверждения подтверждены контекстом"
                if grounded else "в ответе есть факты, которых нет в контексте",
            }, ensure_ascii=False))
        if "инструмент тикет" in system:
            if self.mode == "tool_bad" and not self._ticket_done:
                self._ticket_done = True
                # priority "urgent" не входит в enum схемы — валидация упадёт.
                args = {"title": "Жалоба клиента", "priority": "urgent",
                        "body": "Клиент ждёт возврата денег по заказу, требует ответа."}
            else:
                args = {"title": "Обращение клиента: возврат товара", "priority": "normal",
                        "body": "Клиент задал вопрос о возврате товара. Проверены документы БЗ, "
                                "вопрос требует уточнения специалистом."}
            return ChatResponse(json.dumps(
                {"name": "create_ticket", "arguments": args}, ensure_ascii=False))
        # Генерация: doc-id берём из маркеров [doc_id] в контекстном блоке.
        if self.mode == "badformat":
            return ChatResponse("Ответ составлен по базе знаний. [источники: doc-fake]")
        doc_ids = list(dict.fromkeys(re.findall(r"\[([a-z0-9][a-z0-9-]*)\]", user)))
        src = ", ".join(doc_ids) if doc_ids else "документы не указаны"
        return ChatResponse(f"Ответ составлен по базе знаний. [источники: {src}]")


# --------------------------------------------------------------------------- #
# Промпты
# --------------------------------------------------------------------------- #

CLASSIFY_SYSTEM_PROMPT = """\
Ты — классификатор входящих запросов в поддержку.
Твоя задача: проанализировать запрос пользователя и определить его категорию, уровень риска, сложность и язык.

Сначала кратко проанализируй запрос, а затем верни ТОЛЬКО JSON без пояснений и без markdown-разметки в формате:
{
  "category": "faq" | "complaint" | "other",
  "risk": "low" | "medium" | "high",
  "complexity": "simple" | "complex",
  "language": "ru" | "en" | "other"
}
risk="high" — если запрос содержит угрозу, юридическую тему,
упоминание вреда себе или другим, оскорбления.
risk="medium" — жалоба, требующая решения человека: требование
(вернуть деньги, компенсация), неоднократно нерешённая проблема,
жалоба на сотрудника. Такие запросы агент фиксирует тикетом.
risk="low" — обычный вопрос или уточнение.
complexity="complex" — если вопрос требует синтеза нескольких фактов
или неоднозначен.
"""

ANSWER_SYSTEM_PROMPT = """\
Ты — агент поддержки. Отвечай ТОЛЬКО на основании предоставленного контекста
из базы знаний. Если контекста недостаточно — явно скажи об этом, не
придумывай факты. В конце ответа перечисли id использованных документов
в формате: [источники: doc_id1, doc_id2].
"""

VALIDATE_SYSTEM_PROMPT = """\
Ты — валидатор ответов поддержки. Тебе дан контекст и сгенерированный ответ.
Твоя задача: проверить, что КАЖДОЕ фактическое утверждение в ответе подтверждается контекстом.

Сначала проведи тщательный сравнительный анализ фактов из контекста и ответа.
Затем верни ТОЛЬКО JSON:
{ "grounded": true | false, "reason": "краткое объяснение" }
"""

TICKET_SYSTEM_PROMPT = """\
Ты — инструмент тикет агента поддержки. Запрос пользователя требует передачи
специалисту: сформируй вызов инструмента create_ticket.

Верни ТОЛЬКО один JSON без пояснений и без markdown-разметки в формате:
{"name": "create_ticket", "arguments": {"title": "...", "priority": "...", "body": "..."}}

- title: краткий заголовок сути обращения, 5-200 символов;
- priority: low | normal | high | critical — жалоба с требованием или
  жалоба на сотрудника — high, угроза/юридика — critical, остальное — low/normal;
- body: 10-2000 символов — суть запроса пользователя, что уже проверено
  агентом, id документов базы знаний, если есть.
Не придумывай факты сверх предоставленных сведений.
"""


# --------------------------------------------------------------------------- #
# Инструмент create_ticket (порт DZ_2): схема + валидация + HTTP-вызов
# --------------------------------------------------------------------------- #

def load_tool_schema() -> dict[str, Any]:
    """Читает tool_schema.json и проверяет, что это валидный draft-07."""
    try:
        with open(TOOL_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Не найден файл схемы {TOOL_SCHEMA_PATH}. "
            "Без него инструмент не может быть валидирован — восстановите файл."
        ) from exc
    jsonschema.Draft7Validator.check_schema(schema)
    return schema


TOOL_PARAMETERS: dict[str, Any] = load_tool_schema()
TOOL_SCHEMA_VALIDATOR: jsonschema.Draft7Validator = jsonschema.Draft7Validator(TOOL_PARAMETERS)

CREATE_TICKET_TOOL = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": (
            "Создаёт тикет во внешней тикет-системе поддержки и возвращает "
            "ticket_id. Используется, когда запрос требует передачи специалисту: "
            "средний риск (жалоба с требованием), нерешённая проблема."
        ),
        "parameters": TOOL_PARAMETERS,
    },
}
TOOL_NAME = CREATE_TICKET_TOOL["function"]["name"]


class TicketServiceError(Exception):
    """Сбой тикет-сервиса после всех ретраев (сеть/HTTP/битый ответ)."""


def parse_tool_response(text: str) -> tuple[Optional[str], str]:
    """Извлекает из ответа LLM вызов инструмента: (name, JSON-строка аргументов).

    Понимает две формы: {"name": "create_ticket", "arguments": {...}} и
    «голый» объект с полями title/priority/body. Если ответ не JSON —
    (None, raw).
    """
    raw = _strip_code_fences(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw
    if not isinstance(obj, dict):
        return None, raw
    if isinstance(obj.get("name"), str):
        args = obj.get("arguments")
        if isinstance(args, str):
            return obj["name"], args
        return obj["name"], json.dumps(args if isinstance(args, dict) else {},
                                       ensure_ascii=False)
    if any(k in obj for k in ("title", "priority", "body")):
        return TOOL_NAME, raw
    return None, raw


def validate_tool_call(func_name: str, raw_arguments: str) -> tuple[dict[str, Any], list[str]]:
    """Проверяет имя функции и аргументы, вернувшиеся от LLM (порт DZ_2).

    Возвращает (args, errors): args — распарсенные аргументы (или {}),
    errors — список человекочитаемых ошибок (пустой список = валидно).
    Схема tool_schema.json — единый источник: additionalProperties:false,
    enum priority, длина title/body.
    """
    if func_name != TOOL_NAME:
        return {}, [f"Неизвестная функция '{func_name}', ожидаемо '{TOOL_NAME}'."]
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        return {}, [f"Аргументы не являются корректным JSON: {exc}"]
    if not isinstance(args, dict):
        return {}, ["Аргументы должны быть JSON-объектом."]
    errors = [
        f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in TOOL_SCHEMA_VALIDATOR.iter_errors(args)
    ]
    return args, errors


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON-тело, возвращает разобранный JSON-ответ (stdlib urllib)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def create_ticket_call(args: dict[str, Any], url: str, timeout: float,
                       backoff_s: tuple[float, float] = (1.0, 2.0)
                       ) -> tuple[dict[str, Any], int]:
    """HTTP POST /tickets к тикет-сервису. Ретраи с backoff (1с/2с) только
    на сетевые ошибки/таймауты; HTTP-ошибки (4xx/5xx) ретраются не — сразу
    TicketServiceError. Возвращает ({"ticket_id", "status"}, число повторов).
    """
    endpoint = url.rstrip("/") + "/tickets"
    last: Optional[Exception] = None
    retries = 0
    for attempt in range(3):  # 1 попытка + 2 ретрая (SOP)
        try:
            payload = _http_post_json(endpoint, args, timeout)
            if not isinstance(payload, dict) or not payload.get("ticket_id"):
                raise TicketServiceError(
                    f"ответ без ticket_id: {payload!r}"
                )
            return payload, retries
        except urllib.error.HTTPError as exc:
            last = TicketServiceError(f"HTTP {exc.code}: {exc.reason}")
            break  # 4xx/5xx не ретраем — эскалация (SOP)
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, json.JSONDecodeError) as exc:
            last = TicketServiceError(f"сетевая ошибка: {exc}")
        except TicketServiceError as exc:
            raise  # битый ответ — повтор не поможет
        if attempt < 2:
            retries += 1
            time.sleep(backoff_s[min(attempt, len(backoff_s) - 1)])
    raise last if last else TicketServiceError("недостижимо: сбой без ошибки")


# --------------------------------------------------------------------------- #
# Граф LangGraph: узлы, роутеры, композиция
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Deps:
    """Зависимости узлов графа (инжект через замыкания build_graph)."""
    chat: ChatFn                  # CHEAP модель
    chat_advanced: ChatFn         # ADVANCED модель
    kb: list[Document]
    memory: GraphMemory
    qmem: Any                     # QdrantMemory | MockQdrantMemory
    qa_path: str
    config: ScenarioConfig
    ticket_url: str
    ticket_timeout: float
    ticket_backoff: tuple[float, float] = (1.0, 2.0)


def _answer_problem(answer: str, known_ids: set[str], max_chars: int) -> Optional[str]:
    """Детерминированная проверка ответа. Возвращает причину провала или None.

    - ответ непустой;
    - длина <= max_chars;
    - id в блоке [источники: …] существуют в базе знаний
      (защита от «галлюцинированных» источников).
    """
    if not answer.strip():
        return "ответ пуст"
    if len(answer) > max_chars:
        return f"ответ длиннее {max_chars} символов (сейчас {len(answer)})"
    m = re.search(r"\[источники:\s*([^\]]+)\]", answer)
    if m:
        cited = [t.strip() for t in m.group(1).split(",")]
        cited = [t for t in cited if t and t != "документы не указаны"]
        unknown = [t for t in cited if t not in known_ids]
        if unknown:
            return f"в источниках несуществующие id: {', '.join(unknown)}"
    return None


def build_graph(deps: Deps):
    """Собирает StateGraph сценария. Зависимости захватываются замыканиями.

    10 рабочих узлов + 4 терминала; 5 точек бизнес-ветвления
    (check_memory / classify / check_relevance / check_answer /
    check_validation) + сквозной error-роутинг в escalate.
    """
    known_doc_ids = {d.doc_id for d in deps.kb}
    cfg = deps.config

    # -- Обёртка узла: гард по шагам + перехват исключений (эскалация, а не падение)
    def _safe(name: str, fn: Callable[[AgentState], dict],
              terminal: bool = False) -> Callable[[AgentState], dict]:
        def wrapper(state: AgentState) -> dict:
            # Свой счётчик шагов — второй гард после recursion_limit:
            # recursion_limit LangGraph считает супершаги, нужен запас.
            # Терминалы из-под гарда не выводят — escalate должен выполниться,
            # даже если лимит исчерпан в предыдущем узле.
            if not terminal and len(state.get("trace") or []) >= cfg.max_steps:
                return {"error": "max_steps"}
            try:
                delta = fn(state) or {}
            except BudgetExceeded as e:
                logger.info("бюджет исчерпан в узле %s: %s", name, e)
                return {"error": "budget_exceeded"}
            except Exception as e:
                # logger.error (не exception): при дефолтном LOG_LEVEL=ERROR —
                # одна строка, а не traceback в консоли (эскалация, не падение).
                logger.error("узел %s упал: %s", name, e)
                return {"error": f"step_error:{name}"}
            delta = dict(delta)
            trace = list(state.get("trace") or [])
            trace.append(name)
            delta["trace"] = trace
            return delta
        wrapper.__name__ = f"node_{name}"
        return wrapper

    # -- check_memory: вопрос уже был? --------------------------------------
    def node_check_memory(state: AgentState) -> dict:
        best_node: Optional[Node] = None
        best_score = 0.0
        for node in deps.memory.nodes.values():
            if node.type != "qa":
                continue
            s = keyword_score(state["query"], node.label)
            if s > best_score:
                best_node, best_score = node, s
        if best_node is not None and best_score >= cfg.memory_hit_threshold:
            return {"memory_hit": best_node}
        return {}

    # -- classify: классификация LLM (CHEAP модель) --------------------------
    def node_classify(state: AgentState) -> dict:
        classification = _chat_json(deps.chat, CLASSIFY_SYSTEM_PROMPT, state["query"])
        delta: dict[str, Any] = {"classification": classification}
        if classification.get("_parse_error"):
            delta["escalate_reason"] = "classification_failed"
        elif classification.get("risk") == "high":
            delta["escalate_reason"] = "high_risk"
        return delta

    # -- retrieve: векторный поиск по БЗ (retrieval-контроль) -----------------
    def node_retrieve(state: AgentState) -> dict:
        docs_by_id = {d.doc_id: d for d in deps.kb}
        results = deps.qmem.search(state["query"], top_k=3)
        # Косинусное сходство может быть отрицательным — ограничиваем скор
        # до [0, 1]: пороги и выводы работают в фиксированном диапазоне.
        found = [
            (docs_by_id[doc_id], round(min(1.0, max(0.0, score)), 3))
            for doc_id, score, _ in results if doc_id in docs_by_id
        ]
        threshold = cfg.relevance_threshold
        decision = f"top_k=3, найдено={len(found)}"
        if not found:
            # Пустой result — второй проход с более широкой сетью и
            # пониженным порогом (агент управляет retrieval сам).
            results = deps.qmem.search(state["query"], top_k=6)
            found = [
                (docs_by_id[doc_id], round(min(1.0, max(0.0, score)), 3))
                for doc_id, score, _ in results if doc_id in docs_by_id
            ]
            threshold *= 0.5
            decision += f" → второй проход top_k=6, порог {threshold:.2f}"
        logger.info("retrieve: %s", decision)
        docs = [RetrievedDoc(document=d, score=s) for d, s in found]
        best = max((s for _, s in found), default=0.0)
        logger.info("retrieve: лучший скор %.3f (порог %.2f)", best, threshold)
        return {"docs": docs, "relevance_threshold": threshold}

    # -- check_relevance: «контекста достаточно?» -----------------------------
    def node_check_relevance(state: AgentState) -> dict:
        best = max((d.score for d in state.get("docs") or []), default=0.0)
        threshold = state.get("relevance_threshold", cfg.relevance_threshold)
        return {"relevance_ok": best >= threshold}

    # -- generate: ответ LLM (выбор модели CHEAP/ADVANCED) ---------------------
    def node_generate(state: AgentState) -> dict:
        c = state.get("classification") or {}
        best = max((d.score for d in state.get("docs") or []), default=0.0)
        retry = state.get("retry_count") or 0
        # ADVANCED — для сложного вопроса, слабого контекста или перегенерации
        # после провала валидации; в остальном тянет CHEAP.
        if retry > 0 or c.get("complexity") == "complex" or best < cfg.relevance_threshold * 2:
            model, chat_fn = "ADVANCED", deps.chat_advanced
        else:
            model, chat_fn = "CHEAP", deps.chat
        context_block = "\n\n".join(
            f"[{d.document.doc_id}] {d.document.title}\n{d.document.text}"
            for d in state.get("docs") or []
        )
        user = f"Контекст:\n{context_block}\n\nВопрос пользователя:\n{state['query']}"
        if retry > 0 and (state.get("validation") or {}).get("reason"):
            user += (
                "\n\nПредыдущий ответ не прошёл валидацию. "
                f"Комментарий валидатора: {state['validation']['reason']}"
            )
        return {"answer": chat_fn(ANSWER_SYSTEM_PROMPT, user), "model_used": model}

    # -- check_answer: детерминированная проверка ответа (без LLM) -------------
    def node_check_answer(state: AgentState) -> dict:
        reason = _answer_problem(state.get("answer", ""), known_doc_ids,
                                 cfg.max_answer_chars)
        if reason:
            # Провал оформляем как валидационный — ретраи работают как обычно,
            # а причина финальной эскалации будет answer_check_failed.
            return {"validation": {"grounded": False, "reason": f"проверка ответа: {reason}"},
                    "validation_source": "check_answer"}
        return {"validation_source": ""}

    # -- validate: grounding-проверка (CHEAP модель) ----------------------------
    def node_validate(state: AgentState) -> dict:
        context_block = "\n\n".join(d.document.text for d in state.get("docs") or [])
        user = f"Контекст:\n{context_block}\n\nОтвет для проверки:\n{state.get('answer', '')}"
        validation = _chat_json(deps.chat, VALIDATE_SYSTEM_PROMPT, user)
        return {"validation": validation, "validation_source": ""}

    # -- check_validation: «ответ подтверждён?» (инкремент retry — в теле!) -----
    def node_check_validation(state: AgentState) -> dict:
        validation = state.get("validation") or {}
        if validation.get("grounded") is True:
            return {}
        can_retry = (state.get("retry_count") or 0) < cfg.max_validation_retries
        if can_retry:
            return {"can_retry": True, "retry_count": (state.get("retry_count") or 0) + 1}
        source = state.get("validation_source") or ""
        return {
            "can_retry": False,
            "escalate_reason": "answer_check_failed" if source == "check_answer"
                               else "grounding_validation_failed",
        }

    def _needs_ticket(state: AgentState) -> bool:
        """Тикет нужен, когда риск medium: жалоба с требованием, которая
        требует решения человека (high уже эскалирован в classify)."""
        return (state.get("classification") or {}).get("risk") == "medium"

    # -- tool_create_ticket: function calling (two-step, порт DZ_2) -------------
    def node_tool_create_ticket(state: AgentState) -> dict:
        c = state.get("classification") or {}
        docs = state.get("docs") or []
        doc_ids = ", ".join(d.document.doc_id for d in docs) or "не указаны"
        user = (
            f"Запрос пользователя:\n{state['query']}\n\n"
            f"Ответ агента:\n{state.get('answer', '')}\n\n"
            f"Классификация: category={c.get('category')}, risk={c.get('risk')}\n"
            f"Контекст из БЗ (doc_id): {doc_ids}"
        )
        args: dict[str, Any] = {}
        max_attempts = max(1, cfg.max_tool_steps)
        for attempt in range(1, max_attempts + 1):
            response = deps.chat(TICKET_SYSTEM_PROMPT, user)
            name, raw_args = parse_tool_response(response)
            if name is None:
                errors = ["ответ не является вызовом create_ticket (ожидался JSON "
                          "с name='create_ticket' и arguments)"]
            else:
                args, errors = validate_tool_call(name, raw_args)
            if not errors:
                break
            logger.warning("create_ticket: невалидные аргументы (попытка %d): %s",
                           attempt, "; ".join(errors))
            if attempt >= max_attempts:
                return {"error": "tool_error:create_ticket",
                        "escalate_reason": f"tool_error:create_ticket: "
                                           f"невалидные аргументы после {attempt} попыток"}
            # Ошибка возвращается LLM текстом — повторный вызов с исправлением.
            user += ("\n\nТвои аргументы не прошли валидацию по схеме. Ошибки: "
                     + "; ".join(errors) + ". Исправь аргументы и верни JSON ещё раз.")
        try:
            payload, tool_retries = create_ticket_call(
                args, deps.ticket_url, deps.ticket_timeout, deps.ticket_backoff)
        except TicketServiceError as e:
            # Тикет считается несозданным: эскалация с причиной, не падение.
            logger.error("create_ticket: сбой сервиса: %s", e)
            return {"error": "tool_error:create_ticket",
                    "escalate_reason": f"tool_error:create_ticket: {e}"}
        logger.info("create_ticket: создан тикет %s", payload.get("ticket_id"))
        return {"ticket": payload, "tool_retries": tool_retries}

    # -- save: сохранение в память ---------------------------------------------
    def node_save(state: AgentState) -> dict:
        node_id = _next_qa_id(deps.memory)
        deps.memory.add_node(Node(node_id, state["query"], "qa", state.get("answer", "")))
        for rd in state.get("docs") or []:
            if rd.document.doc_id in deps.memory.nodes:
                deps.memory.add_edge(Edge(node_id, rd.document.doc_id, "использует"))
        save_qa_graph(deps.memory, deps.qa_path)
        return {"memory_saved": True}

    # -- Терминальные узлы -------------------------------------------------------
    def node_finish(state: AgentState) -> dict:
        ticket = state.get("ticket")
        sources = [d.document.doc_id for d in state.get("docs") or []]
        if ticket:
            # Ответ готов, но запрос передан специалисту — исход escalated,
            # id тикета присутствует в сообщении (проверяет check-логика).
            return {"result": {
                "outcome": "escalated",
                "message": f"{state.get('answer', '')}\n\n"
                           f"Ваш запрос передан специалисту: тикет {ticket.get('ticket_id')}.",
                "sources": sources,
                "escalated_reason": "ticket_created",
                "memory_saved": bool(state.get("memory_saved")),
            }}
        return {"result": {
            "outcome": "answered",
            "message": state.get("answer", ""),
            "sources": sources,
            "memory_saved": bool(state.get("memory_saved")),
        }}

    def node_finish_cached(state: AgentState) -> dict:
        hit = state["memory_hit"]
        sources = [e.to_id for e in deps.memory.outgoing.get(hit.id, [])
                   if e.relation == "использует"]
        return {"result": {
            "outcome": "answered_cached",
            "message": f"(ответ из памяти) {hit.text}",
            "sources": sources,
            "memory_saved": False,
        }}

    def node_refuse(state: AgentState) -> dict:
        return {"result": {
            "outcome": "refused",
            "message": "В базе знаний не нашлось релевантной информации. "
                       "Попробуйте переформулировать вопрос или обратитесь к специалисту.",
            "memory_saved": False,
        }}

    def node_escalate(state: AgentState) -> dict:
        reason = state.get("error") or state.get("escalate_reason") or "unknown"
        return {"result": {
            "outcome": "escalated",
            "message": "Передаю ваш запрос оператору поддержки.",
            "escalated_reason": reason,
            "memory_saved": False,
        }}

    # -- Роутеры ------------------------------------------------------------------
    def _err_or(default: str) -> Callable[[AgentState], str]:
        """Сквозной error-роутинг: fatal-ошибка узла → escalate."""
        def route(state: AgentState) -> str:
            if state.get("error"):
                return "escalate"
            return default
        return route

    def route_after_check_memory(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        return "finish_cached" if state.get("memory_hit") is not None else "classify"

    def route_after_classify(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        c = state.get("classification") or {}
        if c.get("_parse_error") or c.get("risk") == "high":
            return "escalate"
        return "retrieve"

    def route_after_check_relevance(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        return "generate" if state.get("relevance_ok") else "refuse"

    def route_after_check_answer(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        return "check_validation" if state.get("validation_source") == "check_answer" \
            else "validate"

    def route_after_check_validation(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        if (state.get("validation") or {}).get("grounded") is not True:
            # can_retry/retry_count установлены в теле node_check_validation.
            return "generate" if state.get("can_retry") else "escalate"
        return "tool_create_ticket" if _needs_ticket(state) else "save"

    def route_after_tool(state: AgentState) -> str:
        if state.get("error"):
            return "escalate"
        return "save"

    # -- Композиция ----------------------------------------------------------------
    g = StateGraph(AgentState)
    working_nodes = {
        "check_memory": node_check_memory,
        "classify": node_classify,
        "retrieve": node_retrieve,
        "check_relevance": node_check_relevance,
        "generate": node_generate,
        "check_answer": node_check_answer,
        "validate": node_validate,
        "check_validation": node_check_validation,
        "tool_create_ticket": node_tool_create_ticket,
        "save": node_save,
    }
    terminal_nodes = {
        "finish": node_finish,
        "finish_cached": node_finish_cached,
        "refuse": node_refuse,
        "escalate": node_escalate,
    }
    for name, fn in working_nodes.items():
        g.add_node(name, _safe(name, fn))
    for name, fn in terminal_nodes.items():
        g.add_node(name, _safe(name, fn, terminal=True))

    g.add_edge(START, "check_memory")
    g.add_conditional_edges("check_memory", route_after_check_memory,
                            {"classify": "classify", "finish_cached": "finish_cached",
                             "escalate": "escalate"})
    g.add_conditional_edges("classify", route_after_classify,
                            {"retrieve": "retrieve", "escalate": "escalate"})
    g.add_conditional_edges("retrieve", _err_or("check_relevance"),
                            {"check_relevance": "check_relevance", "escalate": "escalate"})
    g.add_conditional_edges("check_relevance", route_after_check_relevance,
                            {"generate": "generate", "refuse": "refuse",
                             "escalate": "escalate"})
    g.add_conditional_edges("generate", _err_or("check_answer"),
                            {"check_answer": "check_answer", "escalate": "escalate"})
    g.add_conditional_edges("check_answer", route_after_check_answer,
                            {"validate": "validate", "check_validation": "check_validation",
                             "escalate": "escalate"})
    g.add_conditional_edges("validate", _err_or("check_validation"),
                            {"check_validation": "check_validation", "escalate": "escalate"})
    g.add_conditional_edges("check_validation", route_after_check_validation,
                            {"save": "save", "tool_create_ticket": "tool_create_ticket",
                             "generate": "generate", "escalate": "escalate"})
    g.add_conditional_edges("tool_create_ticket", route_after_tool,
                            {"save": "save", "escalate": "escalate"})
    g.add_conditional_edges("save", _err_or("finish"),
                            {"finish": "finish", "escalate": "escalate"})
    for terminal in ("finish", "finish_cached", "refuse", "escalate"):
        g.add_edge(terminal, END)

    return g.compile()


def _final_state_to_result(final: AgentState) -> AgentResult:
    """Собирает AgentResult из финального состояния графа."""
    r = final.get("result") or {}
    return AgentResult(
        outcome=Outcome(r.get("outcome", "escalated")),
        message=r.get("message", ""),
        sources=list(r.get("sources") or []),
        escalated_reason=r.get("escalated_reason"),
        memory_saved=bool(r.get("memory_saved", False)),
        trace=list(final.get("trace") or []),
    )


def _execute(query: str, graph, meter: MeteredChat, max_steps: int
             ) -> tuple[AgentResult, RunMetrics]:
    """Один прогон графа с метриками: время, LLM-вызовы, токены, ретраи.

    Двойная защита от зацикливания: recursion_limit (супершаги LangGraph,
    с запасом +10) и свой счётчик trace (гард max_steps в обёртке узла).
    """
    metrics = meter.new_run()
    start = time.monotonic()
    try:
        final = graph.invoke(
            {"query": query, "trace": []},
            config={"recursion_limit": max_steps + 10},
        )
        result = _final_state_to_result(final)
        metrics.model_used = final.get("model_used") or ""
        metrics.tool_calls = 1 if final.get("ticket") else 0
        metrics.tool_retries = int(final.get("tool_retries") or 0)
    except GraphRecursionError:
        result = AgentResult(
            outcome=Outcome.ESCALATED,
            message="Сценарий не завершился за отведённое число шагов, передаю оператору.",
            escalated_reason="max_steps",
        )
    metrics.duration_s = round(time.monotonic() - start, 3)
    metrics.outcome = result.outcome
    if result.escalated_reason == "budget_exceeded":
        metrics.budget_violated = True
    return result, metrics


# --------------------------------------------------------------------------- #
# Лог выполнения (JSONL, порт DZ_6)
# --------------------------------------------------------------------------- #

class RunLog:
    """Лог выполнения: JSONL, одна JSON-строка на прогон, append-режим.

    Ротация: когда записей становится >= max_records, файл переименовывается
    в <файл>.1 (предыдущий архив не сохраняется), и лог начинается заново —
    runs.jsonl не растёт бесконечно. max_records <= 0 — ротация выключена.
    """

    def __init__(self, path: str, max_records: int = RUNS_LOG_MAX_RECORDS) -> None:
        self.path = path
        self.max_records = max_records

    def append(self, record: dict[str, Any]) -> None:
        self._rotate_if_needed()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self) -> None:
        if self.max_records <= 0 or not os.path.exists(self.path):
            return
        if len(self.read()) < self.max_records:
            return
        archive = self.path + ".1"
        if os.path.exists(archive):
            os.remove(archive)
        os.replace(self.path, archive)

    def read(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("некорректная строка в логе: %s", line[:100])
        return records


def log_run(run_log: RunLog, query: str, result: AgentResult,
            metrics: RunMetrics) -> dict[str, Any]:
    """Формирует запись о прогоне и дописывает её в лог выполнения."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "outcome": result.outcome.value,
        "escalated_reason": result.escalated_reason,
        "trace": result.trace,
        "duration_s": metrics.duration_s,
        "llm_calls": metrics.llm_calls,
        "tokens": {
            "prompt": metrics.prompt_tokens,
            "completion": metrics.completion_tokens,
            "total": metrics.total_tokens,
        },
        "cost_rub": metrics.cost_rub,
        "retries": metrics.retries,
        "model_used": metrics.model_used or None,
        "tool_calls": metrics.tool_calls,
        "tool_retries": metrics.tool_retries,
        "memory_saved": result.memory_saved,
        "sources": result.sources,
    }
    run_log.append(record)
    return record


# --------------------------------------------------------------------------- #
# Вывод и CLI
# --------------------------------------------------------------------------- #

def print_result(result: AgentResult, show_trace: bool = True) -> None:
    if show_trace and result.trace:
        print(f"Путь по узлам: {' → '.join(result.trace)}")
    print(f"[{result.outcome.value}]")
    print(result.message)
    if result.sources:
        print(f"Источники: {', '.join(result.sources)}")
    if result.escalated_reason:
        print(f"Причина: {result.escalated_reason}")
    if result.memory_saved:
        print("Сохранено в память (qa.json).")


def print_metrics(metrics: RunMetrics) -> None:
    """Одна строка метрик прогона (для одиночного режима и интерактива)."""
    line = (
        f"Метрики: {metrics.duration_s:.2f}с | LLM-вызовов: {metrics.llm_calls} | "
        f"токены: {metrics.total_tokens} (prompt={metrics.prompt_tokens}, "
        f"completion={metrics.completion_tokens}) | стоимость: {metrics.cost_rub:.4f}₽ "
        f"| ретраев: {metrics.retries}"
    )
    if metrics.model_used:
        line += f" | модель: {metrics.model_used}"
    if metrics.tool_calls:
        line += f" | инструментов: {metrics.tool_calls}"
    print(line)


def print_metrics_table(runs: list[tuple[str, AgentResult, RunMetrics]]) -> None:
    """Сводная таблица «прогон × метрики» + агрегаты + алерт (конец --demo)."""
    if not runs:
        return
    print("\n===== Сводка метрик =====")
    print(f"{'№':<4}{'результат':<18}{'LLM':>5}{'токены':>9}{'время':>10}{'ретраи':>9}"
          f"{'стоимость':>12}  комментарий")
    for i, (comment, result, m) in enumerate(runs, 1):
        note = result.escalated_reason or comment
        print(
            f"{i:<4}{result.outcome.value:<18}{m.llm_calls:>5}{m.total_tokens:>9}"
            f"{m.duration_s:>9.2f}с{m.retries:>9}{m.cost_rub:>11.4f}₽  {note}"
        )
    total = len(runs)
    ok = sum(1 for _, r, _ in runs if r.outcome in (Outcome.ANSWERED, Outcome.ANSWERED_CACHED))
    refused = sum(1 for _, r, _ in runs if r.outcome == Outcome.REFUSED)
    errors = sum(1 for _, r, _ in runs if r.outcome == Outcome.ESCALATED)
    avg = sum(m.duration_s for _, _, m in runs) / total
    tokens = sum(m.total_tokens for _, _, m in runs)
    cost = sum(m.cost_rub for _, _, m in runs)
    retries = sum(m.retries for _, _, m in runs)
    print(f"Успешность: {ok}/{total} ({ok * 100 / total:.1f}%) | отказов: {refused} | ошибок: {errors}")
    print(f"Среднее время: {avg:.2f}с | суммарные токены: {tokens} | "
          f"стоимость: {cost:.4f}₽ | ретраев: {retries}")
    alert = check_error_alert(errors, total)
    if alert:
        print(alert)


def check_error_alert(escalated: int, total: int,
                      threshold: float = ERROR_ALERT_THRESHOLD) -> Optional[str]:
    """Алерт: доля прогонов, завершившихся эскалацией, выше порога.

    Порог по умолчанию 0.2 (20%): например, в демо из 4 прогонов одна
    запланированная эскалация (create_ticket) — 25% — уже превышает порог.
    """
    if total == 0 or threshold <= 0:
        return None
    share = escalated / total
    if share > threshold:
        return (
            f"[АЛЕРТ] {share * 100:.1f}% прогонов завершилось эскалацией "
            f"(порог {threshold * 100:.0f}%) — проверьте LLM и конфигурацию"
        )
    return None


def report(path: str) -> int:
    """Агрегированные метрики по логам выполнения (--report)."""
    records = RunLog(path).read()
    if not records:
        print(f"Лог «{path}» пуст или не найден. Сначала выполните прогон: agent.py --demo")
        return 1
    total = len(records)
    by_outcome: dict[str, list[dict]] = {}
    for rec in records:
        by_outcome.setdefault(str(rec.get("outcome", "unknown")), []).append(rec)
    ok = sum(len(v) for k, v in by_outcome.items() if k in ("answered", "answered_cached"))
    print(f"Отчёт по логам выполнения: {path} (всего {total} прогонов)")
    print(f"{'результат':<18}{'прогонов':>9}{'доля':>9}{'среднее время':>16}"
          f"{'токенов':>11}{'стоимость':>13}")
    for outcome in ("answered", "answered_cached", "refused", "escalated"):
        recs = by_outcome.get(outcome)
        if not recs:
            continue
        avg = sum(r.get("duration_s", 0) for r in recs) / len(recs)
        tokens = sum(r.get("tokens", {}).get("total", 0) for r in recs)
        cost = sum(r.get("cost_rub", 0) for r in recs)
        print(f"{outcome:<18}{len(recs):>9}{len(recs) * 100 / total:>8.1f}%"
              f"{avg:>15.2f}с{tokens:>11}{cost:>12.4f}₽")
    for other, recs in by_outcome.items():
        if other in ("answered", "answered_cached", "refused", "escalated"):
            continue
        print(f"{other:<18}{len(recs):>9}{len(recs) * 100 / total:>8.1f}%"
              f"{'—':>16}{'—':>11}{'—':>13}")
    total_dur = sum(r.get("duration_s", 0) for r in records)
    total_tok = sum(r.get("tokens", {}).get("total", 0) for r in records)
    total_ret = sum(r.get("retries", 0) for r in records)
    total_cost = sum(r.get("cost_rub", 0) for r in records)
    budget = sum(1 for r in records if r.get("escalated_reason") == "budget_exceeded")
    tickets = sum(int(r.get("tool_calls", 0)) for r in records)
    escalated = sum(1 for r in records if r.get("outcome") == "escalated")
    print(f"Успешность: {ok}/{total} ({ok * 100 / total:.1f}%)")
    print(f"Суммарно: время {total_dur:.2f}с, токенов {total_tok}, стоимость {total_cost:.4f}₽, "
          f"ретраев {total_ret}, эскалаций по бюджету {budget}, тикетов создано: {tickets}")
    alert = check_error_alert(escalated, total)
    if alert:
        print(alert)
    return 0


DEMO_QUERIES: list[tuple[str, str]] = [
    ("Как вернуть товар, если он не подошёл?",
     "happy path: полный путь + сохранение в память"),
    ("Какая погода в Токио?",
     "нет контекста → ветка refuse"),
    ("Возврат денег по заказу 4821 тянется две недели, менеджер меня игнорирует, "
     "готовлю претензию",
     "средний риск → инструмент create_ticket + тикет"),
    ("Как вернуть товар, если он не подошёл?",
     "повтор вопроса 1 → хит в памяти"),
]


def run_one(graph, query: str, show_trace: bool,
            meter: MeteredChat, run_log: RunLog,
            max_steps: int = MAX_STEPS
            ) -> Optional[tuple[AgentResult, RunMetrics]]:
    """Одиночный прогон: граф + вывод + строка метрик + запись в лог."""
    try:
        result, metrics = _execute(query, graph, meter, max_steps)
    except openai.APIError as e:
        print(f"[ошибка LLM] {e.__class__.__name__}: {str(e)[:300]}")
        return None
    print_result(result, show_trace)
    print_metrics(metrics)
    log_run(run_log, query, result, metrics)
    return result, metrics


def run_demo(graph, meter: MeteredChat, run_log: RunLog) -> None:
    print(f"Демо: {len(DEMO_QUERIES)} прогона (все ветки графа)")
    runs: list[tuple[str, AgentResult, RunMetrics]] = []
    for i, (query, comment) in enumerate(DEMO_QUERIES, 1):
        print(f"\n===== Прогон {i}/{len(DEMO_QUERIES)}: {comment} =====")
        print(f"Запрос: {query}")
        one = run_one(graph, query, show_trace=True, meter=meter, run_log=run_log)
        if one is not None:
            runs.append((comment, one[0], one[1]))
    print_metrics_table(runs)


def chat_loop(graph, meter: MeteredChat, run_log: RunLog) -> None:
    print("Интерактивный режим. Выход: 'exit'/'quit'/'выход' или Ctrl-D.")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "выход"):
            break
        run_one(graph, q, show_trace=True, meter=meter, run_log=run_log)


def _qa_count(memory: GraphMemory) -> int:
    return sum(1 for n in memory.nodes.values() if n.type == "qa")


# --------------------------------------------------------------------------- #
# Самодиагностика (без LLM, без Qdrant, без сети)
# --------------------------------------------------------------------------- #

HAPPY_QUERY = "Как вернуть товар, если он не подошёл?"
REFUSE_QUERY = "Какая погода в Токио?"
RISKY_QUERY = "Я хочу на вас подать в суд и причинить вред себе."
TICKET_QUERY = "Возврат денег по заказу 4821 тянется две недели, менеджер меня игнорирует, " \
               "готовлю претензию"


# In-process тикет-серверы selftest: закрываются разом в конце selftest(),
# иначе каждая среда оставляет daemon-поток с слушателем.
_selftest_servers: list[Any] = []


def _selftest_env(mode: str, config: Optional[ScenarioConfig] = None,
                  max_calls: int = MAX_LLM_CALLS_PER_RUN,
                  ticket_url: Optional[str] = None) -> dict[str, Any]:
    """Свежая среда для проверки: БЗ, tmp-память, мок Qdrant (TF-IDF), FakeLLM
    и in-process мок тикет-сервиса на эфемерном порту.

    Лог выполнения в selftest не ведётся (кроме отдельной проверки, где он
    создаётся в tmp-каталоге) — реальный logs/runs.jsonl не трогает.
    """
    import ticket_server
    kb = load_documents(_resolve(KB_FILE))
    qa_dir = tempfile.mkdtemp(prefix="dz7-case1-selftest-")
    qa_path = os.path.join(qa_dir, "qa.json")
    memory = build_runtime_memory(kb, qa_path)
    # Мок Qdrant + детерминированный TF-IDF эмбеддер: поиск осмыслен, сети нет.
    embedder = TfidfEmbedder([f"{d.title} {d.text}" for d in kb])
    qmem = MockQdrantMemory(embedder.embed)
    qmem.upsert_documents(kb)
    raw = FakeLLM(mode)
    # Две модели: ADVANCED — отдельный FakeLLM того же режима, общий guard.
    raw_advanced = FakeLLM(mode)
    meter = MeteredChat(raw, max_retries=2)
    meter.guard.max_calls = max_calls
    meter_advanced = MeteredChat(raw_advanced, max_retries=2, guard=meter.guard)
    cfg = config or ScenarioConfig()
    # Мок тикет-сервиса in-process: тот же handler, эфемерный порт, tmp-файл.
    server = None
    tickets_path = os.path.join(tempfile.mkdtemp(prefix="dz7-case1-tickets-"), "tickets.json")
    if ticket_url is None:
        ticket_server.TICKETS_PATH = tickets_path
        server = ticket_server.make_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _selftest_servers.append(server)
        ticket_url = f"http://127.0.0.1:{server.server_address[1]}"
    deps = Deps(
        chat=meter, chat_advanced=meter_advanced, kb=kb, memory=memory, qmem=qmem,
        qa_path=qa_path, config=cfg, ticket_url=ticket_url, ticket_timeout=3.0,
        ticket_backoff=(0.0, 0.0),
    )
    graph = build_graph(deps)
    return {"chat": raw, "meter": meter, "memory": memory, "qmem": qmem,
            "qa_path": qa_path, "graph": graph, "config": cfg,
            "server": server, "tickets_path": tickets_path}


def _load_qa_file(qa_path: str) -> dict[str, Any]:
    # Файл появляется только после первого сохранения — «нет файла» = пустая память.
    if not os.path.exists(qa_path):
        return {"nodes": [], "edges": []}
    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tickets_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("tickets", [])


def _gen_call_count(chat: FakeLLM) -> int:
    """Сколько раз FakeLLM генерировала ответ (системный промпт агента)."""
    return sum(1 for s, _ in chat.calls if "агент поддержки" in s)


def selftest() -> int:
    """12 проверок без LLM, без Qdrant и без сети
    (FakeLLM + мок Qdrant + TF-IDF + in-process тикет-сервис)."""
    failures: list[str] = []

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e!r}")
            failures.append(name)

    # 1. Граф собирается, все рёбра валидны, все 14 узлов достижимы.
    def t1_graph_valid() -> None:
        env = _selftest_env("default")
        env["graph"].get_graph().draw_ascii()  # падёт, если граф некорректен
        all_nodes = {n for n in env["graph"].nodes if not n.startswith("__")}
        assert len(all_nodes) == 14, f"ожидается 14 узлов, а их {len(all_nodes)}"
        results = [
            _execute(HAPPY_QUERY, env["graph"], env["meter"], env["config"].max_steps),
            _execute(HAPPY_QUERY, env["graph"], env["meter"], env["config"].max_steps),
        ]
        results.append(_execute(REFUSE_QUERY, env["graph"], env["meter"],
                                env["config"].max_steps))
        for mode, query in [("risky", RISKY_QUERY), ("medium", TICKET_QUERY)]:
            e2 = _selftest_env(mode)
            results.append(_execute(query, e2["graph"], e2["meter"],
                                    e2["config"].max_steps))
        visited: set[str] = set()
        for result, _ in results:
            for node_name in result.trace:
                assert node_name in all_nodes, f"переход в неизвестный узел {node_name}"
            visited.update(result.trace)
        missing = all_nodes - visited
        assert not missing, f"недостижимые узлы: {sorted(missing)}"

    # 2. Честный путь answered: полный trace, [источники] из БЗ, qa.json записан.
    def t2_success_path() -> None:
        env = _selftest_env("default")
        result, metrics = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                                   env["config"].max_steps)
        expected_trace = [
            "check_memory", "classify", "retrieve", "check_relevance",
            "generate", "check_answer", "validate", "check_validation",
            "save", "finish",
        ]
        assert result.trace == expected_trace, f"trace: {result.trace}"
        assert result.outcome == Outcome.ANSWERED, result.outcome
        assert result.memory_saved is True
        kb_ids = {d.doc_id for d in load_documents(_resolve(KB_FILE))}
        assert result.sources, "источники пустые"
        assert set(result.sources) <= kb_ids, f"sources не из БЗ: {result.sources}"
        assert "doc-return" in result.sources, f"sources: {result.sources}"
        assert metrics.success is True and metrics.outcome == Outcome.ANSWERED
        expected_cost = round(
            metrics.prompt_tokens * LLM_COST_INPUT_PER_1M_RUB / 1_000_000
            + metrics.completion_tokens * LLM_COST_OUTPUT_PER_1M_RUB / 1_000_000,
            6,
        )
        assert metrics.cost_rub == expected_cost, (
            f"стоимость: {metrics.cost_rub} != {expected_cost}"
        )
        qa_data = _load_qa_file(env["qa_path"])
        qa_nodes = [n for n in qa_data["nodes"] if n["type"] == "qa"]
        assert len(qa_nodes) == 1, f"qa-узлы: {qa_data['nodes']}"
        assert any(
            e["from"] == qa_nodes[0]["id"] and e["to"] == "doc-return"
            for e in qa_data["edges"]
        ), f"ребро qa→doc-return не найдено: {qa_data['edges']}"

    # 3. Ветка «нет контекста»: отказ, память не пополняется;
    #    отрицательный косинус обрезается до [0, 1].
    def t3_refuse_branch() -> None:
        env = _selftest_env("default")
        result, _ = _execute(REFUSE_QUERY, env["graph"], env["meter"],
                             env["config"].max_steps)
        assert result.outcome == Outcome.REFUSED, result.outcome
        assert result.trace == [
            "check_memory", "classify", "retrieve", "check_relevance", "refuse",
        ], f"trace: {result.trace}"
        assert result.memory_saved is False
        assert _load_qa_file(env["qa_path"])["nodes"] == [], "память была пополнена"
        # Отрицательный косинус (теоретически возможен): скор обрезают до
        # [0, 1], ветка — refuse.
        kb = load_documents(_resolve(KB_FILE))
        class _NegScoreMem:
            def search(self, query, top_k):
                return [(kb[0].doc_id, -0.6, {}), (kb[1].doc_id, -0.1, {})]
        deps = Deps(chat=env["meter"], chat_advanced=env["meter"], kb=kb,
                    memory=env["memory"], qmem=_NegScoreMem(),
                    qa_path=env["qa_path"], config=env["config"],
                    ticket_url="http://127.0.0.1:1", ticket_timeout=3.0)
        final = build_graph(deps).invoke(
            {"query": REFUSE_QUERY, "trace": []},
            config={"recursion_limit": env["config"].max_steps + 10},
        )
        r = final.get("result") or {}
        assert r.get("outcome") == Outcome.REFUSED.value, r
        assert final.get("docs"), "документы не сохранены в state"
        for d in final["docs"]:
            assert 0.0 <= d.score <= 1.0, f"score {d.score} не в [0, 1]"

    # 4. Высокий риск: эскалация детерминированно после классификации.
    def t4_high_risk() -> None:
        env = _selftest_env("risky")
        result, _ = _execute(RISKY_QUERY, env["graph"], env["meter"],
                             env["config"].max_steps)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "high_risk", result.escalated_reason
        assert result.trace == ["check_memory", "classify", "escalate"], f"trace: {result.trace}"

    # 5. check_answer: выдуманный [doc_id] → ретраи → answer_check_failed,
    #    LLM-валидатор не вызывается.
    def t5_check_answer() -> None:
        env = _selftest_env("badformat")
        result, metrics = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                                   env["config"].max_steps)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "answer_check_failed", result.escalated_reason
        assert "validate" not in result.trace, f"LLM-валидатор не должен вызываться: {result.trace}"
        # classify + 3×generate (check_answer детерминированно отвергает каждый раз).
        assert metrics.llm_calls == 4, f"вызовов: {metrics.llm_calls}"

    # 6. finish_cached: повторный вопрос без LLM-вызовов.
    def t6_memory_hit() -> None:
        env = _selftest_env("default")
        first, _ = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                            env["config"].max_steps)
        assert first.outcome == Outcome.ANSWERED and first.memory_saved
        gen_calls_1 = _gen_call_count(env["chat"])
        second, metrics2 = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                                    env["config"].max_steps)
        assert second.outcome == Outcome.ANSWERED_CACHED, second.outcome
        assert second.trace == ["check_memory", "finish_cached"], f"trace: {second.trace}"
        assert _gen_call_count(env["chat"]) == gen_calls_1, "сделана новая генерация при хите"
        assert metrics2.llm_calls == 0, "хит в памяти не должен вызывать LLM"
        assert second.sources == first.sources, f"sources: {second.sources} != {first.sources}"

    # 7. Цикл generate→check_answer→check_validation→generate гасится MAX_STEPS.
    def t7_step_guard() -> None:
        config = ScenarioConfig(max_validation_retries=10, max_steps=8)
        env = _selftest_env("loop", config)
        result, _ = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                             config.max_steps)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "max_steps", result.escalated_reason
        assert len(result.trace) <= config.max_steps + 1, f"trace: {result.trace}"

    # 8. Бюджет: BudgetExceeded → escalate_reason=budget_exceeded, ровно 4 вызова.
    def t8_budget() -> None:
        env = _selftest_env("ungrounded", max_calls=4)
        log_path = os.path.join(tempfile.mkdtemp(prefix="dz7-case1-selftest-log-"), "runs.jsonl")
        run_log = RunLog(log_path)
        result, metrics = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                                   env["config"].max_steps)
        log_run(run_log, HAPPY_QUERY, result, metrics)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "budget_exceeded", result.escalated_reason
        assert metrics.llm_calls == 4, f"вызовов: {metrics.llm_calls}"
        assert metrics.budget_violated is True
        records = run_log.read()
        assert records and records[-1]["llm_calls"] == 4, f"лог: {records}"
        assert "cost_rub" in records[-1], f"в записи нет стоимости: {records[-1]}"

    # 9. MeteredChat: retry на flaky (2 сетевые ошибки → успех, retries=2).
    def t9_retry() -> None:
        env = _selftest_env("flaky")
        result, metrics = _execute(HAPPY_QUERY, env["graph"], env["meter"],
                                   env["config"].max_steps)
        assert result.outcome == Outcome.ANSWERED, result.outcome
        assert metrics.retries == 2, f"ретраев: {metrics.retries}"
        # 3 попытки классификации + generate + validate.
        assert metrics.llm_calls == 5, f"вызовов: {metrics.llm_calls}"

    # 10. Тикет-инструмент: (a) невалидные аргументы LLM → повтор с ошибками →
    #     валидный тикет; (b) сервис down → tool_error + эскалация (не падение).
    def t10_ticket_tool() -> None:
        # (a) tool_bad: первый аргумент невалидный (priority="urgent"), второй — нет.
        env = _selftest_env("tool_bad")
        result, metrics = _execute(TICKET_QUERY, env["graph"], env["meter"],
                                   env["config"].max_steps)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "ticket_created", result.escalated_reason
        assert "тикет TKT-" in result.message, f"ticket id не в ответе: {result.message}"
        tickets = _load_tickets_file(env["tickets_path"])
        assert len(tickets) == 1, f"тикетов в файле: {tickets}"
        assert tickets[0]["priority"] == "normal", f"args: {tickets[0]}"
        # classify + generate + validate + 2 попытки аргументов.
        assert metrics.llm_calls == 5, f"вызовов: {metrics.llm_calls}"
        assert metrics.tool_calls == 1, f"tool_calls: {metrics.tool_calls}"
        # (b) Сервис недоступен → сетевые повторы → эскалация tool_error, не падение.
        down = _selftest_env("medium", ticket_url="http://127.0.0.1:1")
        result2, metrics2 = _execute(TICKET_QUERY, down["graph"], down["meter"],
                                     down["config"].max_steps)
        assert result2.outcome == Outcome.ESCALATED, result2.outcome
        assert (result2.escalated_reason or "").startswith("tool_error:create_ticket"), \
            f"reason: {result2.escalated_reason}"
        assert _load_tickets_file(down["tickets_path"]) == [], "тикет не должен создаться"
        assert metrics2.tool_calls == 0, f"tool_calls: {metrics2.tool_calls}"
        # (c) HTTP 5xx от тикет-сервиса → TicketServiceError сразу, без ретраев.
        http_calls = 0
        orig_post = globals()["_http_post_json"]

        def _http_500(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            nonlocal http_calls
            http_calls += 1
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)

        globals()["_http_post_json"] = _http_500
        try:
            try:
                create_ticket_call({"subject": "s", "description": "d"},
                                    "http://127.0.0.1:1", timeout=1.0,
                                    backoff_s=(0.0, 0.0))
                raise AssertionError("HTTP 500 не превратился в TicketServiceError")
            except TicketServiceError as exc:
                assert str(exc).startswith("HTTP 500"), f"причина: {exc}"
        finally:
            globals()["_http_post_json"] = orig_post
        assert http_calls == 1, f"5xx не должен ретраиться: попыток {http_calls}"

    # 11. RunLog: запись + ротация во временном каталоге.
    def t11_log_rotation() -> None:
        log_path = os.path.join(tempfile.mkdtemp(prefix="dz7-case1-selftest-rot-"), "runs.jsonl")
        run_log = RunLog(log_path, max_records=3)
        for i in range(4):
            run_log.append({"i": i})
        assert [r["i"] for r in run_log.read()] == [3], "после ротации файл не новый"
        assert [r["i"] for r in RunLog(log_path + ".1").read()] == [0, 1, 2], "архив не полный"
        # Вторая ротация: архив заменяется, а не накапливается.
        for i in range(4, 7):
            run_log.append({"i": i})
        assert [r["i"] for r in run_log.read()] == [6], "вторая ротация не сработала"
        assert [r["i"] for r in RunLog(log_path + ".1").read()] == [3, 4, 5], "архив накопился"
        # Лимит 0 — ротация выключена.
        no_rot = os.path.join(tempfile.mkdtemp(prefix="dz7-case1-selftest-rot-"), "runs.jsonl")
        for i in range(5):
            RunLog(no_rot, max_records=0).append({"i": i})
        assert len(RunLog(no_rot).read()) == 5, "ротация сработала при max_records=0"

    # 12. --report считает агрегаты/алерт по tmp-логу.
    def t12_report() -> None:
        import io
        import contextlib
        assert check_error_alert(0, 4, 0.2) is None
        assert check_error_alert(2, 10, 0.2) is None  # ровно 20% — не «превышен»
        assert check_error_alert(3, 10, 0.2) is not None  # 30% > 20%
        assert check_error_alert(1, 3, 0.2) is not None
        assert check_error_alert(0, 0, 0.2) is None  # пустой лог — алерта нет
        log_path = os.path.join(tempfile.mkdtemp(prefix="dz7-case1-selftest-rep-"), "runs.jsonl")
        for rec in (
            {"ts": "t", "query": "q1", "outcome": "answered", "duration_s": 1.0,
             "tokens": {"total": 100}, "cost_rub": 0.01, "retries": 0, "tool_calls": 0},
            {"ts": "t", "query": "q2", "outcome": "refused", "duration_s": 0.5,
             "tokens": {"total": 50}, "cost_rub": 0.005, "retries": 0, "tool_calls": 0},
            {"ts": "t", "query": "q3", "outcome": "escalated",
             "escalated_reason": "ticket_created", "duration_s": 2.0,
             "tokens": {"total": 150}, "cost_rub": 0.02, "retries": 0, "tool_calls": 1},
        ):
            RunLog(log_path).append(rec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report(log_path)
        out = buf.getvalue()
        assert rc == 0, f"report rc={rc}"
        assert "всего 3 прогонов" in out, out
        assert "[АЛЕРТ]" in out, "алерт не напечатан при 1/3 эскалаций"
        assert "тикетов создано: 1" in out, "агрегат по tool_calls отсутствует"

    check("граф собирается, рёбра валидны, все 14 узлов достижимы", t1_graph_valid)
    check("честный путь answered: полный trace, [источники] из БЗ, qa.json записан", t2_success_path)
    check("ветка «нет контекста»: отказ, память не пополняется", t3_refuse_branch)
    check("высокий риск: эскалация high_risk после классификации, детерминированно", t4_high_risk)
    check("check_answer: выдуманный [doc_id] → ретраи → answer_check_failed", t5_check_answer)
    check("finish_cached: повторный вопрос без LLM-вызовов", t6_memory_hit)
    check("цикл generate→check_answer→check_validation гасится MAX_STEPS", t7_step_guard)
    check("бюджет: BudgetExceeded → budget_exceeded, ровно 4 LLM-вызова", t8_budget)
    check("MeteredChat: retry на flaky (2 сетевые ошибки → успех, retries=2)", t9_retry)
    check("тикет-инструмент: невалидные аргументы → повтор; down → tool_error; 5xx → без ретраев",
          t10_ticket_tool)
    check("RunLog: запись + ротация во временном каталоге", t11_log_rotation)
    check("--report считает агрегаты и алерт по tmp-логу", t12_report)

    # Закрываем in-process тикет-серверы всех сред (daemon-потоки + слушатели).
    for server in _selftest_servers:
        try:
            server.shutdown()
            server.server_close()
        except OSError:
            pass
    _selftest_servers.clear()

    if failures:
        print(f"SELF-TEST: {len(failures)} упало: {', '.join(failures)}")
        return 1
    print("SELF-TEST: все проверки пройдены.")
    return 0


# --------------------------------------------------------------------------- #
# Главная
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="DZ_7, кейс 1: ИИ-агент поддержки на LangGraph "
                    "(RAG + инструмент create_ticket + контрольный слой)")
    parser.add_argument("question", nargs="*", help="одиночный вопрос")
    parser.add_argument("--demo", action="store_true",
                        help="4 прогона, покрывающие все ветки + сводная таблица метрик")
    parser.add_argument("--selftest", action="store_true", help="самодиагностика без LLM")
    parser.add_argument("--report", action="store_true",
                        help="агрегированные метрики по логам runs.jsonl")
    parser.add_argument("--show-trace", action="store_true",
                        help="печатать путь по узлам графа")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.report:
        sys.exit(report(_resolve(RUNS_LOG_FILE)))

    kb_path = _resolve(KB_FILE)
    qa_path = _resolve(QA_MEMORY_FILE)
    kb = load_documents(kb_path)
    memory = build_runtime_memory(kb, qa_path)
    client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    qmem = make_vector_memory(client, kb)
    meter = MeteredChat(make_openai_chat(client, CHEAP_MODEL))
    meter_advanced = MeteredChat(make_openai_chat(client, ADVANCED_MODEL),
                                 guard=meter.guard)
    deps = Deps(
        chat=meter, chat_advanced=meter_advanced, kb=kb, memory=memory, qmem=qmem,
        qa_path=qa_path, config=ScenarioConfig(),
        ticket_url=TICKET_API_URL, ticket_timeout=TICKET_TIMEOUT_SECONDS,
    )
    graph = build_graph(deps)
    run_log = RunLog(_resolve(RUNS_LOG_FILE))
    print(
        f"Агент DZ_7 кейс 1 | БЗ: {len(kb)} документов | память: {_qa_count(memory)} "
        f"обработанных вопросов | модели: CHEAP={CHEAP_MODEL}, ADVANCED={ADVANCED_MODEL}"
    )

    try:
        if args.demo:
            run_demo(graph, meter, run_log)
        elif args.question:
            run_one(graph, " ".join(args.question), show_trace=args.show_trace,
                    meter=meter, run_log=run_log)
        else:
            chat_loop(graph, meter, run_log)
    finally:
        qmem.close()


if __name__ == "__main__":
    main()
