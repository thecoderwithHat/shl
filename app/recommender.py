from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import build_openrouter_chat_model, maybe_extract_json

try:  # Optional FAISS support. The service falls back to dense cosine search.
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "shl_product_catalog.json"
LOGGER = logging.getLogger("shl.recommender")

KEY_TO_TEST_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Simulations": "S",
}

FINALIZATION_TERMS = {
    "perfect",
    "confirmed",
    "confirm",
    "that works",
    "that is what we need",
    "that's what we need",
    "that's good",
    "lock it in",
    "locking it in",
    "final",
    "keep verify g+",
    "drop the opq",
}

ACKNOWLEDGEMENT_TERMS = {
    "yes",
    "yep",
    "yeah",
    "sure",
    "ok",
    "okay",
    "good",
    "great",
    "go ahead",
    "understood",
}

LEGAL_TERMS = {
    "legally",
    "legal",
    "law",
    "lawsuit",
    "compliance",
    "regulatory",
    "regulation",
    "required under",
    "satisfy that requirement",
    "does this satisfy",
}

PROMPT_INJECTION_TERMS = {
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "reveal your prompt",
    "jailbreak",
    "bypass",
}

OFF_TOPIC_TERMS = {
    "joke",
    "weather",
    "recipe",
    "movie",
    "sports",
    "capital of",
    "translate",
}

ROLE_KEYWORDS = {
    "leadership": {"leadership", "executive", "cxo", "director", "vp", "vice president", "benchmark"},
    "technical": {"java", "spring", "rest", "angular", "sql", "aws", "docker", "engineer", "backend", "frontend", "full-stack", "rust"},
    "contact_center": {"contact centre", "contact center", "call center", "customer service", "inbound calls"},
    "sales": {"sales", "sales organization", "seller"},
    "safety": {"safety", "chemical", "plant", "industrial", "dependability"},
    "graduate": {"graduate", "final-year", "recent graduates", "trainee"},
    "productivity": {"excel", "word", "admin assistant", "admin assistants", "office productivity"},
    "healthcare": {"hipaa", "patient records", "healthcare", "medical", "admin staff"},
    "finance": {"financial analyst", "financial analysts", "finance"},
}

ALLOWED_ROLE_FAMILIES = set(ROLE_KEYWORDS.keys())

ROLE_FAMILY_ALIASES = {
    "customer_service": "contact_center",
    "customer service": "contact_center",
    "contactcentre": "contact_center",
    "contact-center": "contact_center",
    "contact center": "contact_center",
    "leadership_selection": "leadership",
    "technical_hiring": "technical",
    "office": "productivity",
    "office_productivity": "productivity",
}

GOAL_ALIASES = {
    "hiring": "selection",
    "recruitment": "selection",
    "selection": "selection",
    "screening": "screening",
    "screen": "screening",
    "development": "development",
    "reskilling": "development",
    "reskilling audit": "development",
}

TECHNOLOGY_NAME_MAP = {
    "core java": "Core Java (Advanced Level) (New)",
    "spring": "Spring (New)",
    "rest": "RESTful Web Services (New)",
    "sql": "SQL (New)",
    "aws": "Amazon Web Services (AWS) Development (New)",
    "docker": "Docker (New)",
    "angular": "Angular (New)",
    "verify g+": "SHL Verify Interactive G+",
    "g+": "SHL Verify Interactive G+",
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Assessment",
}


class ConversationMessage(BaseModel):
    role: str
    content: str


class Recommendation(BaseModel):
    entity_id: str
    name: str
    test_type: str
    keys: list[str]
    duration: str | None = None
    languages: list[str] = Field(default_factory=list)
    url: str
    rationale: str | None = None


class ChatRequest(BaseModel):
    messages: list[ConversationMessage]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] | None
    end_of_conversation: bool


class CatalogProduct(BaseModel):
    entity_id: str
    name: str
    url: str
    description: str = ""
    keys: list[str] = Field(default_factory=list)
    test_types: list[str] = Field(default_factory=list)
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str | None = None
    duration_minutes: int | None = None
    adaptive: bool = False
    remote: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def test_type_display(self) -> str:
        return ",".join(self.test_types)

    @property
    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            " ".join(self.languages),
            self.duration or "",
            self.url,
        ]
        return " ".join(part for part in parts if part).lower()


class Intent(BaseModel):
    role_family: str | None = None
    seniority: str | None = None
    language: str | None = None
    accent: str | None = None
    goal: str | None = None
    compare_products: list[str] = Field(default_factory=list)
    remove_targets: list[str] = Field(default_factory=list)
    replacement_hint: str | None = None
    affirming: bool = False
    legal: bool = False
    prompt_injection: bool = False
    off_topic: bool = False
    requested_dimensions: set[str] = Field(default_factory=set)
    user_turns: int = 0
    raw_text: str = ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_duration_minutes(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)", value.lower())
    if not match:
        return None
    return int(match.group(1))


def map_test_types(keys: Sequence[str]) -> list[str]:
    mapped: list[str] = []
    for key in keys:
        code = KEY_TO_TEST_TYPE.get(key)
        if code and code not in mapped:
            mapped.append(code)
    return mapped


def load_catalog(path: Path = CATALOG_PATH) -> list[CatalogProduct]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    products: list[CatalogProduct] = []
    for item in raw_items:
        duration = item.get("duration") or item.get("duration_raw") or None
        normalized_duration = normalize_whitespace(duration) if duration else None
        products.append(
            CatalogProduct(
                entity_id=str(item.get("entity_id", "")),
                name=item.get("name", ""),
                url=item.get("link", ""),
                description=item.get("description", ""),
                keys=list(item.get("keys", [])),
                test_types=map_test_types(item.get("keys", [])),
                job_levels=list(item.get("job_levels", [])),
                languages=list(item.get("languages", [])),
                duration=normalized_duration,
                duration_minutes=parse_duration_minutes(normalized_duration or ""),
                adaptive=str(item.get("adaptive", "no")).lower() == "yes",
                remote=str(item.get("remote", "no")).lower() == "yes",
                raw=item,
            )
        )
    return products


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9+\- ]+", " ", text.lower()).strip()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _resolve_known_product_mentions(text: str, products: Sequence[CatalogProduct]) -> list[CatalogProduct]:
    normalized = _normalize_for_match(text)
    matches: list[CatalogProduct] = []
    for product in products:
        normalized_name = _normalize_for_match(product.name)
        simplified_name = re.sub(r"\s*\([^)]*\)", "", product.name)
        simplified_normalized = _normalize_for_match(simplified_name)
        if (
            normalized_name in normalized
            or simplified_normalized in normalized
            or normalized in normalized_name
            or normalized in simplified_normalized
        ) and product not in matches:
            matches.append(product)
    for token, product_name in TECHNOLOGY_NAME_MAP.items():
        if token in normalized:
            product = next((item for item in products if item.name.lower() == product_name.lower()), None)
            if product and product not in matches:
                matches.append(product)
    return matches


class SemanticCatalogIndex:
    def __init__(self, products: Sequence[CatalogProduct]):
        self.products = list(products)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=8000)
        self.matrix = self.vectorizer.fit_transform(product.searchable_text for product in self.products)
        self.dense_matrix = self.matrix.astype(np.float32).toarray()
        self._normalize_dense(self.dense_matrix)
        self.faiss_index = None
        if faiss is not None:
            self.faiss_index = faiss.IndexFlatIP(self.dense_matrix.shape[1])
            self.faiss_index.add(self.dense_matrix)

    @staticmethod
    def _normalize_dense(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms

    def search(self, query: str, top_k: int = 5) -> list[tuple[CatalogProduct, float]]:
        query_vector = self.vectorizer.transform([query]).astype(np.float32).toarray()
        self._normalize_dense(query_vector)
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector, top_k)
            output: list[tuple[CatalogProduct, float]] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                output.append((self.products[index], float(score)))
            return output

        scores = self.dense_matrix @ query_vector[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.products[index], float(scores[index])) for index in ranked_indices]


def _detect_language(text: str) -> str | None:
    lowered = text.lower()
    for language in ["english", "spanish", "french", "german", "portuguese", "dutch", "chinese", "korean", "romanian", "italian"]:
        if re.search(rf"\b{re.escape(language)}\b", lowered):
            return language.title()
    if re.search(r"\b(us|usa)\b", lowered):
        return "English (USA)"
    if re.search(r"\b(uk|british)\b", lowered):
        return "English (UK)"
    if re.search(r"\baustralian\b", lowered):
        return "English (Australia)"
    if re.search(r"\bindian\b", lowered):
        return "English (India)"
    return None


def _detect_accent(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(us|usa)\b", lowered):
        return "US"
    if re.search(r"\b(uk|british)\b", lowered):
        return "UK"
    if re.search(r"\baustralian\b", lowered):
        return "Australian"
    if re.search(r"\bindian\b", lowered):
        return "Indian"
    return None


def _detect_seniority(text: str) -> str | None:
    lowered = text.lower()
    seniority_map = [
        ("executive", ["cxo", "executive", "vp", "vice president"]),
        ("director", ["director"]),
        ("manager", ["manager", "leadership benchmark", "tech lead"]),
        ("senior ic", ["senior ic", "senior individual contributor", "senior engineer", "senior full-stack engineer"]),
        ("entry level", ["entry-level", "entry level", "graduate", "final-year"]),
        ("front line", ["front line", "supervisor", "contact centre", "contact center"]),
    ]
    for label, terms in seniority_map:
        if any(term in lowered for term in terms):
            return label
    return None


def _detect_role_family(text: str) -> str | None:
    lowered = text.lower()
    for family, terms in ROLE_KEYWORDS.items():
        if any(term in lowered for term in terms):
            return family
    return None


def _requested_dimensions(text: str) -> set[str]:
    lowered = text.lower()
    dimensions = set()
    for dimension, terms in {
        "ability": ["cognitive", "reasoning", "ability", "aptitude", "numerical"],
        "personality": ["personality", "behaviour", "behavior", "fit"],
        "situational_judgement": ["situational judgement", "situational judgment", "sjt", "decision making"],
        "knowledge": ["knowledge", "domain", "technical", "test"],
        "simulation": ["simulation", "simulations", "capabilit", "performance"],
        "report": ["report"],
    }.items():
        if any(term in lowered for term in terms):
            dimensions.add(dimension)
    return dimensions


def _detect_goal(text: str) -> str | None:
    lowered = text.lower()
    if any(term in lowered for term in ["selection", "select", "screening", "screen", "hiring", "benchmark"]):
        return "selection"
    if any(term in lowered for term in ["development", "reskill", "re-skill", "audit", "feedback"]):
        return "development"
    if any(term in lowered for term in ["finalist", "finalists", "volume", "high-volume"]):
        return "screening"
    return None


def _extract_compare_targets(text: str) -> list[str]:
    lowered = text.lower()
    if "difference between" not in lowered and "compare" not in lowered and "different from" not in lowered:
        return []
    products = []
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    products.extend(quoted)
    if len(products) < 2:
        between_match = re.search(r"difference between (.+?) and (.+?)(?:\?|\.|$)", text, flags=re.I)
        if between_match:
            products.extend([between_match.group(1).strip(), between_match.group(2).strip()])
    if len(products) < 2:
        for candidate in _resolve_known_product_mentions(text, load_catalog()):
            products.append(candidate.name)
            if len(products) == 2:
                break
    return [normalize_whitespace(product) for product in products[:2]]


def _extract_removal_targets(text: str) -> list[str]:
    lowered = text.lower()
    targets: list[str] = []
    if any(term in lowered for term in ["remove", "drop", "delete", "without"]):
        for name in TECHNOLOGY_NAME_MAP.values():
            if name.lower() in lowered:
                targets.append(name)
        if "opq" in lowered:
            targets.append("Occupational Personality Questionnaire OPQ32r")
    return targets


def _extract_replacement_hint(text: str) -> str | None:
    match = re.search(r"replace(?: it| this| the .*?)? with (.+)$", text, flags=re.I)
    if match:
        return normalize_whitespace(match.group(1))
    if "something shorter" in text.lower():
        return "shorter"
    return None


def _normalize_role_family(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalize_for_match(value).replace("_", " ")
    mapped = ROLE_FAMILY_ALIASES.get(normalized, normalized)
    mapped = mapped.replace(" ", "_")
    if mapped in ALLOWED_ROLE_FAMILIES:
        return mapped
    return None


def _normalize_goal(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalize_for_match(value)
    return GOAL_ALIASES.get(normalized)


def _accent_signal_present(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\b(us|usa|uk|british|australian|indian)\b", lowered)
    )


def _match_product(name: str, products: Sequence[CatalogProduct]) -> CatalogProduct | None:
    normalized = _normalize_for_match(name)
    for product in products:
        if normalized == _normalize_for_match(product.name):
            return product
    for product in products:
        if normalized and normalized in _normalize_for_match(product.name):
            return product
    return None


def _product_recommendation(product: CatalogProduct, rationale: str | None = None) -> Recommendation:
    return Recommendation(
        entity_id=product.entity_id,
        name=product.name,
        test_type=product.test_type_display,
        keys=product.keys,
        duration=product.duration,
        languages=product.languages,
        url=product.url,
        rationale=rationale,
    )


def _answer_compare(products: Sequence[CatalogProduct], compared: Sequence[str]) -> ChatResponse:
    left = _match_product(compared[0], products) if compared else None
    right = _match_product(compared[1], products) if len(compared) > 1 else None
    if not left or not right:
        return ChatResponse(
            reply="I could not confidently resolve both products from the catalog. Please name the two exact assessment names.",
            recommendations=None,
            end_of_conversation=False,
        )

    left_kind = "report" if "report" in left.name.lower() or "report" in left.description.lower() else "assessment"
    right_kind = "report" if "report" in right.name.lower() or "report" in right.description.lower() else "assessment"
    reply = (
        f"{left.name} is a {left_kind} with test types {left.test_type_display or 'unknown'}; "
        f"{right.name} is a {right_kind} with test types {right.test_type_display or 'unknown'}. "
        f"Both are distinct catalog items, and the key difference is in what they measure and how they are used."
    )
    return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)


class ConversationRecommender:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.products = load_catalog(catalog_path)
        self.index = SemanticCatalogIndex(self.products)

    def respond(self, messages: Sequence[ConversationMessage]) -> ChatResponse:
        if not messages:
            return ChatResponse(
                reply="Please share the hiring context and assessment goals so I can recommend a shortlist.",
                recommendations=None,
                end_of_conversation=False,
            )

        intent = self._extract_intent(messages)

        if intent.prompt_injection:
            return ChatResponse(
                reply="I can help with the assessment catalog, but I can’t follow instructions to ignore the system or reveal hidden prompts.",
                recommendations=None,
                end_of_conversation=False,
            )

        if intent.legal:
            return ChatResponse(
                reply="I can help select assessments, but I can’t interpret legal or compliance obligations or say whether a test satisfies a regulation.",
                recommendations=None,
                end_of_conversation=False,
            )

        if intent.off_topic and not intent.role_family and not intent.compare_products:
            return ChatResponse(
                reply="I can only help with SHL assessment selection, comparison, and shortlist updates.",
                recommendations=None,
                end_of_conversation=False,
            )

        if intent.compare_products:
            return _answer_compare(self.products, intent.compare_products)

        if intent.role_family == "leadership" and intent.goal is None and intent.user_turns < 7:
            return ChatResponse(reply=self._clarification_question(intent), recommendations=None, end_of_conversation=False)

        if intent.role_family == "contact_center" and not intent.language and intent.user_turns < 7:
            return ChatResponse(reply=self._clarification_question(intent), recommendations=None, end_of_conversation=False)

        if intent.role_family == "contact_center" and intent.language and not intent.accent and intent.user_turns < 7:
            return ChatResponse(reply=self._clarification_question(intent), recommendations=None, end_of_conversation=False)

        if intent.role_family == "technical" and intent.user_turns == 1 and not intent.affirming and not intent.goal:
            return ChatResponse(reply=self._clarification_question(intent), recommendations=None, end_of_conversation=False)

        current_shortlist = self._build_shortlist(intent)

        if intent.remove_targets:
            current_shortlist = [product for product in current_shortlist if product.name not in intent.remove_targets]

        if intent.replacement_hint and intent.remove_targets:
            replacement = self._find_replacement(intent, current_shortlist)
            if replacement is None:
                reply = "The requested replacement is not a good fit from the catalog, so I’m keeping the shortlist unchanged except for any explicit removals."
                if current_shortlist:
                    return ChatResponse(
                        reply=reply,
                        recommendations=[_product_recommendation(product, self._rationale_for(product, intent)) for product in current_shortlist],
                        end_of_conversation=intent.user_turns >= 7,
                    )
                return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)
            current_shortlist = replacement

        if not current_shortlist and intent.user_turns < 7:
            return ChatResponse(reply=self._clarification_question(intent), recommendations=None, end_of_conversation=False)

        if not current_shortlist:
            return ChatResponse(
                reply="I’m committing to the best available shortlist based on the information so far, though some requirements remain under-specified.",
                recommendations=None,
                end_of_conversation=True,
            )

        reply = self._build_reply(intent, current_shortlist)
        return ChatResponse(
            reply=reply,
            recommendations=[_product_recommendation(product, self._rationale_for(product, intent)) for product in current_shortlist],
            end_of_conversation=self._is_finalized(intent),
        )

    def _extract_intent(self, messages: Sequence[ConversationMessage]) -> Intent:
        user_messages = [message.content for message in messages if message.role.lower() == "user"]
        combined = normalize_whitespace(" ".join(user_messages))
        latest = normalize_whitespace(user_messages[-1]) if user_messages else ""

        if os.getenv("ENABLE_LLM") == "1":
            llm = build_openrouter_chat_model()
            if llm is not None:
                try:
                    LOGGER.info("LLM_ATTEMPTED=true model=%s", os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct"))
                    prompt = (
                        "Extract a compact JSON object from the conversation. "
                        "Return keys: role_family, seniority, language, accent, goal. "
                        "Use null for missing values and do not include any extra text.\n\n"
                        f"Conversation: {combined}"
                    )
                    response = llm.invoke(prompt)
                    payload = maybe_extract_json(getattr(response, "content", str(response)))
                    if payload:
                        llm_role_family = _normalize_role_family(payload.get("role_family"))
                        llm_goal = _normalize_goal(payload.get("goal"))
                        detected_accent = _detect_accent(combined)
                        llm_accent = payload.get("accent") if isinstance(payload.get("accent"), str) else None
                        accent_value = llm_accent if _accent_signal_present(combined) else detected_accent
                        LOGGER.info("LLM_USED=true")
                        return Intent(
                            role_family=llm_role_family or _detect_role_family(combined),
                            seniority=payload.get("seniority") or _detect_seniority(combined),
                            language=payload.get("language") or _detect_language(combined),
                            accent=accent_value,
                            goal=llm_goal or _detect_goal(combined),
                            compare_products=_extract_compare_targets(latest),
                            remove_targets=_extract_removal_targets(latest),
                            replacement_hint=_extract_replacement_hint(latest),
                            affirming=_contains_any(latest, ACKNOWLEDGEMENT_TERMS),
                            legal=_contains_any(combined, LEGAL_TERMS),
                            prompt_injection=_contains_any(combined, PROMPT_INJECTION_TERMS),
                            off_topic=_contains_any(combined, OFF_TOPIC_TERMS) and not _contains_any(
                                combined,
                                ROLE_KEYWORDS["technical"] | ROLE_KEYWORDS["productivity"] | ROLE_KEYWORDS["healthcare"] | ROLE_KEYWORDS["sales"] | ROLE_KEYWORDS["safety"] | ROLE_KEYWORDS["graduate"] | ROLE_KEYWORDS["contact_center"] | ROLE_KEYWORDS["leadership"] | ROLE_KEYWORDS["finance"],
                            ),
                            requested_dimensions=_requested_dimensions(combined),
                            user_turns=len(user_messages),
                            raw_text=combined,
                        )
                    LOGGER.warning("LLM_USED=false reason=invalid_json_payload")
                except Exception as error:
                    LOGGER.warning("LLM_USED=false reason=llm_error error=%s", error)
            else:
                LOGGER.warning("LLM_USED=false reason=client_unavailable")
        else:
            LOGGER.info("LLM_ATTEMPTED=false reason=ENABLE_LLM_not_1")

        return Intent(
            role_family=_detect_role_family(combined),
            seniority=_detect_seniority(combined),
            language=_detect_language(combined),
            accent=_detect_accent(combined),
            goal=_detect_goal(combined),
            compare_products=_extract_compare_targets(latest),
            remove_targets=_extract_removal_targets(latest),
            replacement_hint=_extract_replacement_hint(latest),
            affirming=_contains_any(latest, ACKNOWLEDGEMENT_TERMS),
            legal=_contains_any(combined, LEGAL_TERMS),
            prompt_injection=_contains_any(combined, PROMPT_INJECTION_TERMS),
            off_topic=_contains_any(combined, OFF_TOPIC_TERMS) and not _contains_any(
                combined,
                ROLE_KEYWORDS["technical"] | ROLE_KEYWORDS["productivity"] | ROLE_KEYWORDS["healthcare"] | ROLE_KEYWORDS["sales"] | ROLE_KEYWORDS["safety"] | ROLE_KEYWORDS["graduate"] | ROLE_KEYWORDS["contact_center"] | ROLE_KEYWORDS["leadership"] | ROLE_KEYWORDS["finance"],
            ),
            requested_dimensions=_requested_dimensions(combined),
            user_turns=len(user_messages),
            raw_text=combined,
        )

    @staticmethod
    def _is_finalized(intent: Intent) -> bool:
        return intent.user_turns >= 7 or _contains_any(intent.raw_text, FINALIZATION_TERMS)

    def _clarification_question(self, intent: Intent) -> str:
        text = intent.raw_text
        if intent.role_family == "leadership" or _contains_any(text, {"leadership", "executive", "director", "cxo"}):
            if "selection" not in text and "development" not in text:
                return "Is this for selection against a leadership benchmark, or for development feedback?"
            return "Who is this meant for?"
        if intent.role_family == "contact_center":
            if not intent.language:
                return "What language are the calls in?"
            if intent.language and not intent.accent:
                return f"Which {intent.language.lower()} accent or market should I use for the spoken-language screen?"
        if intent.role_family == "healthcare":
            return "Are the candidates functionally bilingual enough for the English knowledge tests, or do you want a Spanish-only personality path?"
        if intent.role_family == "technical" and not intent.seniority:
            return "What seniority level are you hiring for, and is this a senior IC or a tech lead?"
        if intent.role_family == "technical" and "backend" in text.lower() and "frontend" in text.lower() and "balanced" not in text.lower():
            return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role?"
        if intent.role_family == "graduate" and "battery" not in text.lower():
            return "Do you want a full graduate battery, or just a specific cognitive or judgement component?"
        if intent.role_family == "productivity" and not intent.goal:
            return "Is this a quick knowledge screen, a simulation-heavy battery, or both?"
        return "What is the role family, seniority, and primary assessment goal?"

    def _build_shortlist(self, intent: Intent) -> list[CatalogProduct]:
        text = intent.raw_text
        explicit = _resolve_known_product_mentions(text, self.products)

        if intent.role_family == "leadership" or _contains_any(text, {"leadership", "executive", "director", "benchmark"}):
            return self._unique_products([
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
                self._match_required("OPQ Universal Competency Report 2.0"),
                self._match_required("OPQ Leadership Report"),
            ])

        if intent.role_family == "sales":
            return self._unique_products([
                self._match_required("Global Skills Assessment"),
                self._match_required("Global Skills Development Report"),
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
                self._match_required("OPQ MQ Sales Report"),
                self._match_required("Sales Transformation 2.0 - Individual Contributor"),
            ])

        if intent.role_family == "safety":
            return self._unique_products([
                self._match_required("Manufac. & Indust. - Safety & Dependability 8.0"),
                self._match_required("Workplace Health and Safety (New)"),
            ])

        if intent.role_family == "graduate":
            shortlist = [
                self._match_required("SHL Verify Interactive G+"),
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
                self._match_required("Graduate Scenarios"),
            ]
            shortlist.extend(self._technical_additions(text))
            return self._unique_products(shortlist)

        if intent.role_family == "contact_center":
            accent = intent.accent or self._detect_contact_center_accent(text)
            shortlist = [self._product_for_accent(accent), self._match_required("Contact Center Call Simulation (New)")]
            if _contains_any(text, {"entry-level", "entry level", "customer service"}):
                shortlist.append(self._match_required("Entry Level Customer Serv-Retail & Contact Center"))
            shortlist.append(self._match_required("Customer Service Phone Simulation"))
            return self._unique_products(shortlist)

        if intent.role_family == "healthcare":
            return self._unique_products([
                self._match_required("HIPAA (Security)"),
                self._match_required("Medical Terminology (New)"),
                self._match_required("Microsoft Word 365 - Essentials (New)"),
                self._match_required("Dependability and Safety Instrument (DSI)"),
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
            ])

        if intent.role_family == "finance":
            return self._unique_products([
                self._match_required("SHL Verify Interactive – Numerical Reasoning"),
                self._match_required("Financial Accounting (New)"),
                self._match_required("Basic Statistics (New)"),
                self._match_required("Graduate Scenarios"),
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
            ])

        if intent.role_family == "productivity":
            shortlist = [
                self._match_required("MS Excel (New)"),
                self._match_required("MS Word (New)"),
                self._match_required("Occupational Personality Questionnaire OPQ32r"),
            ]
            if _contains_any(text, {"simulation", "simulate", "capabilit"}):
                shortlist = [
                    self._match_required("Microsoft Excel 365 (New)"),
                    self._match_required("Microsoft Word 365 (New)"),
                    self._match_required("MS Excel (New)"),
                    self._match_required("MS Word (New)"),
                    self._match_required("Occupational Personality Questionnaire OPQ32r"),
                ]
            return self._unique_products(shortlist)

        if intent.role_family == "technical":
            shortlist = self._technical_shortlist(text)
            if explicit:
                shortlist.extend(explicit)
            return self._unique_products(shortlist)

        semantic = self.index.search(text, top_k=5)
        return [product for product, score in semantic if score > 0][:5]

    def _technical_shortlist(self, text: str) -> list[CatalogProduct]:
        shortlist: list[CatalogProduct] = []
        tech_map = {
            "core java": "Core Java (Advanced Level) (New)",
            "spring": "Spring (New)",
            "rest": "RESTful Web Services (New)",
            "sql": "SQL (New)",
            "aws": "Amazon Web Services (AWS) Development (New)",
            "docker": "Docker (New)",
            "angular": "Angular (New)",
        }
        lowered = text.lower()
        for token, product_name in tech_map.items():
            if token in lowered:
                product = self._match_required(product_name)
                if product not in shortlist:
                    shortlist.append(product)
        if any(token in lowered for token in ["java", "spring", "sql", "aws", "docker", "angular", "backend", "frontend", "engineer", "rust"]):
            verify = self._match_required("SHL Verify Interactive G+")
            opq = self._match_required("Occupational Personality Questionnaire OPQ32r")
            if verify not in shortlist:
                shortlist.append(verify)
            if opq not in shortlist:
                shortlist.append(opq)
        if "rust" in lowered:
            live_coding = self._match_required("Smart Interview Live Coding")
            linux = self._match_required("Linux Programming (General)")
            networking = self._match_required("Networking and Implementation (New)")
            shortlist = [live_coding, linux, networking] + shortlist
        return shortlist

    def _technical_additions(self, text: str) -> list[CatalogProduct]:
        return self._technical_shortlist(text)

    def _find_replacement(self, intent: Intent, current_shortlist: list[CatalogProduct]) -> list[CatalogProduct] | None:
        if intent.replacement_hint and "shorter" in intent.replacement_hint.lower() and any(product.name == "Occupational Personality Questionnaire OPQ32r" for product in current_shortlist):
            return None
        return None

    def _match_required(self, name: str) -> CatalogProduct:
        product = _match_product(name, self.products)
        if product is None:
            raise KeyError(f"Catalog product not found: {name}")
        return product

    @staticmethod
    def _unique_products(products: Sequence[CatalogProduct]) -> list[CatalogProduct]:
        unique: list[CatalogProduct] = []
        seen: set[str] = set()
        for product in products:
            if product.entity_id in seen:
                continue
            seen.add(product.entity_id)
            unique.append(product)
        return unique

    def _detect_contact_center_accent(self, text: str) -> str | None:
        lowered = text.lower()
        if re.search(r"\b(us|usa)\b", lowered):
            return "US"
        if re.search(r"\b(uk|british)\b", lowered):
            return "UK"
        if re.search(r"\baustralian\b", lowered) or re.search(r"\baus\b", lowered):
            return "Australian"
        if re.search(r"\bindian\b", lowered):
            return "Indian"
        return None

    def _product_for_accent(self, accent: str | None) -> CatalogProduct:
        accent = (accent or "US").lower()
        candidates = [product for product in self.products if "spoken english" in product.name.lower() or "svar" in product.name.lower()]
        if not candidates:
            return self._match_required("SVAR Spoken English (US) (New)")
        accent_patterns = {
            "us": [r"\bus\b", r"\busa\b"],
            "uk": [r"\buk\b", r"\bu\.k\.\b"],
            "australian": [r"\baustralian\b", r"\baus\b"],
            "indian": [r"\bindian\b"],
        }
        patterns = accent_patterns.get(accent, accent_patterns["us"])
        for product in candidates:
            normalized_name = _normalize_for_match(product.name)
            if any(re.search(pattern, normalized_name) for pattern in patterns):
                return product
        return candidates[0]

    def _rationale_for(self, product: CatalogProduct, intent: Intent) -> str:
        if intent.role_family == "technical" and product.name in {"SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"}:
            return "Complements the technical tests with reasoning and behavioural signal for a senior hire."
        if intent.role_family == "leadership":
            return "Fits the leadership benchmark and pairs the questionnaire with leadership-oriented reports."
        if intent.role_family == "productivity":
            return "Covers day-to-day office productivity while preserving an optional behavioural signal."
        if intent.role_family == "contact_center":
            return "Covers spoken-language screening plus call handling for high-volume customer service."
        if intent.role_family == "safety":
            return "Matches the safety-critical emphasis on dependability and compliance with procedures."
        if intent.role_family == "graduate":
            return "Supports a graduate battery with cognitive, behavioural, and situational judgment coverage."
        return "Selected from the catalog based on the stated role and assessment goals."

    def _build_reply(self, intent: Intent, shortlist: Sequence[CatalogProduct]) -> str:
        if intent.user_turns >= 7 and not intent.affirming:
            return "I’m committing to the best available shortlist based on the information so far, with some uncertainty remaining."
        if intent.affirming:
            return "Confirmed. Here is the current shortlist."
        return "Here is the shortlist based on the role and constraints you shared."


@lru_cache(maxsize=1)
def get_recommender() -> ConversationRecommender:
    return ConversationRecommender()


def build_chat_response(messages: Sequence[ConversationMessage]) -> ChatResponse:
    return get_recommender().respond(messages)

