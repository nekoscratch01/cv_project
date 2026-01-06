"""QuerySpec extraction for SigLIP-friendly retrieval prompts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Iterable, List, Sequence

from core.config import SystemConfig


_NEGATION_MARKERS = ("not", "without", "excluding", "except")
_NEGATION_WINDOW = 8


@dataclass
class QueryBudget:
    k_min: int = 10
    k_max: int = 60
    margin_delta: float = 0.08
    frames_per_track: int = 3
    topm_frames: int = 2

    def with_overrides(
        self,
        *,
        k_min: int | None = None,
        k_max: int | None = None,
        margin_delta: float | None = None,
        frames_per_track: int | None = None,
        topm_frames: int | None = None,
    ) -> "QueryBudget":
        data = {
            "k_min": self.k_min if k_min is None else k_min,
            "k_max": self.k_max if k_max is None else k_max,
            "margin_delta": self.margin_delta if margin_delta is None else margin_delta,
            "frames_per_track": self.frames_per_track if frames_per_track is None else frames_per_track,
            "topm_frames": self.topm_frames if topm_frames is None else topm_frames,
        }
        return QueryBudget(**data)


@dataclass
class QuerySpec:
    raw_query: str
    positive_prompts: List[str]
    negative_prompts: List[str]
    mode_hint: str = "auto"
    need_context: bool = False
    constraint_intents: List[str] = field(default_factory=list)
    budget: QueryBudget = field(default_factory=QueryBudget)

    def with_budget(self, **kwargs) -> "QuerySpec":
        return replace(self, budget=self.budget.with_overrides(**kwargs))

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "positive_prompts": list(self.positive_prompts),
            "negative_prompts": list(self.negative_prompts),
            "mode_hint": self.mode_hint,
            "need_context": self.need_context,
            "constraint_intents": list(self.constraint_intents),
            "budget": {
                "k_min": self.budget.k_min,
                "k_max": self.budget.k_max,
                "margin_delta": self.budget.margin_delta,
                "frames_per_track": self.budget.frames_per_track,
                "topm_frames": self.budget.topm_frames,
            },
        }


class QuerySpecExtractor:
    """Rule-based QuerySpec extractor for English-only appearance queries."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self._base_budget = QueryBudget(
            k_min=getattr(self.config, "siglip_rerank_k_min", 10),
            k_max=getattr(self.config, "siglip_rerank_k_max", 60),
            margin_delta=getattr(self.config, "siglip_rerank_margin_delta", 0.08),
            frames_per_track=getattr(self.config, "siglip_frames_per_track", 3),
            topm_frames=getattr(self.config, "siglip_topm_frames", 2),
        )

        self.colors = _normalize_phrases(
            [
                "dark blue",
                "light blue",
                "navy",
                "blue",
                "red",
                "black",
                "white",
                "gray",
                "grey",
                "yellow",
                "green",
                "pink",
                "purple",
                "orange",
                "brown",
            ]
        )
        self.garments = _normalize_phrases(
            [
                "t shirt",
                "tshirt",
                "shirt",
                "hoodie",
                "jacket",
                "coat",
                "sweater",
                "pants",
                "jeans",
                "skirt",
                "dress",
                "shorts",
            ]
        )
        self.accessories = _normalize_phrases(
            [
                "backpack",
                "bag",
                "hat",
                "cap",
                "glasses",
                "sunglasses",
                "mask",
            ]
        )

    def extract(self, query: str) -> QuerySpec:
        raw_query = (query or "").strip()
        normalized = _normalize_query(raw_query)
        tokens = normalized.split()

        neg_prompts, neg_colors, neg_garments, neg_accessories = self._extract_negatives(tokens)

        color_hits = _find_phrases(normalized, self.colors)
        garment_hits = _find_phrases(normalized, self.garments)
        accessory_hits = _find_phrases(normalized, self.accessories)

        color_hits = [hit for hit in color_hits if hit[0] not in neg_colors]
        garment_hits = [hit for hit in garment_hits if hit[0] not in neg_garments]
        accessory_hits = [hit for hit in accessory_hits if hit[0] not in neg_accessories]

        prompt = self._build_positive_prompt(raw_query, color_hits, garment_hits, accessory_hits)
        positive_prompts = [prompt] if prompt else []
        if not positive_prompts:
            positive_prompts = [raw_query or "a person"]

        return QuerySpec(
            raw_query=raw_query,
            positive_prompts=positive_prompts,
            negative_prompts=neg_prompts,
            need_context=False,
            constraint_intents=[],
            budget=replace(self._base_budget),
        )

    def _extract_negatives(
        self, tokens: Sequence[str]
    ) -> tuple[List[str], set[str], set[str], set[str]]:
        neg_prompts: List[str] = []
        neg_colors: set[str] = set()
        neg_garments: set[str] = set()
        neg_accessories: set[str] = set()

        for idx, token in enumerate(tokens):
            if token not in _NEGATION_MARKERS:
                continue
            window_tokens = tokens[idx + 1 : idx + 1 + _NEGATION_WINDOW]
            if not window_tokens:
                continue
            window = " ".join(window_tokens)
            color = _pick_first(_find_phrases(window, self.colors))
            garment = _pick_first(_find_phrases(window, self.garments))
            accessory = _pick_first(_find_phrases(window, self.accessories))

            if color:
                neg_colors.add(color)
            if garment:
                neg_garments.add(garment)
            if accessory:
                neg_accessories.add(accessory)

            prompt = _build_attribute_prompt(color=color, garment=garment, accessory=accessory)
            if prompt:
                neg_prompts.append(prompt)

        neg_prompts = _dedupe_preserve_order(neg_prompts)
        return neg_prompts, neg_colors, neg_garments, neg_accessories

    def _build_positive_prompt(
        self,
        raw_query: str,
        color_hits: Sequence[tuple[str, int]],
        garment_hits: Sequence[tuple[str, int]],
        accessory_hits: Sequence[tuple[str, int]],
    ) -> str:
        color = _pick_first(color_hits)
        garment = _pick_first(garment_hits)
        accessory = _pick_first(accessory_hits)

        if color and garment:
            return _build_attribute_prompt(color=color, garment=garment, accessory=None) or raw_query
        if color:
            if accessory:
                return f"person wearing {color} clothing with {_accessory_phrase(accessory)}"
            return f"person wearing {color} clothing"
        if garment:
            return _build_attribute_prompt(color=None, garment=garment, accessory=None) or raw_query
        if accessory:
            return f"person with {_accessory_phrase(accessory)}"
        return raw_query


def _normalize_query(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _normalize_phrases(phrases: Iterable[str]) -> List[str]:
    return [_normalize_query(p) for p in phrases]


def _find_phrases(text: str, phrases: Iterable[str]) -> List[tuple[str, int]]:
    hits: List[tuple[str, int]] = []
    for phrase in phrases:
        if not phrase:
            continue
        pattern = r"\b" + re.escape(phrase) + r"\b"
        for match in re.finditer(pattern, text):
            hits.append((phrase, match.start()))
    return hits


def _pick_first(hits: Sequence[tuple[str, int]]) -> str | None:
    if not hits:
        return None
    return min(hits, key=lambda item: item[1])[0]


def _build_attribute_prompt(*, color: str | None, garment: str | None, accessory: str | None) -> str | None:
    if color and garment:
        if garment in {"pants", "jeans", "shorts"}:
            return f"person wearing {color} {garment}"
        return f"person wearing a {color} {garment}"
    if garment:
        if garment in {"pants", "jeans", "shorts"}:
            return f"person wearing {garment}"
        return f"person wearing a {garment}"
    if color:
        return f"person wearing {color} clothing"
    if accessory:
        return f"person with {_accessory_phrase(accessory)}"
    return None


def _accessory_phrase(accessory: str) -> str:
    if accessory.endswith("s"):
        return accessory
    return f"a {accessory}"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
