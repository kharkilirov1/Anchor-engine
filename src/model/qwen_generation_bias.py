from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from src.model.anchor_types import AnchorRecord


@dataclass(frozen=True)
class BiasDomainProfile:
    name: str
    alpha_multiplier: float
    pressure_threshold_shift: float
    rescue_floor_multiplier: float
    forbidden_penalty: float
    hard_block_forbidden: bool
    allow_terms: tuple[str, ...]
    block_terms: tuple[str, ...]


_GENERIC_BIAS_TERMS: tuple[str, ...] = (
    "the",
    "a",
    "an",
    "and",
)
# Note: reduced from original 12 terms to 4 core articles/conjunctions
# Weight reduced to 0.01 in build_bias_token_weights

_DEFAULT_DOMAIN_PROFILE = BiasDomainProfile(
    name="default",
    alpha_multiplier=0.75,
    pressure_threshold_shift=0.10,
    rescue_floor_multiplier=0.50,
    forbidden_penalty=0.0,
    hard_block_forbidden=False,
    allow_terms=(),
    block_terms=(),
)
_MATH_DOMAIN_PROFILE = BiasDomainProfile(
    name="math",
    alpha_multiplier=0.60,  # Tuned for valid math proof generation
    pressure_threshold_shift=0.22,
    rescue_floor_multiplier=0.20,
    forbidden_penalty=4.0,
    hard_block_forbidden=True,  # Enable hard block for constraint violations
    allow_terms=(
        "assume",
        "suppose",
        "rational",
        "irrational",
        "contradiction",
        "therefore",
        "square",
        "sqrt",
        "integer",
        "coprime",
        "let",
        "then",
    ),
    block_terms=(
        "theory",
        "algebraic",
        "field",
        "polynomial",
        "degree",
        "complex",
        "beta",
        "gamma",
        "delta",
    ),
)
_CODE_DOMAIN_PROFILE = BiasDomainProfile(
    name="code",
    alpha_multiplier=0.90,
    pressure_threshold_shift=0.04,
    rescue_floor_multiplier=0.85,
    forbidden_penalty=6.0,
    hard_block_forbidden=True,  # Enable hard block for constraint violations
    allow_terms=(
        "fastapi",
        "async",
        "await",
        "pydantic",
        "response_model",
        "httpexception",
        "dependency",
        "depends",
        "background",
        "tasks",
        "request",
        "response",
    ),
    block_terms=(
        "django",
        "flask",
        "template",
        "jinja",
        "render",
        "synchronous",
        "session",
        "make_response",
    ),
)
_VEGAN_DOMAIN_PROFILE = BiasDomainProfile(
    name="vegan",
    alpha_multiplier=0.32,  # Reverted to original value
    pressure_threshold_shift=0.12,
    rescue_floor_multiplier=0.35,
    forbidden_penalty=8.0,
    hard_block_forbidden=True,  # Enable hard block for constraint violations
    allow_terms=(
        "vegan",
        "plant",
        "plant-based",
        "tofu",
        "lentil",
        "lentils",
        "bean",
        "beans",
        "chickpea",
        "chickpeas",
        "oat",
        "almond",
        "vegetable",
        "vegetables",
    ),
    block_terms=(
        "vegetarian",
        "egg",
        "eggs",
        "milk",
        "cheese",
        "butter",
        "cream",
        "salmon",
        "tuna",
        "chicken",
        "beef",
        "pork",
    ),
)
_DIETARY_DOMAIN_PROFILE = BiasDomainProfile(
    name="dietary",
    alpha_multiplier=0.40,
    pressure_threshold_shift=0.10,
    rescue_floor_multiplier=0.40,
    forbidden_penalty=8.0,
    hard_block_forbidden=True,
    allow_terms=(),  # populated per-domain via token weights
    block_terms=(),
)
_CODE_STRICT_DOMAIN_PROFILE = BiasDomainProfile(
    name="code_strict",
    alpha_multiplier=0.85,
    pressure_threshold_shift=0.06,
    rescue_floor_multiplier=0.80,
    forbidden_penalty=7.0,
    hard_block_forbidden=True,
    allow_terms=(),
    block_terms=(),
)
_LEGAL_DOMAIN_PROFILE = BiasDomainProfile(
    name="legal",
    alpha_multiplier=0.55,
    pressure_threshold_shift=0.15,
    rescue_floor_multiplier=0.30,
    forbidden_penalty=5.0,
    hard_block_forbidden=True,
    allow_terms=(),
    block_terms=(),
)
_MEDICAL_DOMAIN_PROFILE = BiasDomainProfile(
    name="medical",
    alpha_multiplier=0.50,
    pressure_threshold_shift=0.18,
    rescue_floor_multiplier=0.25,
    forbidden_penalty=6.0,
    hard_block_forbidden=True,
    allow_terms=(),
    block_terms=(),
)
_CONSTRAINT_DOMAIN_PROFILE = BiasDomainProfile(
    name="constraint",
    alpha_multiplier=0.65,
    pressure_threshold_shift=0.10,
    rescue_floor_multiplier=0.45,
    forbidden_penalty=5.0,
    hard_block_forbidden=True,
    allow_terms=(),
    block_terms=(),
)


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)


def _clamp_unit_interval(value: float, default: float = 0.0) -> float:
    if not math.isfinite(value):
        return float(default)
    return min(1.0, max(0.0, float(value)))


_PROFILE_REGISTRY: dict[str, BiasDomainProfile] = {
    "default": _DEFAULT_DOMAIN_PROFILE,
    "math": _MATH_DOMAIN_PROFILE,
    "code": _CODE_DOMAIN_PROFILE,
    "vegan": _VEGAN_DOMAIN_PROFILE,
    "dietary": _DIETARY_DOMAIN_PROFILE,
    "code_strict": _CODE_STRICT_DOMAIN_PROFILE,
    "legal": _LEGAL_DOMAIN_PROFILE,
    "medical": _MEDICAL_DOMAIN_PROFILE,
    "constraint": _CONSTRAINT_DOMAIN_PROFILE,
}


def get_profile_by_name(name: str) -> BiasDomainProfile:
    return _PROFILE_REGISTRY.get(name, _DEFAULT_DOMAIN_PROFILE)


def infer_bias_domain(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token in lowered for token in ("sqrt", "proof by contradiction", "irrational", "rational")):
        return "math"
    if any(token in lowered for token in ("fastapi", "pydantic", "django", "flask", "request handlers")):
        return "code"
    if any(token in lowered for token in ("vegan", "meal plan", "plant-based", "chef")):
        return "vegan"
    if any(token in lowered for token in ("halal", "gluten-free", "gluten free", "kosher")):
        return "dietary"
    if any(token in lowered for token in ("gdpr", "compliance", "legal", "retention policy")):
        return "legal"
    if any(token in lowered for token in ("opioid", "pharmacological", "medication", "chronic pain")):
        return "medical"
    if any(token in lowered for token in ("unsafe", "raw pointer", "strict null", "no orm")):
        return "code_strict"
    if any(token in lowered for token in ("kubernetes", "kubectl", "deployment guide")):
        return "code"
    return "default"


def get_bias_domain_profile(prompt: str) -> BiasDomainProfile:
    domain = infer_bias_domain(prompt)
    return _PROFILE_REGISTRY.get(domain, _DEFAULT_DOMAIN_PROFILE)


def _extract_token_ids_from_term(
    tokenizer: object,
    term: str,
) -> set[int]:
    token_ids: set[int] = set()
    encoded = None
    try:
        encoded = tokenizer(term, add_special_tokens=False)
    except Exception:
        encoded = None
    if isinstance(encoded, dict) and "input_ids" in encoded:
        values = encoded["input_ids"]
        if isinstance(values, torch.Tensor):
            token_ids.update(int(token) for token in values.reshape(-1).tolist())
        elif isinstance(values, list):
            if values and isinstance(values[0], list):
                token_ids.update(int(token) for row in values for token in row)
            else:
                token_ids.update(int(token) for token in values)
    elif isinstance(encoded, list):
        token_ids.update(int(token) for token in encoded)

    if token_ids:
        return token_ids

    batch_encoded = tokenizer(
        [term],
        padding=True,
        truncation=True,
        max_length=max(len(term), 4),
        return_tensors="pt",
    )
    ids = batch_encoded.get("input_ids")
    mask = batch_encoded.get("attention_mask")
    if isinstance(ids, torch.Tensor):
        if isinstance(mask, torch.Tensor):
            active = ids[mask.bool()]
            token_ids.update(int(token) for token in active.tolist())
        else:
            token_ids.update(int(token) for token in ids.reshape(-1).tolist())
    return token_ids


def _normalize_token_surface(text: str) -> str:
    normalized = text.replace("Ġ", " ").replace("▁", " ").replace("</w>", " ")
    normalized = normalized.replace("Ċ", " ").replace("ĉ", " ")
    return normalized.strip().lower()


def _surface_matches_term(surface: str, term: str) -> bool:
    normalized_surface = _normalize_token_surface(surface)
    normalized_term = _normalize_token_surface(term)
    if not normalized_surface or not normalized_term:
        return False
    surface_words = re.findall(r"[a-zA-Z][a-zA-Z\-]*", normalized_surface)
    term_words = re.findall(r"[a-zA-Z][a-zA-Z\-]*", normalized_term)
    if not term_words:
        return False
    if len(normalized_term) <= 3:
        return normalized_surface == normalized_term or normalized_term in surface_words
    return (
        normalized_surface == normalized_term
        or normalized_term in normalized_surface
        or any(word == normalized_term for word in surface_words)
    )


def _token_surface_map(
    tokenizer: object,
    vocab_size: int,
) -> dict[int, str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            token_list = convert(list(range(vocab_size)))
            if isinstance(token_list, list) and len(token_list) == vocab_size:
                return {idx: str(token) for idx, token in enumerate(token_list)}
        except Exception:
            pass

    surface_map: dict[int, str] = {}
    for token_id in range(vocab_size):
        try:
            surface_map[token_id] = str(tokenizer.decode([token_id], skip_special_tokens=False))
        except Exception:
            surface_map[token_id] = ""
    return surface_map


def build_bias_token_weights(
    tokenizer: object | None,
    vocab_size: int,
    device: torch.device,
    prompt: str,
) -> tuple[torch.Tensor | None, set[int], dict[str, Any]]:
    profile = get_bias_domain_profile(prompt)
    if tokenizer is None:
        return None, set(), {
            "domain": profile.name,
            "alpha_multiplier": float(profile.alpha_multiplier),
            "pressure_threshold_shift": float(profile.pressure_threshold_shift),
            "rescue_floor_multiplier": float(profile.rescue_floor_multiplier),
            "forbidden_penalty": float(profile.forbidden_penalty),
            "hard_block_forbidden": bool(profile.hard_block_forbidden),
            "masked_token_fraction": 0.0,
            "boosted_token_fraction": 0.0,
        }

    weights = torch.ones(vocab_size, device=device, dtype=torch.float32)
    surface_map = _token_surface_map(tokenizer, vocab_size)
    generic_ids: set[int] = set()
    for term in _GENERIC_BIAS_TERMS:
        generic_ids.update(_extract_token_ids_from_term(tokenizer, term))
        generic_ids.update(
            token_id for token_id, surface in surface_map.items()
            if _surface_matches_term(surface, term)
        )
    for token_id in generic_ids:
        if 0 <= token_id < vocab_size:
            weights[token_id] = min(float(weights[token_id].item()), 0.01)  # Reduced from 0.05

    blocked_ids: set[int] = set()
    for term in profile.block_terms:
        blocked_ids.update(_extract_token_ids_from_term(tokenizer, term))
        blocked_ids.update(
            token_id for token_id, surface in surface_map.items()
            if _surface_matches_term(surface, term)
        )
    for token_id in blocked_ids:
        if 0 <= token_id < vocab_size:
            weights[token_id] = 0.0  # Mask out blocked terms (hard block via hard_block_forbidden flag)

    allow_ids: set[int] = set()
    for term in profile.allow_terms:
        allow_ids.update(_extract_token_ids_from_term(tokenizer, term))
        allow_ids.update(
            token_id for token_id, surface in surface_map.items()
            if _surface_matches_term(surface, term)
        )
    for token_id in allow_ids:
        if 0 <= token_id < vocab_size and weights[token_id] > 0.0:
            weights[token_id] = max(float(weights[token_id].item()), 1.35)

    return weights, blocked_ids, {
        "domain": profile.name,
        "alpha_multiplier": float(profile.alpha_multiplier),
        "pressure_threshold_shift": float(profile.pressure_threshold_shift),
        "rescue_floor_multiplier": float(profile.rescue_floor_multiplier),
        "forbidden_penalty": float(profile.forbidden_penalty),
        "hard_block_forbidden": bool(profile.hard_block_forbidden),
        "masked_token_fraction": float((weights == 0).float().mean().item()),
        "boosted_token_fraction": float((weights > 1.0).float().mean().item()),
        "forbidden_token_count": int(len(blocked_ids)),
    }


def apply_forbidden_token_penalty(
    logits: torch.Tensor,
    forbidden_token_ids: Iterable[int],
    penalty: float,
    *,
    hard_block: bool = False,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if penalty <= 0.0:
        return logits
    token_ids = sorted({int(token_id) for token_id in forbidden_token_ids if int(token_id) >= 0})
    if not token_ids:
        return logits
    adjusted = logits.clone()
    if hard_block:
        adjusted[0, token_ids] = torch.finfo(adjusted.dtype).min
        return adjusted
    adjusted[0, token_ids] = adjusted[0, token_ids] - float(penalty)
    return adjusted


def compute_anchor_generation_gate(
    similarity: float,
    support: float,
    contradiction_pressure: float,
    viability: float,
    conflict_threshold: float,
) -> float:
    drift = max(0.0, float(conflict_threshold) - float(similarity))
    return (
        drift
        * max(0.0, float(support))
        * (0.20 + 0.80 * float(contradiction_pressure))
        * (0.55 + 0.45 * float(viability))
    )


def compute_normalized_entropy(
    logits: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    return compute_entropy_stats(logits, top_k=top_k)["normalized_entropy"]


def compute_entropy_stats(
    logits: torch.Tensor,
    top_k: int | None = None,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("logits must be shaped [batch, vocab_size]")
    safe_logits = _sanitize_logits(logits)
    if top_k is not None:
        top_k = max(1, min(int(top_k), int(safe_logits.size(-1))))
        if top_k == 1:
            zero = torch.zeros(safe_logits.size(0), device=safe_logits.device, dtype=safe_logits.dtype)
            return {
                "raw_entropy": zero,
                "normalized_entropy": zero,
            }
        safe_logits = torch.topk(safe_logits, k=top_k, dim=-1).values
        norm_value = math.log(float(top_k))
    else:
        norm_value = math.log(float(safe_logits.size(-1)))
    probs = F.softmax(safe_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.tensor(norm_value, device=safe_logits.device, dtype=safe_logits.dtype).clamp_min(1e-12)
    normalized = torch.nan_to_num(entropy / norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return {
        "raw_entropy": entropy,
        "normalized_entropy": normalized,
    }


def compute_entropy_conflict_bias_scale(
    normalized_entropy: float,
    contradiction_pressure: float,
    alpha_max: float,
    entropy_threshold: float,
    pressure_threshold: float,
    entropy_slope: float,
    pressure_slope: float,
    pressure_rescue_floor: float,
) -> dict[str, float]:
    entropy_input_isfinite = math.isfinite(float(normalized_entropy))
    pressure_input_isfinite = math.isfinite(float(contradiction_pressure))
    safe_entropy = _clamp_unit_interval(float(normalized_entropy), default=0.0)
    safe_pressure = _clamp_unit_interval(float(contradiction_pressure), default=0.0)
    entropy_gate = torch.sigmoid(
        torch.tensor((safe_entropy - float(entropy_threshold)) / max(float(entropy_slope), 1e-6))
    ).item()
    pressure_gate = torch.sigmoid(
        torch.tensor((safe_pressure - float(pressure_threshold)) / max(float(pressure_slope), 1e-6))
    ).item()
    rescue_floor = min(1.0, max(0.0, float(pressure_rescue_floor)))
    alpha_t = float(alpha_max) * float(pressure_gate) * (rescue_floor + (1.0 - rescue_floor) * float(entropy_gate))
    alpha_t = float(_clamp_unit_interval(alpha_t / max(float(alpha_max), 1e-6), default=0.0) * max(float(alpha_max), 0.0))
    return {
        "alpha_t": float(alpha_t),
        "entropy_gate": float(entropy_gate),
        "pressure_gate": float(pressure_gate),
        "safe_normalized_entropy": float(safe_entropy),
        "safe_contradiction_pressure": float(safe_pressure),
        "entropy_input_isfinite": float(entropy_input_isfinite),
        "pressure_input_isfinite": float(pressure_input_isfinite),
        "alpha_isfinite": float(math.isfinite(alpha_t)),
    }


def compute_anchor_logits_bias(
    last_hidden: torch.Tensor,
    active_anchors: Iterable[AnchorRecord],
    output_projection: torch.nn.Module,
    conflict_threshold: float,
    bias_scale: float,
    bias_token_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if last_hidden.ndim != 2 or last_hidden.size(0) != 1:
        raise ValueError("last_hidden must be shaped [1, hidden_dim]")

    dtype = last_hidden.dtype
    device = last_hidden.device
    bias: torch.Tensor | None = None
    diagnostics: list[dict[str, float]] = []
    current = F.normalize(last_hidden, dim=-1)

    for anchor in active_anchors:
        anchor_repr = anchor.repr.to(device=device, dtype=dtype).unsqueeze(0)
        similarity = float(F.cosine_similarity(current, F.normalize(anchor_repr, dim=-1), dim=-1).item())
        gate = compute_anchor_generation_gate(
            similarity=similarity,
            support=float(anchor.support),
            contradiction_pressure=float(anchor.contradiction_pressure),
            viability=float(anchor.viability),
            conflict_threshold=conflict_threshold,
        )
        if gate <= 0.0:
            continue
        anchor_logits = output_projection(anchor_repr).squeeze(0)
        anchor_logits = anchor_logits - anchor_logits.mean()
        anchor_logits = anchor_logits / anchor_logits.std().clamp_min(1e-6)
        if bias_token_weights is not None:
            anchor_logits = anchor_logits * bias_token_weights.to(device=device, dtype=anchor_logits.dtype)
        scaled = float(bias_scale) * float(gate) * anchor_logits
        bias = scaled if bias is None else bias + scaled
        diagnostics.append(
            {
                "anchor_id": float(anchor.id),
                "similarity": float(similarity),
                "gate": float(gate),
                "support": float(anchor.support),
                "contradiction_pressure": float(anchor.contradiction_pressure),
                "viability": float(anchor.viability),
            }
        )

    if bias is None:
        output_dim = int(output_projection.weight.shape[0])
        bias = torch.zeros(output_dim, device=device, dtype=dtype)
    return bias.unsqueeze(0), diagnostics


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if penalty <= 1.0:
        return logits

    adjusted = logits.clone()
    token_ids = generated_ids[0].tolist()
    for token_id in set(int(token_id) for token_id in token_ids):
        value = adjusted[0, token_id]
        adjusted[0, token_id] = torch.where(
            value > 0,
            value / penalty,
            value * penalty,
        )
    return adjusted


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if penalty <= 0.0:
        return logits

    adjusted = logits.clone()
    counts = Counter(int(token_id) for token_id in generated_ids[0].tolist())
    for token_id, count in counts.items():
        adjusted[0, token_id] = adjusted[0, token_id] - (float(count) * penalty)
    return adjusted


def _collect_blocked_tokens_for_ngram(
    token_ids: list[int],
    ngram_size: int,
) -> set[int]:
    if ngram_size <= 1 or len(token_ids) < ngram_size - 1:
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1) :])
    blocked: set[int] = set()
    for idx in range(len(token_ids) - ngram_size + 1):
        ngram = token_ids[idx : idx + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(int(ngram[-1]))
    return blocked


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int,
) -> tuple[torch.Tensor, set[int]]:
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("logits must be shaped [1, vocab_size]")
    if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
        raise ValueError("generated_ids must be shaped [1, seq_len]")
    if ngram_size <= 1:
        return logits, set()

    blocked = _collect_blocked_tokens_for_ngram(
        token_ids=[int(token_id) for token_id in generated_ids[0].tolist()],
        ngram_size=ngram_size,
    )
    if not blocked:
        return logits, blocked

    adjusted = logits.clone()
    for token_id in blocked:
        adjusted[0, token_id] = torch.finfo(adjusted.dtype).min
    return adjusted, blocked
