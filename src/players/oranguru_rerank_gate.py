"""Neural-hook utilities for tactical rerank gating.

The gate is intentionally advisory and fail-open by default.  MCTS still owns the
base policy; this hook can later learn when tactical reranks should be allowed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional


RERANK_WINDOWS = (
    ("finish_blow", "finish_blow"),
    ("setup_window", "setup_window"),
    ("recovery_window", "recovery_window"),
    ("progress_window", "progress_window"),
    ("switch_guard", "switch_guard"),
    ("passive_breaker", "passive_breaker"),
)

TAKE_TARGET_KEYS = (
    "finish_choice",
    "setup_choice",
    "recovery_choice",
    "progress_choice",
    "attack_choice",
    "passive_choice",
)

ACTION_KINDS = ("attack", "switch", "setup", "recovery", "status", "protect", "tera_attack", "move", "tera")
SOURCE_KEYS = tuple(source for source, _key in RERANK_WINDOWS) + ("unknown",)

FEATURE_NAMES = (
    "policy_confidence",
    "policy_threshold",
    "policy_margin",
    "top1_score",
    "candidate_score",
    "score_drop_top1_minus_candidate",
    "top1_heuristic",
    "candidate_heuristic",
    "heuristic_delta_candidate_minus_top1",
    "top1_risk",
    "candidate_risk",
    "risk_delta_top1_minus_candidate",
    "matchup_score",
    "best_reply_score",
    "hazard_load",
    "active_hp",
    "opp_hp",
    "reply_score",
    "candidate_policy_ratio",
    "heuristic_gain_payload",
) + tuple(f"source_{source}" for source in SOURCE_KEYS) + tuple(
    f"candidate_kind_{kind}" for kind in ACTION_KINDS
) + tuple(f"top1_kind_{kind}" for kind in ACTION_KINDS)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def top_actions(row: Mapping[str, object]) -> list[dict]:
    actions = row.get("top_actions") or []
    return [dict(action) for action in actions if isinstance(action, Mapping)]


def action_for_choice(row: Mapping[str, object], choice: str) -> Optional[dict]:
    for action in top_actions(row):
        if str(action.get("choice", "") or "") == choice:
            return action
    return None


def action_score(action: Optional[Mapping[str, object]]) -> float:
    if not action:
        return 0.0
    return safe_float(action.get("score", action.get("weight", 0.0)), 0.0)


def action_heuristic(action: Optional[Mapping[str, object]]) -> float:
    if not action:
        return 0.0
    return safe_float(action.get("heuristic_score", 0.0), 0.0)


def action_risk(action: Optional[Mapping[str, object]]) -> float:
    if not action:
        return 0.0
    return safe_float(action.get("risk_penalty", 0.0), 0.0)


def action_kind(choice: str, action: Optional[Mapping[str, object]] = None) -> str:
    if action and action.get("kind"):
        return str(action.get("kind") or "")
    if choice.startswith("switch "):
        return "switch"
    if choice.endswith("-tera"):
        return "tera"
    if choice:
        return "move"
    return "unknown"


def take_source_from_payload(source: str, payload: object, choice: str) -> tuple[str, Optional[dict]]:
    if not isinstance(payload, Mapping):
        return "", None
    reason = str(payload.get("reason", "") or "")
    if not reason.startswith("take_"):
        return "", None
    targets = {str(payload.get(key, "") or "") for key in TAKE_TARGET_KEYS if payload.get(key)}
    if targets and choice not in targets:
        return "", None
    return f"{source}:{reason}", dict(payload)


def trace_rerank_source(row: Mapping[str, object], choice: str) -> tuple[str, Optional[dict]]:
    for source, key in RERANK_WINDOWS:
        source_reason, payload = take_source_from_payload(source, row.get(key), choice)
        if source_reason:
            return source_reason, payload
    return "rerank:unknown", None


def source_name(source_reason: str) -> str:
    if ":" not in source_reason:
        return "unknown"
    source = source_reason.split(":", 1)[0]
    return source if source in SOURCE_KEYS else "unknown"


def empty_feature_dict() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def build_feature_dict(
    *,
    source_reason: str,
    candidate_choice: str,
    top1_choice: str,
    candidate_kind: str,
    top1_kind: str,
    candidate_score: float,
    top1_score: float,
    candidate_heuristic: float,
    top1_heuristic: float,
    candidate_risk: float,
    top1_risk: float,
    policy_confidence: float,
    policy_threshold: float,
    matchup_score: float = 0.0,
    best_reply_score: float = 0.0,
    hazard_load: float = 0.0,
    payload: Optional[Mapping[str, object]] = None,
) -> dict[str, float]:
    features = empty_feature_dict()
    score_drop = max(0.0, top1_score - candidate_score)
    features["policy_confidence"] = safe_float(policy_confidence)
    features["policy_threshold"] = safe_float(policy_threshold)
    features["policy_margin"] = max(0.0, top1_score - candidate_score)
    features["top1_score"] = safe_float(top1_score)
    features["candidate_score"] = safe_float(candidate_score)
    features["score_drop_top1_minus_candidate"] = score_drop
    features["top1_heuristic"] = safe_float(top1_heuristic)
    features["candidate_heuristic"] = safe_float(candidate_heuristic)
    features["heuristic_delta_candidate_minus_top1"] = safe_float(candidate_heuristic) - safe_float(top1_heuristic)
    features["top1_risk"] = safe_float(top1_risk)
    features["candidate_risk"] = safe_float(candidate_risk)
    features["risk_delta_top1_minus_candidate"] = safe_float(top1_risk) - safe_float(candidate_risk)
    features["matchup_score"] = safe_float(matchup_score)
    features["best_reply_score"] = safe_float(best_reply_score)
    features["hazard_load"] = safe_float(hazard_load)

    if payload:
        features["active_hp"] = safe_float(payload.get("active_hp", 0.0))
        features["opp_hp"] = safe_float(payload.get("opp_hp", 0.0))
        features["reply_score"] = safe_float(payload.get("reply_score", 0.0))
        chosen_weight = safe_float(payload.get("chosen_weight", top1_score), top1_score)
        candidate_weight = 0.0
        for key in ("setup_weight", "recovery_weight", "progress_weight", "attack_weight", "finish_weight", "passive_weight"):
            if key in payload:
                candidate_weight = safe_float(payload.get(key, 0.0))
                break
        if candidate_weight <= 0.0:
            candidate_weight = candidate_score
        features["candidate_policy_ratio"] = candidate_weight / max(chosen_weight, 1e-6)
        payload_heuristic = None
        for key in ("setup_heuristic", "recovery_heuristic", "progress_heuristic", "attack_heuristic", "finish_heuristic", "passive_heuristic"):
            if key in payload:
                payload_heuristic = safe_float(payload.get(key, 0.0))
                break
        chosen_heuristic = safe_float(payload.get("chosen_heuristic", top1_heuristic), top1_heuristic)
        if payload_heuristic is not None:
            features["heuristic_gain_payload"] = payload_heuristic - chosen_heuristic

    features[f"source_{source_name(source_reason)}"] = 1.0
    if candidate_kind in ACTION_KINDS:
        features[f"candidate_kind_{candidate_kind}"] = 1.0
    if top1_kind in ACTION_KINDS:
        features[f"top1_kind_{top1_kind}"] = 1.0
    return features


def feature_vector(features: Mapping[str, float], feature_names: Iterable[str] = FEATURE_NAMES) -> list[float]:
    return [safe_float(features.get(name, 0.0), 0.0) for name in feature_names]


def build_trace_rerank_gate_example(row: Mapping[str, object]) -> Optional[dict]:
    if str(row.get("selection_path", "") or "") != "rerank":
        return None
    value_target = safe_float(row.get("value_target", 0.0), 0.0)
    if value_target == 0.0:
        return None
    candidate_choice = str(row.get("chosen_choice", "") or "")
    actions = top_actions(row)
    if not candidate_choice or not actions:
        return None
    top1 = actions[0]
    top1_choice = str(top1.get("choice", "") or "")
    if not top1_choice or top1_choice == candidate_choice:
        return None
    candidate = action_for_choice(row, candidate_choice)
    if candidate is None:
        return None
    source_reason, payload = trace_rerank_source(row, candidate_choice)
    features = build_feature_dict(
        source_reason=source_reason,
        candidate_choice=candidate_choice,
        top1_choice=top1_choice,
        candidate_kind=action_kind(candidate_choice, candidate),
        top1_kind=action_kind(top1_choice, top1),
        candidate_score=action_score(candidate),
        top1_score=action_score(top1),
        candidate_heuristic=action_heuristic(candidate),
        top1_heuristic=action_heuristic(top1),
        candidate_risk=action_risk(candidate),
        top1_risk=action_risk(top1),
        policy_confidence=safe_float(row.get("policy_confidence", 0.0), 0.0),
        policy_threshold=safe_float(row.get("policy_threshold", 0.0), 0.0),
        matchup_score=safe_float(row.get("matchup_score", 0.0), 0.0),
        best_reply_score=safe_float(row.get("best_reply_score", 0.0), 0.0),
        hazard_load=safe_float(row.get("hazard_load", 0.0), 0.0),
        payload=payload,
    )
    return {
        "battle_id": str(row.get("battle_id", "") or ""),
        "turn": int(row.get("turn", 0) or 0),
        "source": source_reason,
        "label": 1 if value_target > 0.0 else 0,
        "value_target": value_target,
        "candidate_choice": candidate_choice,
        "top1_choice": top1_choice,
        "candidate_kind": action_kind(candidate_choice, candidate),
        "top1_kind": action_kind(top1_choice, top1),
        "features": features,
        "feature_vector": feature_vector(features),
        "feature_names": list(FEATURE_NAMES),
    }


def init_rerank_gate(self) -> bool:
    if getattr(self, "_rerank_gate_failed", False):
        return False
    if getattr(self, "_rerank_gate_model", None) is not None:
        return True
    if not getattr(self, "RERANK_GATE_ENABLED", False):
        return False
    try:
        path = Path(str(getattr(self, "RERANK_GATE_MODEL", "")))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            self._rerank_gate_failed = True
            return False
        with path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)
        mode = str(model.get("mode", "linear") or "linear").strip().lower()
        if mode == "bucket_rules":
            rules = model.get("rules") or []
            if not isinstance(rules, list):
                self._rerank_gate_failed = True
                return False
            self._rerank_gate_model = {
                "mode": "bucket_rules",
                "default": str(model.get("default", "allow") or "allow").strip().lower(),
                "rules": [dict(rule) for rule in rules if isinstance(rule, Mapping)],
                "path": str(path),
            }
            return True
        names = model.get("feature_names") or list(FEATURE_NAMES)
        weights = model.get("weights") or []
        if not isinstance(names, list) or not isinstance(weights, list) or len(names) != len(weights):
            self._rerank_gate_failed = True
            return False
        self._rerank_gate_model = {
            "feature_names": [str(name) for name in names],
            "weights": [safe_float(weight) for weight in weights],
            "bias": safe_float(model.get("bias", 0.0), 0.0),
            "threshold": safe_float(model.get("threshold", getattr(self, "RERANK_GATE_THRESHOLD", 0.5)), 0.5),
            "source_thresholds": dict(model.get("source_thresholds") or {}),
            "path": str(path),
        }
        return True
    except Exception:
        self._rerank_gate_failed = True
        return False


def _feature_name_for_rule_key(key: str) -> str:
    aliases = {
        "score_drop": "score_drop_top1_minus_candidate",
        "heuristic_delta": "heuristic_delta_candidate_minus_top1",
        "risk_delta": "risk_delta_top1_minus_candidate",
    }
    return aliases.get(key, key)


def _rule_float_matches(rule: Mapping[str, object], features: Mapping[str, float], name: str) -> bool:
    feature_name = _feature_name_for_rule_key(name)
    value = safe_float(features.get(feature_name, 0.0), 0.0)
    min_key = f"{name}_min"
    max_key = f"{name}_max"
    if min_key in rule and value < safe_float(rule.get(min_key), float("-inf")):
        return False
    if max_key in rule and value > safe_float(rule.get(max_key), float("inf")):
        return False
    return True


def _bucket_rule_matches(
    rule: Mapping[str, object],
    *,
    source_reason: str,
    candidate_kind: str,
    top1_kind: str,
    features: Mapping[str, float],
) -> bool:
    source = str(rule.get("source", "") or "")
    if source and source != source_reason:
        return False
    candidate_kind_rule = str(rule.get("candidate_kind", "") or "")
    if candidate_kind_rule and candidate_kind_rule != candidate_kind:
        return False
    top1_kind_rule = str(rule.get("top1_kind", "") or "")
    if top1_kind_rule and top1_kind_rule != top1_kind:
        return False

    rule_fields = set(rule.keys())
    threshold_bases = {
        key[:-4]
        for key in rule_fields
        if key.endswith("_min") and key not in {"source_min", "candidate_kind_min", "top1_kind_min"}
    }
    threshold_bases.update(
        key[:-4]
        for key in rule_fields
        if key.endswith("_max") and key not in {"source_max", "candidate_kind_max", "top1_kind_max"}
    )
    for name in threshold_bases:
        if not _rule_float_matches(rule, features, name):
            return False
    return True


def _bucket_rules_allow(
    model: Mapping[str, object],
    *,
    source_reason: str,
    candidate_kind: str,
    top1_kind: str,
    features: Mapping[str, float],
) -> tuple[bool, Optional[int], str]:
    default_action = str(model.get("default", "allow") or "allow").strip().lower()
    default_allow = default_action != "block"
    rules = model.get("rules") or []
    if not isinstance(rules, list):
        return default_allow, None, default_action
    for idx, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, Mapping):
            continue
        if not _bucket_rule_matches(
            raw_rule,
            source_reason=source_reason,
            candidate_kind=candidate_kind,
            top1_kind=top1_kind,
            features=features,
        ):
            continue
        action = str(raw_rule.get("action", default_action) or default_action).strip().lower()
        return action != "block", idx, action
    return default_allow, None, default_action


def _record_gate(self, battle, reason: str, **extra) -> None:
    try:
        mem = self._get_battle_memory(battle)
    except Exception:
        return
    if not isinstance(mem, dict):
        return
    payload = {"reason": reason}
    payload.update(extra)
    mem["rerank_gate_last"] = payload


def _last_take_source(self, battle, candidate_choice: str) -> tuple[str, Optional[dict]]:
    try:
        mem = self._get_battle_memory(battle)
    except Exception:
        return "rerank:unknown", None
    if not isinstance(mem, Mapping):
        return "rerank:unknown", None
    for source, key in RERANK_WINDOWS:
        source_reason, payload = take_source_from_payload(source, mem.get(f"{key}_last"), candidate_choice)
        if source_reason:
            return source_reason, payload
    return "rerank:unknown", None


def rerank_gate_allows(
    self,
    battle,
    ordered: list[tuple[str, float]],
    current_choice: str,
    candidate_choice: str,
    confidence: float,
    threshold: float,
) -> bool:
    if not candidate_choice or candidate_choice == current_choice:
        return True
    if not getattr(self, "RERANK_GATE_ENABLED", False):
        return True

    stats = getattr(self, "_mcts_stats", None)
    if isinstance(stats, dict):
        stats["rerank_gate_calls"] = int(stats.get("rerank_gate_calls", 0) or 0) + 1

    if not self._init_rerank_gate():
        if isinstance(stats, dict):
            stats["rerank_gate_init_failed"] = int(stats.get("rerank_gate_init_failed", 0) or 0) + 1
        allow = bool(getattr(self, "RERANK_GATE_FAIL_OPEN", True))
        _record_gate(self, battle, "model_unavailable_allow" if allow else "model_unavailable_block", candidate_choice=candidate_choice)
        return allow

    source_reason, payload = _last_take_source(self, battle, candidate_choice)
    top1_choice = ordered[0][0] if ordered else current_choice
    weights = {choice: safe_float(weight) for choice, weight in ordered}

    def _heur(choice: str) -> float:
        try:
            return safe_float(self._heuristic_action_score(battle, choice), 0.0)
        except Exception:
            return 0.0

    def _risk(choice: str) -> float:
        try:
            return safe_float(self._adaptive_choice_risk_penalty(battle, choice), 0.0)
        except Exception:
            return 0.0

    def _kind(choice: str) -> str:
        try:
            return str(self._search_trace_choice_kind(battle, choice) or "")
        except Exception:
            return action_kind(choice)

    matchup_score = 0.0
    best_reply_score = 0.0
    hazard_load = 0.0
    try:
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if active is not None and opponent is not None:
            matchup_score = safe_float(self._estimate_matchup(active, opponent), 0.0)
            best_reply_score = safe_float(self._estimate_best_reply_score(opponent, active, battle), 0.0)
    except Exception:
        pass
    try:
        hazard_load = safe_float(self._side_hazard_pressure(battle), 0.0)
    except Exception:
        pass

    candidate_kind = _kind(candidate_choice)
    top1_kind = _kind(top1_choice)
    features = build_feature_dict(
        source_reason=source_reason,
        candidate_choice=candidate_choice,
        top1_choice=top1_choice,
        candidate_kind=candidate_kind,
        top1_kind=top1_kind,
        candidate_score=weights.get(candidate_choice, 0.0),
        top1_score=weights.get(top1_choice, 0.0),
        candidate_heuristic=_heur(candidate_choice),
        top1_heuristic=_heur(top1_choice),
        candidate_risk=_risk(candidate_choice),
        top1_risk=_risk(top1_choice),
        policy_confidence=confidence,
        policy_threshold=threshold,
        matchup_score=matchup_score,
        best_reply_score=best_reply_score,
        hazard_load=hazard_load,
        payload=payload,
    )
    model = self._rerank_gate_model or {}
    if str(model.get("mode", "linear") or "linear").strip().lower() == "bucket_rules":
        allow, rule_idx, action = _bucket_rules_allow(
            model,
            source_reason=source_reason,
            candidate_kind=candidate_kind,
            top1_kind=top1_kind,
            features=features,
        )
        if isinstance(stats, dict):
            key = "rerank_gate_allowed" if allow else "rerank_gate_blocked"
            stats[key] = int(stats.get(key, 0) or 0) + 1
        _record_gate(
            self,
            battle,
            "allow" if allow else "block",
            source=source_reason,
            current_choice=current_choice,
            candidate_choice=candidate_choice,
            mode="bucket_rules",
            action=action,
            rule_idx=rule_idx,
            score_drop=float(features.get("score_drop_top1_minus_candidate", 0.0)),
            heuristic_delta=float(features.get("heuristic_delta_candidate_minus_top1", 0.0)),
        )
        return allow

    names = model.get("feature_names") or list(FEATURE_NAMES)
    weights_model = model.get("weights") or []
    logit = safe_float(model.get("bias", 0.0), 0.0)
    for name, weight in zip(names, weights_model):
        logit += safe_float(weight) * safe_float(features.get(str(name), 0.0), 0.0)
    probability = sigmoid(logit)
    source_thresholds = model.get("source_thresholds") or {}
    gate_threshold = safe_float(source_thresholds.get(source_reason, model.get("threshold", 0.5)), 0.5)
    allow = probability >= gate_threshold
    if isinstance(stats, dict):
        key = "rerank_gate_allowed" if allow else "rerank_gate_blocked"
        stats[key] = int(stats.get(key, 0) or 0) + 1
    _record_gate(
        self,
        battle,
        "allow" if allow else "block",
        source=source_reason,
        current_choice=current_choice,
        candidate_choice=candidate_choice,
        probability=float(probability),
        threshold=float(gate_threshold),
    )
    return allow


def maybe_accept_rerank_choice(
    self,
    battle,
    ordered: list[tuple[str, float]],
    current_choice: str,
    candidate_choice: str,
    confidence: float,
    threshold: float,
) -> str:
    if candidate_choice == current_choice:
        return candidate_choice
    if self._rerank_gate_allows(battle, ordered, current_choice, candidate_choice, confidence, threshold):
        return candidate_choice
    return current_choice
