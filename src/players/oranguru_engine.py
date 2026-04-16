#!/usr/bin/env python3
"""
OranguruEnginePlayer - MCTS via poke-engine using a lightweight state builder.

Builds a poke-engine State from poke_env battle + randombattle set sampling
and runs MCTS to choose moves, similar in spirit to foul-play.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from poke_env.battle import AbstractBattle, Battle, Pokemon, SideCondition, MoveCategory
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.weather import Weather

from src.players.rule_bot import RuleBotPlayer
from src.players import oranguru_belief
from src.players import oranguru_decision
from src.players import oranguru_diag
from src.players import oranguru_searchflow
from src.players import oranguru_memory
from src.players import oranguru_state
from src.players import oranguru_models
from src.players import oranguru_tactical
from src.players import oranguru_trace
from src.players import oranguru_worlds
from src.utils.damage_calc import normalize_name
from src.utils.damage_belief import score_set_damage_consistency

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, LastUsedMove, StatRange  # noqa: E402
from data import all_move_json as FP_MOVE_JSON  # noqa: E402
from data.pkmn_sets import RandomBattleTeamDatasets  # noqa: E402
import constants  # noqa: E402


def _maybe_effect(name: str) -> Optional[Effect]:
    return getattr(Effect, name, None)


class OranguruEnginePlayer(RuleBotPlayer):
    CPU_COUNT = os.cpu_count() or 2
    SEARCH_TIME_MS = int(os.getenv("ORANGURU_SEARCH_MS", "200"))
    PARALLELISM = int(os.getenv("ORANGURU_PARALLELISM", str(max(1, min(3, CPU_COUNT // 2 or 1)))))
    SAMPLE_STATES = int(os.getenv("ORANGURU_SAMPLE_STATES", str(max(1, min(PARALLELISM, 3)))))
    MAX_SAMPLE_STATES = int(
        os.getenv("ORANGURU_SAMPLE_STATES_MAX", str(max(6, PARALLELISM * 4)))
    )
    DYNAMIC_SAMPLING = bool(int(os.getenv("ORANGURU_DYNAMIC_SAMPLING", "1")))
    LOW_UNCERTAINTY_WORLD_REDUCTION = bool(
        int(os.getenv("ORANGURU_LOW_UNCERTAINTY_WORLD_REDUCTION", "1"))
    )
    LOW_UNCERTAINTY_MIN_REVEALED = int(
        os.getenv("ORANGURU_LOW_UNCERTAINTY_MIN_REVEALED", "5")
    )
    LOW_UNCERTAINTY_MIN_KNOWN_MOVES = int(
        os.getenv("ORANGURU_LOW_UNCERTAINTY_MIN_KNOWN_MOVES", "2")
    )
    LOW_UNCERTAINTY_MAX_STATES = int(
        os.getenv("ORANGURU_LOW_UNCERTAINTY_MAX_STATES", str(max(2, SAMPLE_STATES)))
    )
    ENDGAME_WORLD_REDUCTION = bool(int(os.getenv("ORANGURU_ENDGAME_WORLD_REDUCTION", "1")))
    ENDGAME_MAX_STATES = int(
        os.getenv("ORANGURU_ENDGAME_MAX_STATES", str(max(2, SAMPLE_STATES)))
    )
    HEURISTIC_BLEND = float(os.getenv("ORANGURU_HEURISTIC_BLEND", "0.35"))
    MIN_HEURISTIC_BLEND = float(os.getenv("ORANGURU_MIN_HEURISTIC_BLEND", "0.0"))
    POLICY_CUTOFF = float(os.getenv("ORANGURU_POLICY_CUTOFF", "0.75"))
    STATUS_KO_GUARD = bool(int(os.getenv("ORANGURU_STATUS_KO_GUARD", "0")))
    STATUS_KO_THRESHOLD = float(os.getenv("ORANGURU_STATUS_KO_THRESHOLD", "200.0"))
    IMMUNITY_INFER = bool(int(os.getenv("ORANGURU_IMMUNITY_INFER", "0")))
    DAMAGE_BELIEF = bool(int(os.getenv("ORANGURU_DAMAGE_BELIEF", "0")))
    DAMAGE_BELIEF_MODE = os.getenv("ORANGURU_DAMAGE_BELIEF_MODE", "soft").strip().lower()
    DAMAGE_BELIEF_TOPK = int(os.getenv("ORANGURU_DAMAGE_BELIEF_TOPK", "6"))
    DAMAGE_BELIEF_MIN_OBS = int(os.getenv("ORANGURU_DAMAGE_BELIEF_MIN_OBS", "2"))
    DAMAGE_BELIEF_STRICT_ONLY = bool(int(os.getenv("ORANGURU_DAMAGE_BELIEF_STRICT_ONLY", "1")))
    DAMAGE_BELIEF_PER_OBS_MIN = float(os.getenv("ORANGURU_DAMAGE_BELIEF_PER_OBS_MIN", "0.90"))
    DAMAGE_BELIEF_PER_OBS_MAX = float(os.getenv("ORANGURU_DAMAGE_BELIEF_PER_OBS_MAX", "1.10"))
    DAMAGE_BELIEF_FINAL_MIN = float(os.getenv("ORANGURU_DAMAGE_BELIEF_FINAL_MIN", "0.80"))
    DAMAGE_BELIEF_FINAL_MAX = float(os.getenv("ORANGURU_DAMAGE_BELIEF_FINAL_MAX", "1.20"))
    DECISION_DIAG_ENABLED = bool(int(os.getenv("ORANGURU_DECISION_DIAG", "0")))
    DECISION_DIAG_LOG = bool(int(os.getenv("ORANGURU_DECISION_DIAG_LOG", "0")))
    DECISION_DIAG_TOPK = int(os.getenv("ORANGURU_DECISION_DIAG_TOPK", "3"))
    DECISION_DIAG_LOW_MARGIN = float(os.getenv("ORANGURU_DECISION_DIAG_LOW_MARGIN", "0.08"))
    MCTS_DETERMINISTIC = bool(int(os.getenv("ORANGURU_MCTS_DETERMINISTIC", "0")))
    MCTS_DETERMINISTIC_EVAL_ONLY = bool(int(os.getenv("ORANGURU_MCTS_DETERMINISTIC_EVAL_ONLY", "0")))
    BELIEF_SAMPLING = bool(int(os.getenv("ORANGURU_BELIEF_SAMPLING", "1")))
    BELIEF_IMMUNITY_MATCH = float(os.getenv("ORANGURU_BELIEF_IMMUNITY_MATCH", "1.5"))
    BELIEF_IMMUNITY_MISS = float(os.getenv("ORANGURU_BELIEF_IMMUNITY_MISS", "0.7"))
    MCTS_CONFIDENCE_THRESHOLD = float(os.getenv("ORANGURU_MCTS_CONFIDENCE", "0.6"))
    GATE_MODE = os.getenv("ORANGURU_GATE_MODE", "hard").lower()
    SELECTION_MODE = os.getenv("ORANGURU_SELECTION_MODE", "blend").lower()
    RERANK_TOPK = int(os.getenv("ORANGURU_RERANK_TOPK", "3"))
    SLEEP_STATUS_IDS = {"slp", "sleep"}
    SLEEP_CLAUSE_ENABLED = bool(int(os.getenv("ORANGURU_SLEEP_CLAUSE", "1")))
    STALL_SHUTDOWN_BOOST = bool(int(os.getenv("ORANGURU_STALL_SHUTDOWN_BOOST", "1")))
    AUTO_TERA = bool(int(os.getenv("ORANGURU_AUTO_TERA", "1")))
    SPEED_BOUNDS_ENABLED = bool(int(os.getenv("ORANGURU_SPEED_BOUNDS", "1")))
    MOVE_SAFETY_GUARD = bool(int(os.getenv("ORANGURU_MOVE_SAFETY_GUARD", "1")))
    TACTICAL_KO_THRESHOLD = float(os.getenv("ORANGURU_TACTICAL_KO_THRESHOLD", "220.0"))
    STATUS_STALL_MAX = int(os.getenv("ORANGURU_STATUS_STALL_MAX", "2"))
    RL_PRIOR_ENABLED = bool(int(os.getenv("ORANGURU_RL_PRIOR", "0")))
    RL_PRIOR_CHECKPOINT = os.getenv(
        "ORANGURU_RL_PRIOR_CHECKPOINT",
        "checkpoints/rl/replay_imitation_gen9human.pt",
    )
    RL_PRIOR_BLEND = float(os.getenv("ORANGURU_RL_PRIOR_BLEND", "0.2"))
    RL_PRIOR_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_RL_PRIOR_LOWCONF_ONLY", "1")))
    RL_PRIOR_DEVICE = os.getenv("ORANGURU_RL_PRIOR_DEVICE", "cpu").strip().lower()
    SEARCH_PRIOR_ENABLED = bool(int(os.getenv("ORANGURU_SEARCH_PRIOR", "0")))
    SEARCH_PRIOR_CHECKPOINT = os.getenv(
        "ORANGURU_SEARCH_PRIOR_CHECKPOINT",
        "checkpoints/rl/search_prior_value_mixed_g6_all_v1.pt",
    )
    SEARCH_PRIOR_BLEND = float(os.getenv("ORANGURU_SEARCH_PRIOR_BLEND", "0.15"))
    SEARCH_PRIOR_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_SEARCH_PRIOR_LOWCONF_ONLY", "1")))
    SEARCH_PRIOR_DEVICE = os.getenv("ORANGURU_SEARCH_PRIOR_DEVICE", "cpu").strip().lower()
    SWITCH_PRIOR_ENABLED = bool(int(os.getenv("ORANGURU_SWITCH_PRIOR", "0")))
    SWITCH_PRIOR_CHECKPOINT = os.getenv(
        "ORANGURU_SWITCH_PRIOR_CHECKPOINT",
        "checkpoints/rl/switch_prior_g6_all_v1.pt",
    )
    SWITCH_PRIOR_DEVICE = os.getenv("ORANGURU_SWITCH_PRIOR_DEVICE", "cpu").strip().lower()
    SWITCH_PRIOR_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_SWITCH_PRIOR_LOWCONF_ONLY", "1")))
    SWITCH_PRIOR_KEEP_TOPK = int(os.getenv("ORANGURU_SWITCH_PRIOR_KEEP_TOPK", "2"))
    SWITCH_PRIOR_MIN_CANDIDATES = int(os.getenv("ORANGURU_SWITCH_PRIOR_MIN_CANDIDATES", "2"))
    PASSIVE_BREAKER_ENABLED = bool(int(os.getenv("ORANGURU_PASSIVE_BREAKER", "0")))
    PASSIVE_BREAKER_CHECKPOINT = os.getenv(
        "ORANGURU_PASSIVE_BREAKER_CHECKPOINT",
        "checkpoints/rl/passive_break_g6_server_v1.pt",
    )
    PASSIVE_BREAKER_DEVICE = os.getenv("ORANGURU_PASSIVE_BREAKER_DEVICE", "cpu").strip().lower()
    PASSIVE_BREAKER_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_PASSIVE_BREAKER_LOWCONF_ONLY", "1")))
    PASSIVE_BREAKER_TOPK = int(os.getenv("ORANGURU_PASSIVE_BREAKER_TOPK", "3"))
    PASSIVE_BREAKER_MIN_PROB = float(os.getenv("ORANGURU_PASSIVE_BREAKER_MIN_PROB", "0.40"))
    PASSIVE_BREAKER_MIN_MARGIN = float(os.getenv("ORANGURU_PASSIVE_BREAKER_MIN_MARGIN", "0.05"))
    TERA_PRUNER_ENABLED = bool(int(os.getenv("ORANGURU_TERA_PRUNER", "0")))
    TERA_PRUNER_CHECKPOINT = os.getenv(
        "ORANGURU_TERA_PRUNER_CHECKPOINT",
        "checkpoints/rl/tera_pruner_g6_all_v1.pt",
    )
    TERA_PRUNER_DEVICE = os.getenv("ORANGURU_TERA_PRUNER_DEVICE", "cpu").strip().lower()
    TERA_PRUNER_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_TERA_PRUNER_LOWCONF_ONLY", "1")))
    TERA_PRUNER_KEEP_TOPK = int(os.getenv("ORANGURU_TERA_PRUNER_KEEP_TOPK", "1"))
    TERA_PRUNER_MIN_CANDIDATES = int(os.getenv("ORANGURU_TERA_PRUNER_MIN_CANDIDATES", "2"))
    WORLD_RANKER_ENABLED = bool(int(os.getenv("ORANGURU_WORLD_RANKER", "0")))
    WORLD_RANKER_CHECKPOINT = os.getenv(
        "ORANGURU_WORLD_RANKER_CHECKPOINT",
        "checkpoints/rl/world_ranker_g6_server_v3.pt",
    )
    WORLD_RANKER_DEVICE = os.getenv("ORANGURU_WORLD_RANKER_DEVICE", "cpu").strip().lower()
    WORLD_RANKER_KEEP_TOPK = int(os.getenv("ORANGURU_WORLD_RANKER_KEEP_TOPK", "12"))
    WORLD_RANKER_MIN_CANDIDATES = int(os.getenv("ORANGURU_WORLD_RANKER_MIN_CANDIDATES", "14"))
    WORLD_RANKER_LOW_UNCERTAINTY_ONLY = bool(
        int(os.getenv("ORANGURU_WORLD_RANKER_LOW_UNCERTAINTY_ONLY", "0"))
    )
    WORLD_RANKER_ENDGAME_ONLY = bool(int(os.getenv("ORANGURU_WORLD_RANKER_ENDGAME_ONLY", "1")))
    LEAF_VALUE_ENABLED = bool(int(os.getenv("ORANGURU_LEAF_VALUE", "0")))
    LEAF_VALUE_CHECKPOINT = os.getenv(
        "ORANGURU_LEAF_VALUE_CHECKPOINT",
        "checkpoints/rl/leaf_value_strength1500.pt",
    )
    LEAF_VALUE_DEVICE = os.getenv("ORANGURU_LEAF_VALUE_DEVICE", "cpu").strip().lower()
    LEAF_VALUE_LOWCONF_ONLY = bool(int(os.getenv("ORANGURU_LEAF_VALUE_LOWCONF_ONLY", "1")))
    LEAF_VALUE_MIN_TURN = int(os.getenv("ORANGURU_LEAF_VALUE_MIN_TURN", "4"))
    LEAF_VALUE_TRIGGER_ABS_MAX = float(os.getenv("ORANGURU_LEAF_VALUE_TRIGGER_ABS_MAX", "0.35"))
    LEAF_VALUE_ESCALATE_MS_MULT = float(os.getenv("ORANGURU_LEAF_VALUE_ESCALATE_MS_MULT", "1.35"))
    LEAF_VALUE_ESCALATE_SAMPLE_MULT = float(os.getenv("ORANGURU_LEAF_VALUE_ESCALATE_SAMPLE_MULT", "1.0"))
    LEAF_VALUE_ESCALATE_MAX_MS = int(os.getenv("ORANGURU_LEAF_VALUE_ESCALATE_MAX_MS", "700"))
    LEAF_VALUE_ESCALATE_MAX_STATES = int(
        os.getenv("ORANGURU_LEAF_VALUE_ESCALATE_MAX_STATES", str(MAX_SAMPLE_STATES))
    )
    SEARCH_TRACE_ENABLED = bool(int(os.getenv("ORANGURU_SEARCH_TRACE", "0")))
    SEARCH_TRACE_OUT = os.getenv(
        "ORANGURU_SEARCH_TRACE_OUT",
        "logs/search_traces/current/search_trace.jsonl",
    )
    SEARCH_TRACE_SOURCE = os.getenv("ORANGURU_SEARCH_TRACE_SOURCE", "oranguru_mcts")
    SEARCH_TRACE_TAG = os.getenv("ORANGURU_SEARCH_TRACE_TAG", "live_search")
    SEARCH_TRACE_MIN_TOTAL = float(os.getenv("ORANGURU_SEARCH_TRACE_MIN_TOTAL", "0.0"))
    SEARCH_TRACE_SKIP_FALLBACK = bool(int(os.getenv("ORANGURU_SEARCH_TRACE_SKIP_FALLBACK", "1")))
    SEARCH_TRACE_INCLUDE_STATE_STR = bool(int(os.getenv("ORANGURU_SEARCH_TRACE_INCLUDE_STATE_STR", "0")))
    SEARCH_TRACE_INCLUDE_FP_ORACLE = bool(
        int(os.getenv("ORANGURU_SEARCH_TRACE_INCLUDE_FP_ORACLE", "0"))
    )
    ADAPTIVE_FALLBACK_ENABLED = bool(int(os.getenv("ORANGURU_ADAPTIVE_FALLBACK", "0")))
    ADAPTIVE_FALLBACK_CONFIDENCE = float(
        os.getenv("ORANGURU_ADAPTIVE_FALLBACK_CONFIDENCE", "0.30")
    )
    ADAPTIVE_FALLBACK_MAX_HEURISTIC = float(
        os.getenv("ORANGURU_ADAPTIVE_FALLBACK_MAX_HEURISTIC", "70.0")
    )
    ADAPTIVE_FALLBACK_TOPK = int(os.getenv("ORANGURU_ADAPTIVE_FALLBACK_TOPK", "4"))
    ADAPTIVE_FALLBACK_MIN_TURN = int(os.getenv("ORANGURU_ADAPTIVE_FALLBACK_MIN_TURN", "4"))
    ADAPTIVE_FALLBACK_COOLDOWN = int(os.getenv("ORANGURU_ADAPTIVE_FALLBACK_COOLDOWN", "3"))
    ADAPTIVE_FALLBACK_MAX_TOP_RATIO = float(
        os.getenv("ORANGURU_ADAPTIVE_FALLBACK_MAX_TOP_RATIO", "0.45")
    )
    ADAPTIVE_FALLBACK_NONDAMAGING_SHARE = float(
        os.getenv("ORANGURU_ADAPTIVE_FALLBACK_NONDAMAGING_SHARE", "0.75")
    )
    ADAPTIVE_FALLBACK_REQUIRE_STALLISH = bool(
        int(os.getenv("ORANGURU_ADAPTIVE_FALLBACK_REQUIRE_STALLISH", "1"))
    )
    ADAPTIVE_FALLBACK_MODE = os.getenv(
        "ORANGURU_ADAPTIVE_FALLBACK_MODE", "super"
    ).strip().lower()
    ADAPTIVE_RERANK_HEUR_WEIGHT = float(
        os.getenv("ORANGURU_ADAPTIVE_RERANK_HEUR_WEIGHT", "0.30")
    )
    ADAPTIVE_RERANK_RISK_WEIGHT = float(
        os.getenv("ORANGURU_ADAPTIVE_RERANK_RISK_WEIGHT", "0.70")
    )
    ADAPTIVE_RERANK_MAX_POLICY_DROP = float(
        os.getenv("ORANGURU_ADAPTIVE_RERANK_MAX_POLICY_DROP", "0.12")
    )
    ADAPTIVE_RERANK_MIN_SCORE_GAIN = float(
        os.getenv("ORANGURU_ADAPTIVE_RERANK_MIN_SCORE_GAIN", "0.03")
    )
    ADAPTIVE_RERANK_MIN_RISK_DELTA = float(
        os.getenv("ORANGURU_ADAPTIVE_RERANK_MIN_RISK_DELTA", "8.0")
    )
    LOOP_BREAKER_ENABLED = bool(int(os.getenv("ORANGURU_LOOP_BREAKER", "0")))
    LOOP_BREAKER_STALL_STREAK = int(os.getenv("ORANGURU_LOOP_BREAKER_STALL_STREAK", "2"))
    LOOP_BREAKER_MIN_SCORE = float(os.getenv("ORANGURU_LOOP_BREAKER_MIN_SCORE", "120.0"))
    LOOP_BREAKER_KO_FRACTION = float(os.getenv("ORANGURU_LOOP_BREAKER_KO_FRACTION", "0.60"))
    LOSS_FORCED_SWITCH_RATIO = float(os.getenv("ORANGURU_LOSS_FORCED_SWITCH_RATIO", "0.35"))
    LOSS_PASSIVE_RATIO = float(os.getenv("ORANGURU_LOSS_PASSIVE_RATIO", "0.18"))
    LOSS_HAZARD_SWITCH_RATIO = float(os.getenv("ORANGURU_LOSS_HAZARD_SWITCH_RATIO", "0.16"))
    LOSS_TEMPO_RATIO = float(os.getenv("ORANGURU_LOSS_TEMPO_RATIO", "0.28"))
    PROGRESS_NEED_MATCHUP = float(os.getenv("ORANGURU_PROGRESS_NEED_MATCHUP", "-0.05"))
    PROGRESS_NEED_REPLY = float(os.getenv("ORANGURU_PROGRESS_NEED_REPLY", "190.0"))
    PROGRESS_NEED_DAMAGE = float(os.getenv("ORANGURU_PROGRESS_NEED_DAMAGE", "140.0"))
    PASSIVE_BREAK_RECOVERY_HP_MAX = float(
        os.getenv("ORANGURU_PASSIVE_BREAK_RECOVERY_HP_MAX", "0.42")
    )
    PASSIVE_REPEAT_HIGH_HP_MAX = float(
        os.getenv("ORANGURU_PASSIVE_REPEAT_HIGH_HP_MAX", "0.70")
    )
    SETUP_PRESSURE_REPLY = float(os.getenv("ORANGURU_SETUP_PRESSURE_REPLY", "160.0"))
    SETUP_PRESSURE_HP_MAX = float(os.getenv("ORANGURU_SETUP_PRESSURE_HP_MAX", "0.55"))
    SETUP_WINDOW_MIN_HP = float(os.getenv("ORANGURU_SETUP_WINDOW_MIN_HP", "0.65"))
    SETUP_WINDOW_MAX_REPLY = float(os.getenv("ORANGURU_SETUP_WINDOW_MAX_REPLY", "110.0"))
    SETUP_WINDOW_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_SETUP_WINDOW_MIN_POLICY_RATIO", "0.65"))
    SETUP_WINDOW_MIN_HEUR_GAIN = float(os.getenv("ORANGURU_SETUP_WINDOW_MIN_HEUR_GAIN", "15.0"))
    SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO", "0.20"))
    SETUP_WINDOW_HIGH_HEUR_GAIN = float(os.getenv("ORANGURU_SETUP_WINDOW_HIGH_HEUR_GAIN", "60.0"))
    RECOVERY_WINDOW_MAX_HP = float(os.getenv("ORANGURU_RECOVERY_WINDOW_MAX_HP", "0.40"))
    RECOVERY_WINDOW_MAX_REPLY = float(os.getenv("ORANGURU_RECOVERY_WINDOW_MAX_REPLY", "110.0"))
    RECOVERY_WINDOW_MIN_OPP_HP = float(os.getenv("ORANGURU_RECOVERY_WINDOW_MIN_OPP_HP", "0.25"))
    RECOVERY_WINDOW_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_RECOVERY_WINDOW_MIN_POLICY_RATIO", "0.65"))
    RECOVERY_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_RECOVERY_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO", "0.45"))
    RECOVERY_WINDOW_CRITICAL_HP = float(os.getenv("ORANGURU_RECOVERY_WINDOW_CRITICAL_HP", "0.30"))
    RECOVERY_WINDOW_CRITICAL_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_RECOVERY_WINDOW_CRITICAL_MIN_POLICY_RATIO", "0.33"))
    RECOVERY_WINDOW_MIN_HEUR_GAIN = float(os.getenv("ORANGURU_RECOVERY_WINDOW_MIN_HEUR_GAIN", "1.0"))
    RECOVERY_WINDOW_HIGH_HEUR_GAIN = float(os.getenv("ORANGURU_RECOVERY_WINDOW_HIGH_HEUR_GAIN", "10.0"))
    SWITCH_GUARD_MIN_ACTIVE_HP = float(os.getenv("ORANGURU_SWITCH_GUARD_MIN_ACTIVE_HP", "0.45"))
    SWITCH_GUARD_POLICY_RATIO = float(os.getenv("ORANGURU_SWITCH_GUARD_POLICY_RATIO", "0.70"))
    SWITCH_GUARD_HEUR_GAIN = float(os.getenv("ORANGURU_SWITCH_GUARD_HEUR_GAIN", "1.0"))
    SWITCH_GUARD_RISK_POLICY_RATIO = float(os.getenv("ORANGURU_SWITCH_GUARD_RISK_POLICY_RATIO", "0.60"))
    SWITCH_GUARD_RISK_MIN_RISK = float(os.getenv("ORANGURU_SWITCH_GUARD_RISK_MIN_RISK", "20.0"))
    SWITCH_GUARD_RISK_HEUR_FLOOR = float(os.getenv("ORANGURU_SWITCH_GUARD_RISK_HEUR_FLOOR", "-0.5"))
    PROGRESS_WINDOW_MIN_ACTIVE_HP = float(os.getenv("ORANGURU_PROGRESS_WINDOW_MIN_ACTIVE_HP", "0.50"))
    PROGRESS_WINDOW_MIN_OPP_HP = float(os.getenv("ORANGURU_PROGRESS_WINDOW_MIN_OPP_HP", "0.55"))
    PROGRESS_WINDOW_MAX_REPLY = float(os.getenv("ORANGURU_PROGRESS_WINDOW_MAX_REPLY", "110.0"))
    PROGRESS_WINDOW_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_PROGRESS_WINDOW_MIN_POLICY_RATIO", "0.65"))
    PROGRESS_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = float(os.getenv("ORANGURU_PROGRESS_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO", "0.30"))
    PROGRESS_WINDOW_MIN_HEUR_GAIN = float(os.getenv("ORANGURU_PROGRESS_WINDOW_MIN_HEUR_GAIN", "1.0"))
    PROGRESS_WINDOW_HIGH_HEUR_GAIN = float(os.getenv("ORANGURU_PROGRESS_WINDOW_HIGH_HEUR_GAIN", "10.0"))
    ADAPTIVE_ESCALATE_ENABLED = bool(
        int(os.getenv("ORANGURU_ADAPTIVE_ESCALATE", "1"))
    )
    ADAPTIVE_ESCALATE_MS_MULT = float(
        os.getenv("ORANGURU_ADAPTIVE_ESCALATE_MS_MULT", "1.8")
    )
    ADAPTIVE_ESCALATE_SAMPLE_MULT = float(
        os.getenv("ORANGURU_ADAPTIVE_ESCALATE_SAMPLE_MULT", "1.5")
    )
    ADAPTIVE_ESCALATE_MAX_MS = int(
        os.getenv("ORANGURU_ADAPTIVE_ESCALATE_MAX_MS", "1200")
    )
    ADAPTIVE_ESCALATE_MAX_STATES = int(
        os.getenv("ORANGURU_ADAPTIVE_ESCALATE_MAX_STATES", str(MAX_SAMPLE_STATES))
    )
    _VOLATILE_RAW = {
        _maybe_effect("CONFUSION"): constants.CONFUSION,
        _maybe_effect("LEECH_SEED"): constants.LEECH_SEED,
        _maybe_effect("SUBSTITUTE"): constants.SUBSTITUTE,
        _maybe_effect("TAUNT"): constants.TAUNT,
        _maybe_effect("ENCORE"): "encore",
        _maybe_effect("LOCKED_MOVE"): constants.LOCKED_MOVE,
        _maybe_effect("YAWN"): constants.YAWN,
        _maybe_effect("SLOW_START"): constants.SLOW_START,
        _maybe_effect("PROTECT"): constants.PROTECT,
        _maybe_effect("BANEFUL_BUNKER"): constants.BANEFUL_BUNKER,
        _maybe_effect("SPIKY_SHIELD"): constants.SPIKY_SHIELD,
        _maybe_effect("SILK_TRAP"): constants.SILK_TRAP,
        _maybe_effect("ENDURE"): constants.ENDURE,
        _maybe_effect("PARTIALLY_TRAPPED"): constants.PARTIALLY_TRAPPED,
        _maybe_effect("ROOST"): constants.ROOST,
        _maybe_effect("DYNAMAX"): constants.DYNAMAX,
        _maybe_effect("TRANSFORM"): constants.TRANSFORM,
    }
    VOLATILE_EFFECT_MAP = {k: v for k, v in _VOLATILE_RAW.items() if k is not None}
    IMMUNITY_ABILITY_MAP = {
        "electric": {"lightningrod", "motordrive", "voltabsorb"},
        "water": {"waterabsorb", "stormdrain", "dryskin"},
        "fire": {"flashfire", "wellbakedbody"},
        "ground": {"levitate", "eartheater"},
        "grass": {"sapsipper"},
    }
    DAMAGE_BELIEF_UNSTABLE_MOVES = {
        "acrobatics",
        "avalanche",
        "beatup",
        "brine",
        "counter",
        "crushgrip",
        "dragonenergy",
        "electroball",
        "endeavor",
        "eruption",
        "facade",
        "ficklebeam",
        "finalgambit",
        "flail",
        "foulplay",
        "grassknot",
        "gyroball",
        "heatcrash",
        "hex",
        "hydrosteam",
        "knockoff",
        "lastrespects",
        "lowkick",
        "magnitude",
        "metalburst",
        "mirrorcoat",
        "naturesmadness",
        "nightshade",
        "payback",
        "powertrip",
        "present",
        "psywave",
        "ragefist",
        "retaliate",
        "reversal",
        "risingvoltage",
        "ruination",
        "seismictoss",
        "spitup",
        "storedpower",
        "superfang",
        "terrainpulse",
        "waterspout",
        "weatherball",
        "wringout",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcts_pool = None
        self._pool_workers = 0
        self._randbats_initialized = False
        self._randbats_gen = None
        self._randbats_sanitized = False
        self._mcts_stats = oranguru_diag.init_mcts_stats()
        self._rl_prior_ready = False
        self._rl_prior_failed = False
        self._rl_prior_model = None
        self._rl_prior_feature_builder = None
        self._rl_prior_device = "cpu"
        self._rl_prior_torch = None
        self._rl_prior_checkpoint = ""
        self._search_prior_ready = False
        self._search_prior_failed = False
        self._search_prior_model = None
        self._search_prior_feature_builder = None
        self._search_prior_device = "cpu"
        self._search_prior_torch = None
        self._search_prior_checkpoint = ""
        self._switch_prior_ready = False
        self._switch_prior_failed = False
        self._switch_prior_model = None
        self._switch_prior_feature_builder = None
        self._switch_prior_device = "cpu"
        self._switch_prior_torch = None
        self._switch_prior_checkpoint = ""
        self._passive_breaker_ready = False
        self._passive_breaker_failed = False
        self._passive_breaker_model = None
        self._passive_breaker_feature_builder = None
        self._passive_breaker_device = "cpu"
        self._passive_breaker_torch = None
        self._passive_breaker_checkpoint = ""
        self._tera_pruner_ready = False
        self._tera_pruner_failed = False
        self._tera_pruner_model = None
        self._tera_pruner_feature_builder = None
        self._tera_pruner_device = "cpu"
        self._tera_pruner_torch = None
        self._tera_pruner_checkpoint = ""
        self._world_ranker_ready = False
        self._world_ranker_failed = False
        self._world_ranker_model = None
        self._world_ranker_device = "cpu"
        self._world_ranker_torch = None
        self._world_ranker_checkpoint = ""
        self._leaf_value_ready = False
        self._leaf_value_failed = False
        self._leaf_value_model = None
        self._leaf_value_device = "cpu"
        self._leaf_value_torch = None
        self._leaf_value_checkpoint = ""
        self._diag_finished_battle_tags = set()
        self._search_trace_finished_battle_tags = set()
        self._search_trace_builder = None
        self._search_trace_builder_failed = False

    _search_trace_token_hash = staticmethod(oranguru_trace.search_trace_token_hash)
    _search_trace_species_hash = oranguru_trace.search_trace_species_hash
    _init_search_trace_builder = oranguru_trace.init_search_trace_builder
    _load_search_prior_family_model = oranguru_models.load_search_prior_family_model
    _resolve_model_checkpoint = oranguru_models.resolve_model_checkpoint
    _resolve_model_device = oranguru_models.resolve_model_device
    _init_rl_prior = oranguru_models.init_rl_prior
    _init_search_prior = oranguru_models.init_search_prior
    _init_switch_prior = oranguru_models.init_switch_prior
    _init_passive_breaker = oranguru_models.init_passive_breaker
    _init_tera_pruner = oranguru_models.init_tera_pruner
    _init_world_ranker = oranguru_models.init_world_ranker
    _init_leaf_value = oranguru_models.init_leaf_value

    def _build_rl_action_mask_and_maps(
        self, battle: Battle
    ) -> Tuple[List[bool], Dict[str, int], Dict[str, int]]:
        mask = [False] * 13
        move_map: Dict[str, int] = {}
        switch_map: Dict[str, int] = {}

        if battle.force_switch:
            for i in range(min(5, len(battle.available_switches))):
                sw = battle.available_switches[i]
                mask[4 + i] = True
                switch_map[normalize_name(sw.species)] = 4 + i
            return mask, move_map, switch_map

        for i in range(min(4, len(battle.available_moves))):
            move = battle.available_moves[i]
            move_id = normalize_name(getattr(move, "id", ""))
            mask[i] = True
            if move_id:
                move_map[move_id] = i

        for i in range(min(5, len(battle.available_switches))):
            sw = battle.available_switches[i]
            sw_id = normalize_name(sw.species)
            mask[4 + i] = True
            if sw_id:
                switch_map[sw_id] = 4 + i

        if getattr(battle, "can_tera", False):
            for i in range(min(4, len(battle.available_moves))):
                mask[9 + i] = True
        return mask, move_map, switch_map

    def _choice_to_rl_action_idx(
        self,
        choice: str,
        mask: List[bool],
        move_map: Dict[str, int],
        switch_map: Dict[str, int],
    ) -> Optional[int]:
        if not choice:
            return None
        raw = choice.strip()
        if raw.startswith("switch "):
            target = normalize_name(raw.split("switch ", 1)[1])
            idx = switch_map.get(target)
            if idx is not None and 0 <= idx < len(mask) and mask[idx]:
                return idx
            return None

        tera = raw.endswith("-tera")
        move_id = normalize_name(raw.replace("-tera", ""))
        idx = move_map.get(move_id)
        if idx is None:
            return None
        if tera:
            tera_idx = idx + 9
            if 0 <= tera_idx < len(mask) and mask[tera_idx]:
                return tera_idx
        if 0 <= idx < len(mask) and mask[idx]:
            return idx
        return None

    _rl_choice_priors = oranguru_models.rl_choice_priors
    _search_choice_priors = oranguru_models.search_choice_priors
    _switch_choice_priors = oranguru_models.switch_choice_priors
    _passive_break_choice_priors = oranguru_models.passive_break_choice_priors
    _tera_choice_priors = oranguru_models.tera_choice_priors

    def _apply_switch_prior_prune(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        confidence: float,
        threshold: float,
    ) -> List[Tuple[str, float]]:
        if not ordered or not self.SWITCH_PRIOR_ENABLED:
            return ordered
        if self.SWITCH_PRIOR_LOWCONF_ONLY and confidence >= threshold:
            return ordered

        switch_positions = [i for i, (choice, _) in enumerate(ordered) if choice.startswith("switch ")]
        min_candidates = max(2, self.SWITCH_PRIOR_MIN_CANDIDATES)
        if len(switch_positions) < min_candidates:
            return ordered

        switch_choices = [ordered[i][0] for i in switch_positions]
        priors = self._switch_choice_priors(battle, switch_choices)
        if not priors:
            return ordered

        keep_topk = max(1, self.SWITCH_PRIOR_KEEP_TOPK)
        top_mcts_switch_pos = switch_positions[0]
        ranked = sorted(
            zip(switch_positions, switch_choices, priors),
            key=lambda item: item[2],
            reverse=True,
        )
        keep_positions = {top_mcts_switch_pos}
        for pos, _, _ in ranked:
            keep_positions.add(pos)
            if len(keep_positions) >= keep_topk:
                break

        pruned = []
        pruned_count = 0
        for idx, item in enumerate(ordered):
            if idx in switch_positions and idx not in keep_positions:
                pruned_count += 1
                continue
            pruned.append(item)
        if pruned_count > 0:
            self._mcts_stats["switch_prior_used"] = int(self._mcts_stats.get("switch_prior_used", 0) or 0) + 1
            self._mcts_stats["switch_prior_pruned"] = int(self._mcts_stats.get("switch_prior_pruned", 0) or 0) + pruned_count
        return pruned

    def _maybe_passive_break_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        confidence: float,
        threshold: float,
    ) -> Optional[str]:
        if not ordered or not self.PASSIVE_BREAKER_ENABLED:
            return None
        if self.PASSIVE_BREAKER_LOWCONF_ONLY and confidence >= threshold:
            return None

        top_choice = ordered[0][0]
        top_kind = self._search_trace_choice_kind(battle, top_choice)
        if top_kind not in {"protect", "recovery", "status", "setup"}:
            return None

        topk = max(2, self.PASSIVE_BREAKER_TOPK)
        candidates = ordered[:topk]
        choices = [choice for choice, _ in candidates]
        priors = self._passive_break_choice_priors(battle, choices)
        if not priors:
            return None

        ranked = sorted(zip(choices, priors), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None
        best_choice, best_prob = ranked[0]
        top_prob = 0.0
        for choice, prob in zip(choices, priors):
            if choice == top_choice:
                top_prob = prob
                break
        best_kind = self._search_trace_choice_kind(battle, best_choice)
        if best_choice == top_choice or best_kind not in {"attack", "tera_attack", "switch"}:
            return None
        if best_prob < self.PASSIVE_BREAKER_MIN_PROB:
            return None
        if (best_prob - top_prob) < self.PASSIVE_BREAKER_MIN_MARGIN:
            return None
        self._mcts_stats["passive_breaker_used"] = int(self._mcts_stats.get("passive_breaker_used", 0) or 0) + 1
        return best_choice

    def _apply_tera_prune(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        confidence: float,
        threshold: float,
    ) -> List[Tuple[str, float]]:
        if not ordered or not self.TERA_PRUNER_ENABLED:
            return ordered
        if self.TERA_PRUNER_LOWCONF_ONLY and confidence >= threshold:
            return ordered
        if not bool(getattr(battle, "can_tera", False)):
            return ordered

        tera_positions = [i for i, (choice, _) in enumerate(ordered) if choice.endswith("-tera")]
        min_candidates = max(2, self.TERA_PRUNER_MIN_CANDIDATES)
        if len(tera_positions) < min_candidates:
            return ordered

        tera_choices = [ordered[i][0] for i in tera_positions]
        priors = self._tera_choice_priors(battle, tera_choices)
        if not priors:
            return ordered

        keep_topk = max(1, self.TERA_PRUNER_KEEP_TOPK)
        top_mcts_tera_pos = tera_positions[0]
        ranked = sorted(zip(tera_positions, tera_choices, priors), key=lambda item: item[2], reverse=True)
        keep_positions = {top_mcts_tera_pos}
        for pos, _, _ in ranked:
            keep_positions.add(pos)
            if len(keep_positions) >= keep_topk:
                break

        pruned = []
        pruned_count = 0
        for idx, item in enumerate(ordered):
            if idx in tera_positions and idx not in keep_positions:
                pruned_count += 1
                continue
            pruned.append(item)
        if pruned_count > 0:
            self._mcts_stats["tera_pruner_used"] = int(self._mcts_stats.get("tera_pruner_used", 0) or 0) + 1
            self._mcts_stats["tera_pruner_pruned"] = int(self._mcts_stats.get("tera_pruner_pruned", 0) or 0) + pruned_count
        return pruned

    def _should_trigger_adaptive_fallback(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        confidence: float,
        threshold: float,
        record_diag: bool = True,
    ) -> bool:
        mem = self._get_battle_memory(battle)
        def _record(reason: str) -> None:
            if not record_diag:
                return
            self._diag_record_adaptive_reason(reason)
            mem["diag_last_adaptive_reason"] = reason

        if not self.ADAPTIVE_FALLBACK_ENABLED:
            _record("disabled")
            return False
        if not ordered:
            _record("empty")
            return False
        if int(getattr(battle, "turn", 0) or 0) < max(1, self.ADAPTIVE_FALLBACK_MIN_TURN):
            _record("early_turn")
            return False

        last_turn = int(mem.get("adaptive_fallback_last_turn", -999) or -999)
        now_turn = int(getattr(battle, "turn", 0) or 0)
        if now_turn - last_turn < max(0, self.ADAPTIVE_FALLBACK_COOLDOWN):
            _record("cooldown")
            return False

        conf_gate = min(max(0.0, self.ADAPTIVE_FALLBACK_CONFIDENCE), threshold)
        if confidence > conf_gate:
            _record("confidence")
            return False

        topk = max(1, self.ADAPTIVE_FALLBACK_TOPK)
        candidates = ordered[:topk]
        top_total = sum(max(0.0, w) for _, w in candidates)
        if top_total > 0:
            top_ratio = max(0.0, candidates[0][1]) / top_total
            if top_ratio > max(0.0, min(1.0, self.ADAPTIVE_FALLBACK_MAX_TOP_RATIO)):
                _record("top_ratio")
                return False

        heuristics: List[float] = []
        nondamaging = 0
        for choice, _ in candidates:
            score = self._heuristic_action_score(battle, choice)
            if score is not None:
                heuristics.append(float(score))
            move_id = normalize_name(choice.replace("-tera", ""))
            is_damaging = False
            for move in battle.available_moves or []:
                if normalize_name(getattr(move, "id", "")) != move_id:
                    continue
                category = getattr(move, "category", None)
                base_power = float(getattr(move, "base_power", 0) or 0)
                damage_attr = getattr(move, "damage", None)
                is_damaging = bool(
                    base_power > 0
                    or damage_attr is not None
                    or (category is not None and category != MoveCategory.STATUS)
                )
                break
            if not is_damaging:
                nondamaging += 1
            if is_damaging and score is not None and float(score) >= (
                max(0.0, self.ADAPTIVE_FALLBACK_MAX_HEURISTIC) * 0.90
            ):
                _record("damaging_available")
                return False
        if not heuristics:
            _record("no_heuristics")
            return False
        if max(heuristics) > max(0.0, self.ADAPTIVE_FALLBACK_MAX_HEURISTIC):
            _record("high_heuristic")
            return False

        nondamaging_share = nondamaging / max(1, len(candidates))
        if self.ADAPTIVE_FALLBACK_REQUIRE_STALLISH:
            status_stall = int(mem.get("status_stall_streak", 0) or 0)
            if (
                nondamaging_share
                < max(0.0, min(1.0, self.ADAPTIVE_FALLBACK_NONDAMAGING_SHARE))
                and status_stall < 1
            ):
                _record("not_stallish")
                return False
        _record("triggered")
        return True

    get_mcts_stats = oranguru_diag.get_mcts_stats

    _build_search_trace_action_features = oranguru_trace.build_search_trace_action_features
    _search_trace_phase = oranguru_trace.search_trace_phase
    _search_trace_status_code = oranguru_trace.search_trace_status_code
    _build_world_rank_features = oranguru_worlds.build_world_rank_features
    _build_world_plausibility_features = oranguru_worlds.build_world_plausibility_features
    _is_low_uncertainty_turn = oranguru_worlds.is_low_uncertainty_turn
    _is_endgame_turn = oranguru_worlds.is_endgame_turn
    _world_ranker_turn_allowed = oranguru_worlds.world_ranker_turn_allowed

    def _build_world_ranker_input_features(
        self,
        battle: Battle,
        fp_battle: FPBattle,
        sample_weight: float,
    ) -> tuple[list[float], list[float]] | tuple[None, None]:
        return oranguru_worlds.build_world_ranker_input_features(
            self, battle, fp_battle, sample_weight
        )

    def _build_state_value_features(
        self,
        battle: Battle,
        *,
        phase: Optional[str] = None,
        hazard_load: Optional[float] = None,
        matchup_score: Optional[float] = None,
        best_reply_score: Optional[float] = None,
    ) -> list[float]:
        if phase is None:
            phase = self._search_trace_phase(battle)
        if hazard_load is None:
            try:
                hazard_load = float(self._side_hazard_pressure(battle))
            except Exception:
                hazard_load = 0.0
        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if matchup_score is None:
            if active is not None and opponent is not None:
                try:
                    matchup_score = float(self._estimate_matchup(active, opponent))
                except Exception:
                    matchup_score = 0.0
            else:
                matchup_score = 0.0
        if best_reply_score is None:
            if active is not None and opponent is not None:
                try:
                    best_reply_score = float(self._estimate_best_reply_score(opponent, active, battle))
                except Exception:
                    best_reply_score = 0.0
            else:
                best_reply_score = 0.0
        mask, _, _ = self._build_rl_action_mask_and_maps(battle)
        switch_candidate_count = sum(1 for idx in range(4, 9) if idx < len(mask) and mask[idx])
        tera_candidate_count = sum(1 for idx in range(9, 13) if idx < len(mask) and mask[idx])
        return [
            min(1.0, float(getattr(battle, "turn", 0) or 0) / 30.0),
            1.0 if phase == "opening" else 0.0,
            1.0 if phase == "mid" else 0.0,
            1.0 if phase == "end" else 0.0,
            1.0 if bool(getattr(battle, "can_tera", False)) else 0.0,
            min(1.0, float(switch_candidate_count) / 5.0),
            min(1.0, float(tera_candidate_count) / 4.0),
            max(-1.0, min(1.0, float(hazard_load))),
            max(-1.0, min(1.0, float(matchup_score))),
            max(-1.0, min(1.0, float(best_reply_score))),
        ]

    _predict_leaf_value = oranguru_models.predict_leaf_value

    def _should_trigger_leaf_value_escalation(
        self,
        battle: Battle,
        confidence: float,
        threshold: float,
    ) -> bool:
        if not self.LEAF_VALUE_ENABLED:
            return False
        if int(getattr(battle, "turn", 0) or 0) < max(1, self.LEAF_VALUE_MIN_TURN):
            return False
        if self.LEAF_VALUE_LOWCONF_ONLY and confidence >= threshold:
            return False
        pred = self._predict_leaf_value(battle)
        if pred is None:
            return False
        return abs(pred) <= max(0.0, min(1.0, self.LEAF_VALUE_TRIGGER_ABS_MAX))

    _rank_and_trim_worlds = oranguru_worlds.rank_and_trim_worlds
    _build_world_candidate_summary = oranguru_worlds.build_world_candidate_summary
    _build_search_trace_action_labels = oranguru_trace.build_search_trace_action_labels
    _serialize_fp_last_used_move = staticmethod(oranguru_trace.serialize_fp_last_used_move)
    _serialize_fp_move = staticmethod(oranguru_trace.serialize_fp_move)
    _serialize_fp_pokemon = staticmethod(oranguru_trace.serialize_fp_pokemon)
    _serialize_fp_battler = oranguru_trace.serialize_fp_battler
    _serialize_fp_battle = oranguru_trace.serialize_fp_battle
    _apply_world_budget_controls = oranguru_worlds.apply_world_budget_controls
    _search_trace_choice_kind = oranguru_trace.search_trace_choice_kind

    def _append_search_trace_example(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        chosen_choice: str,
        confidence: float,
        threshold: float,
        path: str,
        world_candidates: Optional[List[dict]] = None,
    ) -> None:
        return oranguru_trace.append_search_trace_example(
            self,
            battle,
            ordered,
            chosen_choice,
            confidence,
            threshold,
            path,
            world_candidates=world_candidates,
        )

    _flush_finished_search_traces = oranguru_trace.flush_finished_search_traces
    _diag_record_adaptive_reason = oranguru_diag.diag_record_adaptive_reason

    def _diag_record_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        chosen: str,
        confidence: float,
        threshold: float,
        path: str,
    ) -> None:
        return oranguru_diag.diag_record_choice(
            self, battle, ordered, chosen, confidence, threshold, path
        )

    _flush_finished_battle_diags = oranguru_diag.flush_finished_battle_diags

    def _get_mcts_pool(self, desired_workers: int) -> Optional[ProcessPoolExecutor]:
        if desired_workers <= 1:
            return None
        if self._mcts_pool is None or self._pool_workers != desired_workers:
            if self._mcts_pool is not None:
                try:
                    self._mcts_pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._mcts_pool = ProcessPoolExecutor(max_workers=desired_workers)
            self._pool_workers = desired_workers
        return self._mcts_pool

    def close(self):
        self._flush_finished_search_traces()
        if self._mcts_pool is not None:
            try:
                self._mcts_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._mcts_pool = None
            self._pool_workers = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    _status_to_fp = oranguru_state.status_to_fp
    _weather_turns_remaining = oranguru_memory.weather_turns_remaining
    _terrain_turns_remaining = oranguru_memory.terrain_turns_remaining
    _is_trapped = oranguru_state.is_trapped
    _boosts_to_fp = oranguru_state.boosts_to_fp
    _fill_moves_from_set = oranguru_state.fill_moves_from_set
    _poke_env_to_fp = oranguru_state.poke_env_to_fp
    _map_side_conditions = oranguru_state.map_side_conditions
    _map_weather = oranguru_state.map_weather
    _map_terrain = oranguru_state.map_terrain
    _clear_invalid_encore = oranguru_state.clear_invalid_encore
    _sleep_clause_active = oranguru_state.sleep_clause_active
    _opponent_has_sleeping_mon = oranguru_state.opponent_has_sleeping_mon
    _sleep_clause_blocked = oranguru_state.sleep_clause_blocked
    _move_inflicts_sleep = oranguru_state.move_inflicts_sleep
    _fp_move_inflicts_sleep = oranguru_state.fp_move_inflicts_sleep
    _sleep_clause_banned_choices = oranguru_state.sleep_clause_banned_choices
    _apply_opponent_item_flags = oranguru_state.apply_opponent_item_flags
    _apply_opponent_ability_flags = oranguru_state.apply_opponent_ability_flags
    _apply_known_opponent_moves = oranguru_state.apply_known_opponent_moves
    _apply_opponent_switch_memory = oranguru_state.apply_opponent_switch_memory
    _apply_speed_bounds = oranguru_state.apply_speed_bounds
    _damage_belief_has_unmodeled_state = oranguru_belief.damage_belief_has_unmodeled_state
    _damage_belief_observations = oranguru_belief.damage_belief_observations
    _side_hazard_pressure = oranguru_state.side_hazard_pressure
    _opponent_progress_markers = oranguru_state.opponent_progress_markers
    _resolve_passive_progress = oranguru_state.resolve_passive_progress
    _passive_choice_kind = oranguru_state.passive_choice_kind

    def _progress_need_score(
        self,
        battle: Battle,
        active: Pokemon,
        opponent: Pokemon,
        best_damage_score: float,
    ) -> int:
        return oranguru_state.progress_need_score(self, battle, active, opponent, best_damage_score)

    _record_last_action = oranguru_memory.record_last_action
    _update_damage_observation = oranguru_memory.update_damage_observation
    _parse_hp_fraction = staticmethod(oranguru_memory.parse_hp_fraction)
    _ensure_randbats_sets = oranguru_state.ensure_randbats_sets
    _sanitize_randbats_moves = oranguru_state.sanitize_randbats_moves

    def _belief_weight_for_set(
        self,
        fp_mon: FPPokemon,
        set_info: dict,
        battle: Battle,
        apply_damage: bool = True,
    ) -> float:
        return oranguru_state.belief_weight_for_set(
            self, fp_mon, set_info, battle, apply_damage=apply_damage
        )

    _candidate_randombattle_sets = oranguru_state.candidate_randombattle_sets
    _extract_last_opponent_move = oranguru_worlds.extract_last_opponent_move

    def _sample_set_for_species(
        self,
        species: str,
        battle: Battle,
        mon: Optional[Pokemon] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[dict]:
        return oranguru_worlds.sample_set_for_species(self, species, battle, mon=mon, rng=rng)

    def _sample_unknown_opponents(
        self,
        battle: Battle,
        taken: set,
        count: int,
        rng: Optional[random.Random] = None,
    ) -> List[FPPokemon]:
        return oranguru_worlds.sample_unknown_opponents(self, battle, taken, count, rng=rng)

    def _build_fp_battle(self, battle: Battle, seed: int, fill_opponent_sets: bool = False) -> FPBattle:
        return oranguru_worlds.build_fp_battle(
            self, battle, seed, fill_opponent_sets=fill_opponent_sets
        )

    _fp_move_id = oranguru_worlds.fp_move_id

    def _best_damaging_move(self, battle: Battle, active: Pokemon, opponent: Pokemon):
        best_move = None
        best_score = 0.0
        for move in battle.available_moves:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except Exception:
                continue
            score = self._calculate_move_score(move, active, opponent, battle)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    _switch_faints_to_entry_hazards = oranguru_tactical.switch_faints_to_entry_hazards

    def _status_choice_is_obviously_bad(
        self,
        battle: Battle,
        move,
        active: Pokemon,
        opponent: Pokemon,
    ) -> bool:
        return oranguru_tactical.status_choice_is_obviously_bad(self, battle, move, active, opponent)

    _apply_tactical_safety = oranguru_tactical.apply_tactical_safety
    _heuristic_action_score = oranguru_decision.heuristic_action_score
    _adaptive_choice_risk_penalty = oranguru_decision.adaptive_choice_risk_penalty
    _maybe_force_finish_blow_choice = oranguru_decision.maybe_force_finish_blow_choice
    _maybe_take_setup_window_choice = oranguru_decision.maybe_take_setup_window_choice
    _maybe_take_safe_recovery_choice = oranguru_decision.maybe_take_safe_recovery_choice
    _maybe_take_progress_when_behind_choice = oranguru_decision.maybe_take_progress_when_behind_choice
    _maybe_reduce_negative_matchup_switch = oranguru_decision.maybe_reduce_negative_matchup_switch

    def _adaptive_rerank_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        topk: int,
    ) -> str:
        return oranguru_decision.adaptive_rerank_choice(self, battle, ordered, topk)

    _aggregate_policy_from_results = oranguru_decision.aggregate_policy_from_results
    _collect_mcts_results = oranguru_worlds.collect_mcts_results
    _select_move_from_results = oranguru_decision.select_move_from_results
    _is_damaging_move_choice = oranguru_decision.is_damaging_move_choice

    _choose_adaptive_fallback_order = oranguru_decision.choose_adaptive_fallback_order

    def choose_move(self, battle: AbstractBattle):
        return oranguru_searchflow.choose_move(self, battle)
