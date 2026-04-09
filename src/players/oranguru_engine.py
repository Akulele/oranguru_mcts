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
from src.players import oranguru_memory
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

    @staticmethod
    def _search_trace_token_hash(token: str) -> float:
        return oranguru_trace.search_trace_token_hash(token)

    @classmethod
    def _search_trace_species_hash(cls, species: str) -> float:
        return oranguru_trace.search_trace_species_hash(cls, species)

    def _init_search_trace_builder(self):
        return oranguru_trace.init_search_trace_builder(self)

    def _load_search_prior_family_model(self, checkpoint_path: str, device_name: str):
        return oranguru_models.load_search_prior_family_model(self, checkpoint_path, device_name)

    @staticmethod
    def _resolve_model_checkpoint(path_str: str) -> Path:
        return oranguru_models.resolve_model_checkpoint(None, path_str)

    @staticmethod
    def _resolve_model_device(device_name: str, torch_module) -> str:
        return oranguru_models.resolve_model_device(None, device_name, torch_module)

    def _init_rl_prior(self) -> bool:
        return oranguru_models.init_rl_prior(self)

    def _init_search_prior(self) -> bool:
        return oranguru_models.init_search_prior(self)

    def _init_switch_prior(self) -> bool:
        return oranguru_models.init_switch_prior(self)

    def _init_passive_breaker(self) -> bool:
        return oranguru_models.init_passive_breaker(self)

    def _init_tera_pruner(self) -> bool:
        return oranguru_models.init_tera_pruner(self)

    def _init_world_ranker(self) -> bool:
        return oranguru_models.init_world_ranker(self)

    def _init_leaf_value(self) -> bool:
        return oranguru_models.init_leaf_value(self)

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

    def _rl_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
        return oranguru_models.rl_choice_priors(self, battle, choices)

    def _search_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
        return oranguru_models.search_choice_priors(self, battle, choices)

    def _switch_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
        return oranguru_models.switch_choice_priors(self, battle, choices)

    def _passive_break_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
        return oranguru_models.passive_break_choice_priors(self, battle, choices)

    def _tera_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
        return oranguru_models.tera_choice_priors(self, battle, choices)

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

    def get_mcts_stats(self) -> Dict[str, float]:
        return oranguru_diag.get_mcts_stats(self)

    def _build_search_trace_action_features(
        self,
        battle: Battle,
        mask: List[bool],
    ) -> List[List[float]]:
        return oranguru_trace.build_search_trace_action_features(self, battle, mask)

    def _search_trace_phase(self, battle: Battle) -> str:
        return oranguru_trace.search_trace_phase(self, battle)

    @staticmethod
    def _search_trace_status_code(status: object) -> float:
        return oranguru_trace.search_trace_status_code(None, status)

    def _build_world_rank_features(self, fp_battle: FPBattle) -> List[float]:
        return oranguru_worlds.build_world_rank_features(self, fp_battle)

    def _build_world_plausibility_features(self, fp_battle: FPBattle) -> List[float]:
        return oranguru_worlds.build_world_plausibility_features(self, fp_battle)

    def _is_low_uncertainty_turn(self, battle: Battle) -> bool:
        return oranguru_worlds.is_low_uncertainty_turn(self, battle)

    def _is_endgame_turn(self, battle: Battle) -> bool:
        return oranguru_worlds.is_endgame_turn(self, battle)

    def _world_ranker_turn_allowed(self, battle: Battle) -> bool:
        return oranguru_worlds.world_ranker_turn_allowed(self, battle)

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

    def _predict_leaf_value(self, battle: Battle) -> Optional[float]:
        return oranguru_models.predict_leaf_value(self, battle)

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

    def _rank_and_trim_worlds(
        self,
        battle: Battle,
        fp_battles: List[FPBattle],
        weights: List[float],
    ) -> tuple[List[FPBattle], List[float]]:
        return oranguru_worlds.rank_and_trim_worlds(self, battle, fp_battles, weights)

    def _build_world_candidate_summary(
        self,
        battle: Battle,
        fp_battle: FPBattle,
        sample_weight: float,
        result,
        state_str: str = "",
    ) -> dict:
        return oranguru_worlds.build_world_candidate_summary(
            self, battle, fp_battle, sample_weight, result, state_str=state_str
        )

    def _build_search_trace_action_labels(
        self,
        battle: Battle,
        mask: List[bool],
    ) -> List[str]:
        return oranguru_trace.build_search_trace_action_labels(self, battle, mask)

    @staticmethod
    def _serialize_fp_last_used_move(move: LastUsedMove) -> dict:
        return oranguru_trace.serialize_fp_last_used_move(move)

    @staticmethod
    def _serialize_fp_move(move) -> dict:
        return oranguru_trace.serialize_fp_move(move)

    @staticmethod
    def _serialize_fp_pokemon(mon: Optional[FPPokemon]) -> Optional[dict]:
        return oranguru_trace.serialize_fp_pokemon(mon)

    @classmethod
    def _serialize_fp_battler(cls, battler: Battler) -> dict:
        return oranguru_trace.serialize_fp_battler(cls, battler)

    @classmethod
    def _serialize_fp_battle(cls, battle: FPBattle) -> dict:
        return oranguru_trace.serialize_fp_battle(cls, battle)

    def _apply_world_budget_controls(self, battle: Battle, sample_states: int) -> int:
        return oranguru_worlds.apply_world_budget_controls(self, battle, sample_states)

    def _search_trace_choice_kind(self, battle: Battle, choice: str) -> str:
        return oranguru_trace.search_trace_choice_kind(self, battle, choice)

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

    def _flush_finished_search_traces(self) -> None:
        return oranguru_trace.flush_finished_search_traces(self)

    def _diag_record_adaptive_reason(self, reason: str) -> None:
        return oranguru_diag.diag_record_adaptive_reason(self, reason)

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

    def _flush_finished_battle_diags(self) -> None:
        return oranguru_diag.flush_finished_battle_diags(self)

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

    def _status_to_fp(self, status) -> Optional[str]:
        if status is None:
            return None
        status_id = normalize_name(str(status))
        if status_id in {"slp", "sleep"}:
            return constants.SLEEP
        if status_id in {"brn", "burn"}:
            return constants.BURN
        if status_id in {"frz", "freeze"}:
            return constants.FROZEN
        if status_id in {"par", "paralysis"}:
            return constants.PARALYZED
        if status_id in {"psn", "poison"}:
            return constants.POISON
        if status_id in {"tox", "toxic"}:
            return constants.TOXIC
        return None

    def _weather_turns_remaining(self, battle: Battle) -> int:
        return oranguru_memory.weather_turns_remaining(self, battle)

    def _terrain_turns_remaining(self, battle: Battle) -> int:
        return oranguru_memory.terrain_turns_remaining(self, battle)

    def _is_trapped(self, mon: Optional[Pokemon]) -> bool:
        if mon is None:
            return False
        effects = getattr(mon, "effects", None) or {}
        return Effect.TRAPPED in effects or Effect.PARTIALLY_TRAPPED in effects

    def _boosts_to_fp(self, boosts: Dict[str, int]) -> Dict[str, int]:
        result = {}
        if not boosts:
            return result
        mapping = {
            "atk": constants.ATTACK,
            "def": constants.DEFENSE,
            "spa": constants.SPECIAL_ATTACK,
            "spd": constants.SPECIAL_DEFENSE,
            "spe": constants.SPEED,
            "accuracy": "accuracy",
            "evasion": "evasion",
        }
        for key, val in boosts.items():
            fp_key = mapping.get(key)
            if fp_key:
                result[fp_key] = int(val)
        return result

    def _fill_moves_from_set(self, fp_mon: FPPokemon, set_info: dict, known_moves: set):
        if not set_info:
            return
        for move_id in set_info.get("moves", []):
            if not move_id:
                continue
            if move_id not in FP_MOVE_JSON:
                continue
            if fp_mon.get_move(move_id) is not None:
                continue
            if not known_moves or move_id in known_moves or len(known_moves) < 4:
                fp_mon.add_move(move_id)
        if len(fp_mon.moves) < 4:
            for move_id in set_info.get("moves", []):
                if move_id and fp_mon.get_move(move_id) is None:
                    if move_id not in FP_MOVE_JSON:
                        continue
                    fp_mon.add_move(move_id)
                if len(fp_mon.moves) >= 4:
                    break

    def _poke_env_to_fp(self, mon: Pokemon, battle: Battle, set_info: Optional[dict]) -> FPPokemon:
        species = normalize_name(getattr(mon, "species", "") or "")
        level = getattr(mon, "level", None) or (set_info.get("level") if set_info else 100)
        fp_mon = FPPokemon(species, level)

        ability = self._canon_id(getattr(mon, "ability", None)) if getattr(mon, "ability", None) else ""
        item = self._canon_id(getattr(mon, "item", None)) if getattr(mon, "item", None) else ""
        if set_info:
            if set_info.get("ability"):
                fp_mon.ability = set_info["ability"]
            if set_info.get("item"):
                fp_mon.item = set_info["item"]
            if set_info.get("tera"):
                fp_mon.tera_type = set_info["tera"]
        if ability:
            fp_mon.ability = ability
        if item:
            fp_mon.item = item
        tera_type = getattr(mon, "tera_type", None)
        if tera_type is not None:
            fp_mon.tera_type = normalize_name(tera_type.name if hasattr(tera_type, "name") else str(tera_type))
        terastallized = getattr(mon, "terastallized", None)
        if terastallized:
            fp_mon.terastallized = True
            if isinstance(terastallized, str):
                fp_mon.tera_type = normalize_name(terastallized)
        if getattr(mon, "is_terastallized", False):
            fp_mon.terastallized = True

        # Moves
        known_moves: List[str] = []
        for move in self._get_known_moves(mon):
            move_id = normalize_name(move.id)
            if move_id and move_id not in known_moves:
                known_moves.append(move_id)
                if move_id not in FP_MOVE_JSON:
                    continue
                fp_move = fp_mon.add_move(move_id)
                if fp_move is not None and hasattr(move, "current_pp"):
                    try:
                        fp_move.current_pp = int(move.current_pp)
                    except Exception:
                        pass
        if len(fp_mon.moves) < 4 and set_info:
            self._fill_moves_from_set(fp_mon, set_info, set(known_moves))

        # HP
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            fp_mon.max_hp = int(max_hp)
            if current_hp is not None:
                fp_mon.hp = int(current_hp)
            elif mon.current_hp_fraction is not None:
                fp_mon.hp = int(fp_mon.max_hp * mon.current_hp_fraction)
        elif mon.current_hp_fraction is not None and fp_mon.max_hp:
            fp_mon.hp = max(1, int(fp_mon.max_hp * mon.current_hp_fraction))

        # Stats
        stats = getattr(mon, "stats", None) or {}
        if stats:
            stat_map = {
                constants.ATTACK: stats.get("atk"),
                constants.DEFENSE: stats.get("def"),
                constants.SPECIAL_ATTACK: stats.get("spa"),
                constants.SPECIAL_DEFENSE: stats.get("spd"),
                constants.SPEED: stats.get("spe"),
            }
            for key, value in stat_map.items():
                if value is not None:
                    fp_mon.stats[key] = int(value)
            if stats.get("hp") is not None:
                fp_mon.max_hp = int(stats["hp"])
                if current_hp is not None:
                    fp_mon.hp = int(current_hp)
                elif mon.current_hp_fraction is not None:
                    fp_mon.hp = int(fp_mon.max_hp * mon.current_hp_fraction)

        # Types (honor temporary/tera types from poke_env)
        try:
            type_1 = getattr(mon, "type_1", None)
            type_2 = getattr(mon, "type_2", None)
            types = []
            if type_1 is not None:
                types.append(normalize_name(type_1.name if hasattr(type_1, "name") else str(type_1)))
            if type_2 is not None:
                types.append(normalize_name(type_2.name if hasattr(type_2, "name") else str(type_2)))
            if types:
                fp_mon.types = tuple(types)
        except Exception:
            pass

        if getattr(mon, "fainted", False):
            fp_mon.hp = 0
            fp_mon.fainted = True

        # Status
        fp_status = self._status_to_fp(getattr(mon, "status", None))
        if fp_status:
            fp_mon.status = fp_status
            if fp_status == constants.SLEEP:
                try:
                    fp_mon.sleep_turns = int(getattr(mon, "status_counter", 0))
                except Exception:
                    fp_mon.sleep_turns = 0

        # Boosts
        boosts = self._boosts_to_fp(getattr(mon, "boosts", {}) or {})
        for key, val in boosts.items():
            fp_mon.boosts[key] = val

        # Volatile statuses
        effects = getattr(mon, "effects", None) or {}
        for effect, count in effects.items():
            mapped = self.VOLATILE_EFFECT_MAP.get(effect)
            if not mapped:
                continue
            if mapped not in fp_mon.volatile_statuses:
                fp_mon.volatile_statuses.append(mapped)
            if count is not None:
                try:
                    fp_mon.volatile_status_durations[mapped] = int(count)
                except Exception:
                    pass
        # Poke-engine rejects taunt duration 3; clamp to a safe range.
        if constants.TAUNT in fp_mon.volatile_statuses:
            try:
                taunt_turns = int(fp_mon.volatile_status_durations[constants.TAUNT])
                fp_mon.volatile_status_durations[constants.TAUNT] = max(0, min(2, taunt_turns))
            except Exception:
                fp_mon.volatile_status_durations[constants.TAUNT] = 2

        # Tera
        if getattr(mon, "terastallized", False):
            fp_mon.terastallized = True
            if getattr(mon, "tera_type", None):
                fp_mon.tera_type = normalize_name(mon.tera_type.name)

        # Substitute state (track whether it took a hit)
        try:
            mem = self._get_battle_memory(battle)
            sub_state = mem.get("substitute_state", {})
            side_key = "self" if mon is battle.active_pokemon or mon in battle.team.values() else "opp"
            entry = sub_state.get(side_key, {}).get(normalize_name(fp_mon.name))
            if entry and Effect.SUBSTITUTE in (getattr(mon, "effects", None) or {}):
                fp_mon.substitute_hit = bool(entry.get("hit", False))
        except Exception:
            pass

        return fp_mon

    def _map_side_conditions(self, src: dict, dest: dict) -> None:
        if not src:
            return
        if SideCondition.SPIKES in src:
            dest[constants.SPIKES] = int(src[SideCondition.SPIKES])
        if SideCondition.STEALTH_ROCK in src:
            dest[constants.STEALTH_ROCK] = int(src[SideCondition.STEALTH_ROCK])
        if SideCondition.TOXIC_SPIKES in src:
            dest[constants.TOXIC_SPIKES] = int(src[SideCondition.TOXIC_SPIKES])
        if SideCondition.STICKY_WEB in src:
            dest[constants.STICKY_WEB] = int(src[SideCondition.STICKY_WEB])
        if SideCondition.REFLECT in src:
            dest[constants.REFLECT] = int(src[SideCondition.REFLECT])
        if SideCondition.LIGHT_SCREEN in src:
            dest[constants.LIGHT_SCREEN] = int(src[SideCondition.LIGHT_SCREEN])
        if SideCondition.AURORA_VEIL in src:
            dest[constants.AURORA_VEIL] = int(src[SideCondition.AURORA_VEIL])
        if SideCondition.TAILWIND in src:
            dest[constants.TAILWIND] = int(src[SideCondition.TAILWIND])
        if SideCondition.SAFEGUARD in src:
            dest[constants.SAFEGUARD] = int(src[SideCondition.SAFEGUARD])
        if SideCondition.MIST in src:
            dest[constants.MIST] = int(src[SideCondition.MIST])

    def _map_weather(self, battle: Battle) -> Optional[str]:
        weather = getattr(battle, "weather", None)
        if not weather:
            return None
        if isinstance(weather, dict):
            if not weather:
                return None
            weather = next(iter(weather.keys()))
        if isinstance(weather, Weather):
            name = normalize_name(weather.name)
        else:
            name = normalize_name(str(weather))
        mapping = {
            "raindance": constants.RAIN,
            "rain": constants.RAIN,
            "sunnyday": constants.SUN,
            "sun": constants.SUN,
            "sandstorm": constants.SAND,
            "hail": constants.HAIL,
            "snowscape": constants.SNOW,
        }
        return mapping.get(name, None)

    def _map_terrain(self, battle: Battle) -> Optional[str]:
        fields = getattr(battle, "fields", None) or {}
        for field in fields:
            if field == Field.ELECTRIC_TERRAIN:
                return constants.ELECTRIC_TERRAIN
            if field == Field.GRASSY_TERRAIN:
                return constants.GRASSY_TERRAIN
            if field == Field.MISTY_TERRAIN:
                return constants.MISTY_TERRAIN
            if field == Field.PSYCHIC_TERRAIN:
                return constants.PSYCHIC_TERRAIN
        return None

    def _clear_invalid_encore(self, battler: Battler) -> None:
        if battler.active is None:
            return
        if "encore" not in battler.active.volatile_statuses:
            return
        last_move = getattr(battler.last_used_move, "move", "") or ""
        if not last_move or last_move.startswith("switch"):
            battler.active.volatile_statuses = [
                v for v in battler.active.volatile_statuses if v != "encore"
            ]
            battler.active.volatile_status_durations["encore"] = 0

    def _sleep_clause_active(self, battle: Battle) -> bool:
        if not self.SLEEP_CLAUSE_ENABLED:
            return False
        fmt = normalize_name(getattr(battle, "format", "") or "")
        return "randombattle" in fmt

    def _opponent_has_sleeping_mon(self, battle: Battle) -> bool:
        for mon in battle.opponent_team.values():
            if mon is None:
                continue
            status = getattr(mon, "status", None)
            if status is None:
                continue
            status_id = getattr(status, "value", None) or normalize_name(str(status))
            if status_id in self.SLEEP_STATUS_IDS:
                return True
        opp_active = getattr(battle, "opponent_active_pokemon", None)
        if opp_active is not None:
            status = getattr(opp_active, "status", None)
            if status is not None:
                status_id = getattr(status, "value", None) or normalize_name(str(status))
                if status_id in self.SLEEP_STATUS_IDS:
                    return True
        return False

    def _sleep_clause_blocked(self, battle: Battle) -> bool:
        return self._sleep_clause_active(battle) and self._opponent_has_sleeping_mon(battle)

    def _move_inflicts_sleep(self, move) -> bool:
        if move is None:
            return False
        entry = self._get_move_entry(move)
        if entry.get("self", {}).get("status") == "slp":
            return False
        status = normalize_name(entry.get("status", ""))
        if status in self.SLEEP_STATUS_IDS:
            return True
        status_type = self.STATUS_MOVES.get(move.id)
        return status_type == "sleep"

    def _fp_move_inflicts_sleep(self, move_id: str) -> bool:
        entry = FP_MOVE_JSON.get(move_id, {})
        if entry.get("self", {}).get("status") == "slp":
            return False
        status = normalize_name(entry.get("status", ""))
        return status in self.SLEEP_STATUS_IDS

    def _sleep_clause_banned_choices(self, battle: Battle) -> set:
        if not self._sleep_clause_blocked(battle):
            return set()
        banned = set()
        for move in battle.available_moves or []:
            if self._move_inflicts_sleep(move):
                banned.add(move.id)
                banned.add(f"{move.id}-tera")
        return banned

    def _apply_opponent_item_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        flags = mem.get("opponent_item_flags", {}).get(normalize_name(fp_mon.name))
        if not flags:
            return
        if flags.get("no_choice"):
            fp_mon.can_have_choice_item = False
        if flags.get("no_assaultvest"):
            fp_mon.impossible_items.add("assaultvest")
        if flags.get("no_boots"):
            fp_mon.impossible_items.add("heavydutyboots")
        if flags.get("has_boots"):
            fp_mon.item = "heavydutyboots"
            fp_mon.item_inferred = True
        known_item = flags.get("known_item")
        if known_item:
            fp_mon.item = known_item
            fp_mon.item_inferred = True
        removed_item = flags.get("removed_item")
        if removed_item:
            fp_mon.removed_item = removed_item
            if fp_mon.item in {"", constants.UNKNOWN_ITEM}:
                fp_mon.knocked_off = True

    def _apply_opponent_ability_flags(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        ability = mem.get("opponent_abilities", {}).get(normalize_name(fp_mon.name))
        if ability:
            fp_mon.ability = ability
            if not fp_mon.original_ability:
                fp_mon.original_ability = ability
        impossible = mem.get("opponent_impossible_abilities", {}).get(normalize_name(fp_mon.name))
        if impossible:
            fp_mon.impossible_abilities.update(impossible)

    def _apply_known_opponent_moves(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        moves = mem.get("opponent_moves", {}).get(normalize_name(fp_mon.name), set())
        if not moves:
            return
        for move_id in moves:
            if len(fp_mon.moves) >= 4:
                break
            if move_id in FP_MOVE_JSON and fp_mon.get_move(move_id) is None:
                fp_mon.add_move(move_id)

    def _apply_opponent_switch_memory(self, fp_mon: FPPokemon, battle: Battle) -> None:
        mem = self._get_battle_memory(battle)
        species = normalize_name(fp_mon.name)
        info = mem.get("opp_switch_info", {}).get(species)
        if info:
            hp_at_switch = info.get("hp")
            if isinstance(hp_at_switch, (int, float)) and hp_at_switch > 0:
                try:
                    fp_mon.hp_at_switch_in = int(hp_at_switch)
                except Exception:
                    pass
            status_at_switch = info.get("status")
            if status_at_switch:
                fp_status = self._status_to_fp(status_at_switch)
                if fp_status:
                    fp_mon.status_at_switch_in = fp_status
        moves_since = mem.get("opp_moves_since_switch", {}).get(species, set())
        if moves_since:
            fp_mon.moves_used_since_switch_in = set(moves_since)
            if len(moves_since) >= 2:
                fp_mon.can_have_choice_item = False

    def _apply_speed_bounds(self, fp_mon: FPPokemon, battle: Battle) -> None:
        if not self.SPEED_BOUNDS_ENABLED:
            return
        mem = self._get_battle_memory(battle)
        bounds = mem.get("speed_bounds", {}).get(normalize_name(fp_mon.name))
        if not bounds:
            return
        min_speed = bounds.get("min", 0)
        max_speed = bounds.get("max", float("inf"))
        if min_speed <= 0 and max_speed == float("inf"):
            return
        if min_speed > max_speed:
            return
        fp_mon.speed_range = StatRange(min=min_speed, max=max_speed)

    def _damage_belief_has_unmodeled_state(self, battle: Battle) -> bool:
        return oranguru_belief.damage_belief_has_unmodeled_state(self, battle)

    def _damage_belief_observations(self, battle: Battle, species: str) -> List[dict]:
        return oranguru_belief.damage_belief_observations(self, battle, species)

    def _side_hazard_pressure(self, battle: Battle) -> float:
        side_conditions = getattr(battle, "side_conditions", None) or {}
        pressure = 0.0
        if SideCondition.STEALTH_ROCK in side_conditions:
            pressure += 0.125
        spikes_layers = int(side_conditions.get(SideCondition.SPIKES, 0) or 0)
        if spikes_layers == 1:
            pressure += 0.125
        elif spikes_layers == 2:
            pressure += 1.0 / 6.0
        elif spikes_layers >= 3:
            pressure += 0.25
        tspikes_layers = int(side_conditions.get(SideCondition.TOXIC_SPIKES, 0) or 0)
        if tspikes_layers > 0:
            pressure += 0.08
        if SideCondition.STICKY_WEB in side_conditions:
            pressure += 0.05
        return pressure

    def _opponent_progress_markers(self, battle: Battle, opponent: Optional[Pokemon]) -> dict:
        status = normalize_name(str(getattr(opponent, "status", "") or "")) if opponent is not None else ""
        opp_sc = getattr(battle, "opponent_side_conditions", None) or {}
        return {
            "status": status,
            "rocks": int(SideCondition.STEALTH_ROCK in opp_sc),
            "web": int(SideCondition.STICKY_WEB in opp_sc),
            "spikes": int(opp_sc.get(SideCondition.SPIKES, 0) or 0),
            "tspikes": int(opp_sc.get(SideCondition.TOXIC_SPIKES, 0) or 0),
        }

    def _resolve_passive_progress(self, battle: Battle) -> None:
        return oranguru_tactical.resolve_passive_progress(self, battle)

    def _passive_choice_kind(self, move) -> str:
        return oranguru_tactical.passive_choice_kind(self, move)

    def _progress_need_score(
        self,
        battle: Battle,
        active: Pokemon,
        opponent: Pokemon,
        best_damage_score: float,
    ) -> int:
        return oranguru_tactical.progress_need_score(self, battle, active, opponent, best_damage_score)

    # ------------------------------------------------------------------
    # Damage-belief: capture attacker context at action time
    # ------------------------------------------------------------------
    def _record_last_action(self, battle: Battle, order) -> None:
        return oranguru_memory.record_last_action(self, battle, order)

    # ------------------------------------------------------------------
    # Damage-belief: parse events to find observed damage
    # ------------------------------------------------------------------
    def _update_damage_observation(self, battle: Battle) -> None:
        return oranguru_memory.update_damage_observation(self, battle)

    @staticmethod
    def _parse_hp_fraction(hp_str: str) -> Optional[float]:
        return oranguru_memory.parse_hp_fraction(hp_str)

    def _ensure_randbats_sets(self, battle: Battle) -> str:
        return oranguru_belief.ensure_randbats_sets(self, battle)

    def _sanitize_randbats_moves(self) -> None:
        return oranguru_belief.sanitize_randbats_moves(self)

    def _belief_weight_for_set(
        self,
        fp_mon: FPPokemon,
        set_info: dict,
        battle: Battle,
        apply_damage: bool = True,
    ) -> float:
        return oranguru_belief.belief_weight_for_set(
            self, fp_mon, set_info, battle, apply_damage=apply_damage
        )

    def _candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, float]]:
        return oranguru_belief.candidate_randombattle_sets(self, opponent, battle)

    def _extract_last_opponent_move(self, battle: Battle) -> Optional[Tuple[str, str, int]]:
        return oranguru_worlds.extract_last_opponent_move(self, battle)

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

    @staticmethod
    def _fp_move_id(move) -> str:
        return oranguru_worlds.fp_move_id(None, move)

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

    def _switch_faints_to_entry_hazards(self, battle: Battle, mon: Pokemon) -> bool:
        return oranguru_tactical.switch_faints_to_entry_hazards(self, battle, mon)

    def _status_choice_is_obviously_bad(
        self,
        battle: Battle,
        move,
        active: Pokemon,
        opponent: Pokemon,
    ) -> bool:
        return oranguru_tactical.status_choice_is_obviously_bad(self, battle, move, active, opponent)

    def _apply_tactical_safety(self, battle: Battle, choice: str, active: Pokemon, opponent: Pokemon) -> str:
        return oranguru_tactical.apply_tactical_safety(self, battle, choice, active, opponent)

    def _heuristic_action_score(self, battle: Battle, choice: str) -> Optional[float]:
        return oranguru_decision.heuristic_action_score(self, battle, choice)

    def _adaptive_choice_risk_penalty(self, battle: Battle, choice: str) -> float:
        return oranguru_decision.adaptive_choice_risk_penalty(self, battle, choice)

    def _adaptive_rerank_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        topk: int,
    ) -> str:
        return oranguru_decision.adaptive_rerank_choice(self, battle, ordered, topk)

    def _aggregate_policy_from_results(
        self,
        results: List[Tuple[object, float]],
        banned_choices: Optional[set] = None,
    ) -> Tuple[List[Tuple[str, float]], float, float, float]:
        return oranguru_decision.aggregate_policy_from_results(self, results, banned_choices=banned_choices)

    def _collect_mcts_results(
        self,
        battle: Battle,
        sample_states: int,
        search_time_ms: int,
        base_fp_battle: Optional[FPBattle] = None,
    ) -> Tuple[List[Tuple[object, float]], List[dict]]:
        return oranguru_worlds.collect_mcts_results(
            self,
            battle,
            sample_states,
            search_time_ms,
            base_fp_battle=base_fp_battle,
        )

    def _select_move_from_results(
        self,
        results: List[Tuple[object, float]],
        battle: Battle,
        banned_choices: Optional[set] = None,
        world_candidates: Optional[List[dict]] = None,
    ) -> str:
        return oranguru_decision.select_move_from_results(
            self,
            results,
            battle,
            banned_choices=banned_choices,
            world_candidates=world_candidates,
        )

    def _is_damaging_move_choice(self, battle: Battle, choice: str) -> bool:
        return oranguru_decision.is_damaging_move_choice(self, battle, choice)

    def _choose_adaptive_fallback_order(
        self, battle: Battle, active: Pokemon, opponent: Pokemon
    ):
        return oranguru_decision.choose_adaptive_fallback_order(self, battle, active, opponent)

    def choose_move(self, battle: AbstractBattle):
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        self._flush_finished_battle_diags()
        self._flush_finished_search_traces()
        noop_order = self._empty_order_if_no_choices(battle)
        if noop_order is not None:
            return noop_order
        if getattr(battle, "_wait", False) or getattr(battle, "teampreview", False):
            return self.choose_random_move(battle)

        self._current_battle = battle
        self._update_immunity_memory(battle)
        self._update_active_turns(battle)
        self._update_battle_memory(battle)
        self._update_speed_order_memory(battle)
        self._update_switch_in_memory(battle)
        self._update_opponent_item_memory(battle)
        self._update_opponent_move_history(battle)
        self._update_opponent_ability_memory(battle)
        self._update_opponent_ability_constraints(battle)
        if self.IMMUNITY_INFER:
            self._infer_ability_from_immunity(battle)
        self._update_field_memory(battle)
        self._update_switch_flags(battle)
        self._update_substitute_memory(battle)
        self._update_damage_observation(battle)
        self._resolve_passive_progress(battle)
        self._cleanup_battle_memory(battle)
        mem = self._get_battle_memory(battle)
        last_action = mem.get("last_action")
        if last_action == "move" and mem.get("last_move_category") == "status":
            prev_hp = mem.get("last_opponent_hp")
            cur_hp = getattr(battle.opponent_active_pokemon, "current_hp_fraction", None)
            unchanged = (
                isinstance(prev_hp, (int, float))
                and isinstance(cur_hp, (int, float))
                and abs(cur_hp - prev_hp) <= 0.03
            )
            if unchanged:
                mem["status_stall_streak"] = int(mem.get("status_stall_streak", 0) or 0) + 1
            else:
                mem["status_stall_streak"] = 0
        else:
            mem["status_stall_streak"] = 0

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.force_switch:
            if self.DECISION_DIAG_ENABLED:
                self._mcts_stats["diag_forced_switch_turns"] += 1
                mem = self._get_battle_memory(battle)
                mem["diag_forced_switch_turns"] = int(mem.get("diag_forced_switch_turns", 0) or 0) + 1
            if battle.available_switches:
                best_switch = max(
                    battle.available_switches,
                    key=lambda s: self._score_switch(s, opponent, battle),
                )
                return self._commit_order(battle, self.create_order(best_switch))
            return self.choose_random_move(battle)

        self._mcts_stats["calls"] += 1
        sample_states = self.SAMPLE_STATES
        search_time_ms = self.SEARCH_TIME_MS
        if self.DYNAMIC_SAMPLING:
            opp_known_moves = len(self._get_known_moves(opponent))
            revealed = len([m for m in battle.opponent_team.values() if m is not None])
            opp_hp = opponent.current_hp_fraction or 0.0
            time_remaining = getattr(battle, "time_remaining", None)
            in_time_pressure = time_remaining is not None and time_remaining <= 60

            if revealed <= 3 and opp_hp > 0 and opp_known_moves == 0:
                multiplier = 2 if in_time_pressure else 4
                sample_states = max(sample_states, self.PARALLELISM * multiplier)
                search_time_ms = max(80, int(self.SEARCH_TIME_MS * 0.5))
            else:
                multiplier = 1 if in_time_pressure else 2
                sample_states = max(sample_states, self.PARALLELISM * multiplier)

            sample_states = min(self.MAX_SAMPLE_STATES, sample_states)
        sample_states = self._apply_world_budget_controls(battle, sample_states)

        results, world_candidates = self._collect_mcts_results(
            battle,
            sample_states=sample_states,
            search_time_ms=search_time_ms,
        )

        if not results:
            self._mcts_stats["empty_results"] += 1
            self._mcts_stats["fallback_super"] += 1
            if self.DECISION_DIAG_ENABLED:
                self._mcts_stats["diag_path_fallback_super"] += 1
            return super().choose_move(battle)

        banned_choices = self._sleep_clause_banned_choices(battle)
        ordered_pre, _total_pre, conf_pre, th_pre = self._aggregate_policy_from_results(
            results,
            banned_choices=banned_choices,
        )
        adaptive_escalate = False
        leaf_escalate = False
        if self.ADAPTIVE_ESCALATE_ENABLED and ordered_pre:
            adaptive_escalate = self._should_trigger_adaptive_fallback(
                battle, ordered_pre, conf_pre, th_pre, record_diag=False
            )
        if ordered_pre:
            leaf_escalate = self._should_trigger_leaf_value_escalation(
                battle,
                conf_pre,
                th_pre,
            )
        if adaptive_escalate or leaf_escalate:
            boosted_ms = search_time_ms
            boosted_states = sample_states
            if adaptive_escalate:
                boosted_ms = max(
                    boosted_ms,
                    int(search_time_ms * max(1.0, self.ADAPTIVE_ESCALATE_MS_MULT)),
                )
                boosted_ms = min(
                    boosted_ms,
                    max(search_time_ms, self.ADAPTIVE_ESCALATE_MAX_MS),
                )
                boosted_states = max(
                    boosted_states,
                    int(sample_states * max(1.0, self.ADAPTIVE_ESCALATE_SAMPLE_MULT)),
                )
                boosted_states = min(
                    boosted_states,
                    max(sample_states, self.ADAPTIVE_ESCALATE_MAX_STATES),
                )
            if leaf_escalate:
                boosted_ms = max(
                    boosted_ms,
                    int(search_time_ms * max(1.0, self.LEAF_VALUE_ESCALATE_MS_MULT)),
                )
                boosted_ms = min(
                    boosted_ms,
                    max(search_time_ms, self.LEAF_VALUE_ESCALATE_MAX_MS),
                )
                boosted_states = max(
                    boosted_states,
                    int(sample_states * max(1.0, self.LEAF_VALUE_ESCALATE_SAMPLE_MULT)),
                )
                boosted_states = min(
                    boosted_states,
                    max(sample_states, self.LEAF_VALUE_ESCALATE_MAX_STATES),
                )
            if boosted_ms > search_time_ms or boosted_states > sample_states:
                second_results, second_world_candidates = self._collect_mcts_results(
                    battle,
                    sample_states=boosted_states,
                    search_time_ms=boosted_ms,
                )
                if second_results:
                    results = second_results
                    world_candidates = second_world_candidates
                    if adaptive_escalate:
                        self._mcts_stats["adaptive_second_pass_used"] += 1
                    if leaf_escalate:
                        self._mcts_stats["leaf_value_escalated"] += 1
                else:
                    if adaptive_escalate:
                        self._mcts_stats["adaptive_second_pass_failed"] += 1
                    if leaf_escalate:
                        self._mcts_stats["leaf_value_escalate_failed"] += 1

        choice = self._select_move_from_results(
            results,
            battle,
            banned_choices=banned_choices,
            world_candidates=world_candidates,
        )
        if not choice:
            mem = self._get_battle_memory(battle)
            adaptive_pending = int(mem.get("adaptive_fallback_pending", 0) or 0) == 1
            if adaptive_pending:
                mem["adaptive_fallback_pending"] = 0
                if self.ADAPTIVE_FALLBACK_MODE == "heuristic":
                    adaptive_order = self._choose_adaptive_fallback_order(battle, active, opponent)
                    if adaptive_order is not None:
                        self._mcts_stats["adaptive_heuristic_used"] += 1
                        if self.DECISION_DIAG_ENABLED:
                            self._mcts_stats["diag_path_fallback_super"] += 1
                        return self._commit_order(battle, adaptive_order)
                    self._mcts_stats["adaptive_heuristic_failed"] += 1
                self._mcts_stats["adaptive_super_used"] += 1
            self._mcts_stats["fallback_super"] += 1
            if self.DECISION_DIAG_ENABLED:
                self._mcts_stats["diag_path_fallback_super"] += 1
            return super().choose_move(battle)
        choice = self._apply_tactical_safety(battle, choice, active, opponent)

        if choice.startswith("switch "):
            if self._is_switch_churn_risk(battle) and battle.available_moves:
                emergency_order = self._choose_emergency_non_switch_order(
                    battle,
                    active,
                    opponent,
                    len([m for m in battle.team.values() if not m.fainted]),
                )
                if emergency_order is not None:
                    mem = self._get_battle_memory(battle)
                    mem["switch_churn_breaks"] = int(mem.get("switch_churn_breaks", 0) or 0) + 1
                    return self._commit_order(battle, emergency_order)
            switch_name = normalize_name(choice.split("switch ", 1)[1])
            for sw in battle.available_switches:
                if normalize_name(sw.species) == switch_name:
                    return self._commit_order(battle, self.create_order(sw))
            self._mcts_stats["fallback_random"] += 1
            if self.DECISION_DIAG_ENABLED:
                self._mcts_stats["diag_path_fallback_random"] += 1
            return self.choose_random_move(battle)

        tera = False
        if choice.endswith("-tera"):
            choice = choice.replace("-tera", "")
            tera = bool(getattr(battle, "can_tera", False))
        move_id = normalize_name(choice)
        for move in battle.available_moves:
            if move.id == move_id:
                return self._commit_order(
                    battle,
                    self.create_order(
                        move,
                        terastallize=(
                            tera
                            or (
                                self.AUTO_TERA
                                and getattr(battle, "can_tera", False)
                                and self._should_terastallize(battle, move)
                            )
                        ),
                        dynamax=self._should_dynamax(battle, len([m for m in battle.team.values() if not m.fainted])),
                    ),
                )
        self._mcts_stats["fallback_random"] += 1
        if self.DECISION_DIAG_ENABLED:
            self._mcts_stats["diag_path_fallback_random"] += 1
        return self.choose_random_move(battle)
