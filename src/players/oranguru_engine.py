#!/usr/bin/env python3
"""
OranguruEnginePlayer - MCTS via poke-engine using a lightweight state builder.

Builds a poke-engine State from poke_env battle + randombattle set sampling
and runs MCTS to choose moves, similar in spirit to foul-play.
"""

from __future__ import annotations

import math
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
        # Screen effects (opponent side) are not modeled by our lightweight
        # damage estimate; skip these turns to avoid poisoning observations.
        opp_sc = set((battle.opponent_side_conditions or {}).keys())
        screen_conds = {SideCondition.REFLECT, SideCondition.LIGHT_SCREEN}
        aurora = getattr(SideCondition, "AURORA_VEIL", None)
        if aurora is not None:
            screen_conds.add(aurora)
        if opp_sc.intersection(screen_conds):
            return True

        # Model currently covers only sun/rain weather modifiers.
        for w in battle.weather:
            wn = normalize_name(w.name if hasattr(w, "name") else str(w))
            if "sun" in wn or "sunnyday" in wn or "rain" in wn or "raindance" in wn:
                continue
            if wn:
                return True
        return False

    def _damage_belief_observations(self, battle: Battle, species: str) -> List[dict]:
        mem = self._get_battle_memory(battle)
        obs = list(mem.get("damage_observations", {}).get(species, []) or [])
        if not obs:
            return []
        if self.DAMAGE_BELIEF_STRICT_ONLY:
            obs = [entry for entry in obs if entry.get("high_confidence", False)]
        return obs

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
        gen_num = getattr(battle, "gen", None) or 9
        gen_name = f"gen{gen_num}"
        if not self._randbats_initialized or self._randbats_gen != gen_name:
            RandomBattleTeamDatasets.initialize(gen_name)
            self._randbats_initialized = True
            self._randbats_gen = gen_name
            self._randbats_sanitized = False
        if not self._randbats_sanitized:
            self._sanitize_randbats_moves()
        return gen_name

    def _sanitize_randbats_moves(self) -> None:
        invalid = set()
        for pkmn_sets in RandomBattleTeamDatasets.pkmn_sets.values():
            for predicted in pkmn_sets:
                moves = list(predicted.pkmn_moveset.moves)
                filtered = [m for m in moves if m in FP_MOVE_JSON]
                if not filtered:
                    invalid.update(m for m in moves if m not in FP_MOVE_JSON)
                    continue
                if len(filtered) != len(moves):
                    invalid.update(m for m in moves if m not in FP_MOVE_JSON)
                    predicted.pkmn_moveset.moves = tuple(filtered)
        self._randbats_sanitized = True
        if invalid:
            pass

    def _belief_weight_for_set(
        self,
        fp_mon: FPPokemon,
        set_info: dict,
        battle: Battle,
        apply_damage: bool = True,
    ) -> float:
        species = normalize_name(fp_mon.name)
        mem = self._get_battle_memory(battle)
        ability = normalize_name(set_info.get("ability", "") or "")

        # --- Immunity-based scoring (existing) ---
        multiplier = 1.0
        immune_types = mem.get("immune_types", {}).get(species, set())
        if immune_types and ability:
            types = set(fp_mon.types or [])
            for immune_type in immune_types:
                if immune_type in types:
                    continue
                candidates = self.IMMUNITY_ABILITY_MAP.get(immune_type, set())
                if not candidates:
                    continue
                if ability in candidates:
                    multiplier *= self.BELIEF_IMMUNITY_MATCH
                else:
                    multiplier *= self.BELIEF_IMMUNITY_MISS

        # --- Damage-based scoring ---
        if self.DAMAGE_BELIEF and apply_damage:
            dmg_obs = self._damage_belief_observations(battle, species)
            if dmg_obs:
                set_item = normalize_name(set_info.get("item", "") or "")
                base_stats = {}
                if fp_mon.base_stats:
                    raw = fp_mon.base_stats
                    # FP pokedex uses long-form keys; normalize to short form
                    base_stats = {
                        "hp": raw.get("hp", 80),
                        "atk": raw.get("attack", raw.get("atk", 80)),
                        "def": raw.get("defense", raw.get("def", 80)),
                        "spa": raw.get("special-attack", raw.get("spa", 80)),
                        "spd": raw.get("special-defense", raw.get("spd", 80)),
                        "spe": raw.get("speed", raw.get("spe", 80)),
                    }
                if not base_stats:
                    base_stats = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
                level = int(set_info.get("level", 0) or 0) or int(fp_mon.level or 100)
                sp_types = list(fp_mon.types or [])
                dmg_weight = score_set_damage_consistency(
                    observations=dmg_obs,
                    set_ability=ability,
                    set_item=set_item,
                    species_base_stats=base_stats,
                    species_level=level,
                    species_types=sp_types,
                    mode=self.DAMAGE_BELIEF_MODE,
                    per_obs_min=self.DAMAGE_BELIEF_PER_OBS_MIN,
                    per_obs_max=self.DAMAGE_BELIEF_PER_OBS_MAX,
                    final_min=self.DAMAGE_BELIEF_FINAL_MIN,
                    final_max=self.DAMAGE_BELIEF_FINAL_MAX,
                )
                multiplier *= dmg_weight

        return multiplier

    def _candidate_randombattle_sets(self, opponent: Pokemon, battle: Battle) -> List[Tuple[dict, float]]:
        if not self.BELIEF_SAMPLING:
            return super()._candidate_randombattle_sets(opponent, battle)
        if opponent is None:
            return []
        try:
            self._ensure_randbats_sets(battle)
        except Exception:
            return super()._candidate_randombattle_sets(opponent, battle)

        fp_mon = self._poke_env_to_fp(opponent, battle, None)
        self._apply_opponent_item_flags(fp_mon, battle)
        self._apply_opponent_ability_flags(fp_mon, battle)
        self._apply_known_opponent_moves(fp_mon, battle)
        self._apply_speed_bounds(fp_mon, battle)

        predicted_sets = RandomBattleTeamDatasets.get_all_remaining_sets(fp_mon)
        if not predicted_sets:
            return super()._candidate_randombattle_sets(opponent, battle)

        candidates: List[Tuple[dict, float]] = []
        for predicted in predicted_sets:
            pset = predicted.pkmn_set
            moves = [m for m in predicted.pkmn_moveset.moves if m in FP_MOVE_JSON]
            if not moves:
                continue
            set_info = {
                "level": int(pset.level or fp_mon.level or 100),
                "item": normalize_name(pset.item or ""),
                "ability": normalize_name(pset.ability or ""),
                "moves": [normalize_name(m) for m in moves],
                "tera": normalize_name(pset.tera_type or ""),
            }
            weight = float(pset.count or 0)
            if weight <= 0:
                continue
            weight = weight * self._belief_weight_for_set(
                fp_mon,
                set_info,
                battle,
                apply_damage=False,
            )
            if weight <= 0:
                continue
            candidates.append((set_info, weight))

        if not candidates:
            return super()._candidate_randombattle_sets(opponent, battle)

        candidates.sort(key=lambda x: x[1], reverse=True)
        if self.DAMAGE_BELIEF:
            species = normalize_name(fp_mon.name)
            dmg_obs = self._damage_belief_observations(battle, species)
            if len(dmg_obs) >= self.DAMAGE_BELIEF_MIN_OBS:
                topk = max(1, min(self.DAMAGE_BELIEF_TOPK, len(candidates)))
                base_stats = {}
                if fp_mon.base_stats:
                    raw = fp_mon.base_stats
                    base_stats = {
                        "hp": raw.get("hp", 80),
                        "atk": raw.get("attack", raw.get("atk", 80)),
                        "def": raw.get("defense", raw.get("def", 80)),
                        "spa": raw.get("special-attack", raw.get("spa", 80)),
                        "spd": raw.get("special-defense", raw.get("spd", 80)),
                        "spe": raw.get("speed", raw.get("spe", 80)),
                    }
                if not base_stats:
                    base_stats = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
                sp_types = list(fp_mon.types or [])

                reranked = list(candidates)
                for idx in range(topk):
                    set_info, base_weight = reranked[idx]
                    set_ability = normalize_name(set_info.get("ability", "") or "")
                    set_item = normalize_name(set_info.get("item", "") or "")
                    level = int(set_info.get("level", 0) or 0) or int(fp_mon.level or 100)
                    dmg_weight = score_set_damage_consistency(
                        observations=dmg_obs,
                        set_ability=set_ability,
                        set_item=set_item,
                        species_base_stats=base_stats,
                        species_level=level,
                        species_types=sp_types,
                        mode=self.DAMAGE_BELIEF_MODE,
                        per_obs_min=self.DAMAGE_BELIEF_PER_OBS_MIN,
                        per_obs_max=self.DAMAGE_BELIEF_PER_OBS_MAX,
                        final_min=self.DAMAGE_BELIEF_FINAL_MIN,
                        final_max=self.DAMAGE_BELIEF_FINAL_MAX,
                    )
                    reranked[idx] = (set_info, base_weight * dmg_weight)
                candidates = reranked
                candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:20]

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
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return None
        mem = self._get_battle_memory(battle)

        if choice.startswith("switch "):
            target = normalize_name(choice.split("switch ", 1)[1])
            for sw in battle.available_switches:
                if normalize_name(sw.species) == target:
                    score = max(0.0, self._score_switch(sw, opponent, battle))
                    switch_streak = int(mem.get("self_switch_streak", 0) or 0)
                    if switch_streak >= 2:
                        score -= 20.0 * min(4, switch_streak - 1)
                    return max(0.0, score)
            return None

        move_id = normalize_name(choice.replace("-tera", ""))
        for move in battle.available_moves:
            if move.id != move_id:
                continue
            if move.id in self.PROTECT_MOVES:
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                if self._should_use_protect(battle, reply_score):
                    hp_frac = active.current_hp_fraction or 0.0
                    threshold = max(1.0, 220.0 * max(hp_frac, 0.1))
                    danger = min(1.0, reply_score / threshold)
                    stall = int(mem.get("status_stall_streak", 0) or 0)
                    penalty = 18.0 * min(3, stall)
                    return max(0.0, 120.0 * danger - penalty)
                return 0.0
            if self._is_recovery_move(move):
                reply_score = self._estimate_best_reply_score(opponent, active, battle)
                safe_recover = reply_score < 150
                if self._estimate_matchup(active, opponent) > 0.35 and (active.current_hp_fraction or 0.0) < 0.4:
                    safe_recover = True
                if getattr(self, "RECOVERY_KO_GUARD", False):
                    best_damage = self._estimate_best_damage_score(active, opponent, battle)
                    opp_hp = opponent.current_hp_fraction or 0.0
                    threshold = getattr(self, "RECOVERY_KO_THRESHOLD", 220.0) * max(opp_hp, 0.05)
                    if best_damage >= threshold:
                        return 0.0
                hp_frac = active.current_hp_fraction or 0.0
                if hp_frac < 0.65 and safe_recover:
                    missing = 1.0 - hp_frac
                    stall = int(mem.get("status_stall_streak", 0) or 0)
                    penalty = 20.0 * min(3, stall)
                    return max(0.0, 140.0 * missing - penalty)
                return 0.0
            if self._sleep_clause_blocked(battle) and self._move_inflicts_sleep(move):
                return 0.0
            if move.category == MoveCategory.STATUS:
                if self.STATUS_KO_GUARD:
                    opp_hp = opponent.current_hp_fraction or 0.0
                    best_damage = self._estimate_best_damage_score(active, opponent, battle)
                    threshold = self.STATUS_KO_THRESHOLD * max(opp_hp, 0.05)
                    if best_damage >= threshold:
                        return 0.0
                boosts = getattr(move, "boosts", None) or {}
                if boosts and move.target and "self" in str(move.target).lower():
                    if self._should_setup_move(move, active, opponent):
                        matchup = self._estimate_matchup(active, opponent)
                        hp_frac = active.current_hp_fraction or 0.0
                        base = 80.0 + 40.0 * max(0.0, min(1.0, matchup + 0.5))
                        return max(0.0, base * max(0.0, min(1.0, hp_frac + 0.1)))
                score = self._should_use_status_move(move, active, opponent, battle)
                status_type = self.STATUS_MOVES.get(move.id)
                if status_type is None:
                    status_type = self._status_from_move_entry(self._get_move_entry(move))
                if (
                    self.STALL_SHUTDOWN_BOOST
                    and (status_type in {"taunt", "encore"} or move.id in self.ANTI_SETUP_MOVES)
                    and (
                        self._opponent_is_stallish(opponent)
                        or (opponent.boosts and sum(opponent.boosts.values()) > 0)
                    )
                ):
                    score *= 1.25
                stall = int(mem.get("status_stall_streak", 0) or 0)
                if stall >= 1 and status_type not in {"taunt", "encore"}:
                    score -= 20.0 * min(3, stall)
                return max(0.0, score)
            predicted_switch = self._predict_opponent_switch(battle)
            return max(
                0.0,
                self._score_move_with_prediction(
                    move, active, opponent, predicted_switch, battle
                ),
            )
        return None

    def _adaptive_choice_risk_penalty(self, battle: Battle, choice: str) -> float:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return 0.0

        mem = self._get_battle_memory(battle)
        risk = 0.0

        if choice.startswith("switch "):
            target = normalize_name(choice.split("switch ", 1)[1])
            selected = None
            for sw in battle.available_switches or []:
                if normalize_name(sw.species) == target:
                    selected = sw
                    break
            if selected is None:
                return 80.0
            if self._switch_faints_to_entry_hazards(battle, selected):
                risk += 150.0
            cur_match = self._estimate_matchup(active, opponent)
            new_match = self._estimate_matchup(selected, opponent)
            if new_match < cur_match - 0.15:
                risk += 45.0
            switch_streak = int(mem.get("self_switch_streak", 0) or 0)
            if switch_streak >= 2:
                risk += 25.0 * min(4, switch_streak - 1)
            return risk

        move_id = normalize_name(choice.replace("-tera", ""))
        selected_move = None
        for move in battle.available_moves or []:
            if move.id == move_id:
                selected_move = move
                break
        if selected_move is None:
            return 80.0

        best_damage_move, best_damage_score = self._best_damaging_move(battle, active, opponent)
        opp_hp = opponent.current_hp_fraction or 0.0
        ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
        has_ko_window = best_damage_move is not None and best_damage_score >= ko_threshold

        try:
            is_status = selected_move.category == MoveCategory.STATUS
        except Exception:
            is_status = False
        is_recovery = self._is_recovery_move(selected_move)
        is_setup = bool(getattr(selected_move, "boosts", None) or {}) and (
            selected_move.target and "self" in str(selected_move.target).lower()
        )

        if is_status and self._status_choice_is_obviously_bad(battle, selected_move, active, opponent):
            risk += 90.0
        if has_ko_window and (is_status or is_recovery or is_setup or selected_move.id in self.PROTECT_MOVES):
            risk += 120.0
        if is_recovery and (active.current_hp_fraction or 0.0) > 0.75:
            risk += 35.0
        if selected_move.id in self.PROTECT_MOVES:
            reply_score = self._estimate_best_reply_score(opponent, active, battle)
            if not self._should_use_protect(battle, reply_score):
                risk += 70.0

        stall = int(mem.get("status_stall_streak", 0) or 0)
        if stall >= 1 and (is_status or is_recovery or selected_move.id in self.PROTECT_MOVES):
            risk += 25.0 * min(3, stall)
        return risk

    def _adaptive_rerank_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        topk: int,
    ) -> str:
        if not ordered:
            return ""
        candidates = ordered[: max(1, topk)]
        total = sum(max(0.0, w) for _, w in candidates)
        if total <= 0:
            return candidates[0][0]

        heur_weight = max(0.0, self.ADAPTIVE_RERANK_HEUR_WEIGHT)
        risk_weight = max(0.0, self.ADAPTIVE_RERANK_RISK_WEIGHT)
        max_policy_drop = max(0.0, min(1.0, self.ADAPTIVE_RERANK_MAX_POLICY_DROP))
        min_score_gain = max(0.0, self.ADAPTIVE_RERANK_MIN_SCORE_GAIN)
        min_risk_delta = max(0.0, self.ADAPTIVE_RERANK_MIN_RISK_DELTA)

        scored: List[dict] = []
        for choice, weight in candidates:
            mcts_term = max(0.0, weight) / total
            heur = max(0.0, float(self._heuristic_action_score(battle, choice) or 0.0))
            risk = max(0.0, float(self._adaptive_choice_risk_penalty(battle, choice)))
            score = mcts_term + heur_weight * (heur / 100.0) - risk_weight * (risk / 100.0)
            scored.append(
                {
                    "choice": choice,
                    "weight": max(0.0, float(weight)),
                    "risk": risk,
                    "score": score,
                }
            )
        if not scored:
            return candidates[0][0]

        baseline = scored[0]
        winner = max(scored, key=lambda row: row["score"])
        if winner["choice"] == baseline["choice"]:
            return baseline["choice"]

        policy_drop = (baseline["weight"] - winner["weight"]) / total
        if policy_drop > max_policy_drop:
            return baseline["choice"]

        score_gain = float(winner["score"]) - float(baseline["score"])
        if score_gain < min_score_gain:
            return baseline["choice"]

        risk_delta = float(baseline["risk"]) - float(winner["risk"])
        if risk_delta < min_risk_delta:
            return baseline["choice"]

        return winner["choice"]

    def _aggregate_policy_from_results(
        self,
        results: List[Tuple[object, float]],
        banned_choices: Optional[set] = None,
    ) -> Tuple[List[Tuple[str, float]], float, float, float]:
        final_policy: Dict[str, float] = {}
        for res, weight in results:
            total_visits = res.total_visits or 1
            for opt in res.side_one:
                final_policy[opt.move_choice] = final_policy.get(opt.move_choice, 0.0) + (
                    weight * (opt.visits / total_visits)
                )
        if banned_choices:
            filtered_policy = {k: v for k, v in final_policy.items() if k not in banned_choices}
            if filtered_policy:
                final_policy = filtered_policy
        ordered = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
        total_policy = sum(w for _, w in ordered)
        confidence = 0.0
        if ordered and total_policy > 0:
            best = ordered[0][1]
            confidence = best / total_policy
            if len(ordered) > 1:
                second = ordered[1][1]
                margin = (best - second) / total_policy
                confidence = max(confidence, margin)
        threshold = max(0.0, min(1.0, self.MCTS_CONFIDENCE_THRESHOLD))
        return ordered, total_policy, confidence, threshold

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
        battle_tag = str(getattr(battle, "battle_tag", "")).lower()
        is_eval_tag = any(key in battle_tag for key in ("eval", "evaluation", "heuristic", "oranguru"))
        env_eval_mode = os.getenv("ORANGURU_EVAL_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        deterministic = self.MCTS_DETERMINISTIC and (
            not self.MCTS_DETERMINISTIC_EVAL_ONLY or is_eval_tag or env_eval_mode
        )
        if deterministic:
            self._mcts_stats["deterministic_decisions"] += 1
        else:
            self._mcts_stats["stochastic_decisions"] += 1
        ordered, total_policy, confidence, threshold = self._aggregate_policy_from_results(
            results,
            banned_choices=banned_choices,
        )
        if not ordered:
            return ""
        def _return_choice(chosen_choice: str, path: str) -> str:
            if chosen_choice:
                self._diag_record_choice(
                    battle,
                    ordered,
                    chosen_choice,
                    confidence,
                    threshold,
                    path,
                )
                self._append_search_trace_example(
                    battle,
                    ordered,
                    chosen_choice,
                    confidence,
                    threshold,
                    path,
                    world_candidates=world_candidates,
                )
            return chosen_choice
        if total_policy <= 0:
            return _return_choice(ordered[0][0], "mcts")

        def _pick_choice(choices: List[str], weights: List[float]) -> str:
            if not choices:
                return ""
            total = sum(weights) if weights else 0.0
            if deterministic:
                if total <= 0:
                    return choices[0]
                best_idx = max(range(len(choices)), key=lambda i: weights[i])
                return choices[best_idx]
            if total <= 0:
                return random.choice(choices)
            return random.choices(choices, weights=weights, k=1)[0]

        best = ordered[0][1]

        cutoff = best * 0.75
        filtered = [o for o in ordered if o[1] >= cutoff]
        filtered = self._apply_switch_prior_prune(battle, filtered, confidence, threshold)
        filtered = self._apply_tera_prune(battle, filtered, confidence, threshold)
        if not filtered:
            filtered = [ordered[0]]
        passive_break_choice = self._maybe_passive_break_choice(battle, filtered, confidence, threshold)
        if passive_break_choice:
            return _return_choice(passive_break_choice, "rerank")

        if self._should_trigger_adaptive_fallback(battle, ordered, confidence, threshold):
            mem = self._get_battle_memory(battle)
            mem["adaptive_fallback_last_turn"] = int(getattr(battle, "turn", 0) or 0)
            self._mcts_stats["adaptive_triggered"] += 1
            mode = (self.ADAPTIVE_FALLBACK_MODE or "super").strip().lower()
            if mode in {"rerank", "risk", "adaptive"}:
                reranked = self._adaptive_rerank_choice(
                    battle,
                    ordered,
                    topk=max(1, self.ADAPTIVE_FALLBACK_TOPK),
                )
                if reranked:
                    self._mcts_stats["adaptive_rerank_used"] += 1
                    if self.DECISION_DIAG_ENABLED:
                        mem["diag_adaptive_triggered"] = int(mem.get("diag_adaptive_triggered", 0) or 0) + 1
                    return _return_choice(reranked, "adaptive_rerank")
                self._mcts_stats["adaptive_rerank_failed"] += 1
                return _return_choice(ordered[0][0], "mcts")
            mem["adaptive_fallback_pending"] = 1
            if self.DECISION_DIAG_ENABLED:
                mem["diag_adaptive_triggered"] = int(mem.get("diag_adaptive_triggered", 0) or 0) + 1
            return ""
        if self.SELECTION_MODE == "policy":
            cutoff_ratio = max(0.0, min(1.0, self.POLICY_CUTOFF))
            cutoff = best * cutoff_ratio
            policy_choices = [(choice, weight) for choice, weight in ordered if weight >= cutoff]
            if not policy_choices:
                policy_choices = [ordered[0]]
            choices = [choice for choice, _ in policy_choices]
            policy_weights = [max(0.0, weight) for _, weight in policy_choices]
            if confidence < threshold:
                heuristic_weights = []
                for choice in choices:
                    score = self._heuristic_action_score(battle, choice)
                    heuristic_weights.append(max(0.0, score or 0.0))
                heur_total = sum(heuristic_weights)
                if heur_total > 0:
                    return _return_choice(_pick_choice(choices, heuristic_weights), "policy")
            return _return_choice(_pick_choice(choices, policy_weights), "policy")

        if self.SELECTION_MODE == "rerank" and confidence < threshold:
            candidates = filtered[: max(1, self.RERANK_TOPK)]
            scored = []
            for choice, weight in candidates:
                score = self._heuristic_action_score(battle, choice)
                if score is None:
                    continue
                scored.append((score, weight, choice))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return _return_choice(scored[0][2], "rerank")

        blend = max(0.0, min(1.0, self.HEURISTIC_BLEND))
        min_blend = max(0.0, min(1.0, self.MIN_HEURISTIC_BLEND))
        if self.GATE_MODE == "entropy":
            probs = [w / total_policy for _, w in ordered]
            if len(probs) <= 1:
                normalized_entropy = 0.0
            else:
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                normalized_entropy = entropy / math.log(len(probs))
            if threshold > 0:
                gate = min(1.0, normalized_entropy / threshold)
            else:
                gate = normalized_entropy
            blend *= gate
        elif self.GATE_MODE == "soft":
            if threshold > 0:
                gate = max(0.0, min(1.0, (threshold - confidence) / threshold))
            else:
                gate = 1.0
            blend *= gate
        else:
            confidence = best / total_policy if total_policy > 0 else 0.0
            if len(ordered) > 1:
                second = ordered[1][1]
                margin = (best - second) / total_policy if total_policy > 0 else 0.0
                confidence = max(confidence, margin)
            if confidence >= threshold:
                blend = 0.0
        if min_blend > 0.0:
            blend = max(blend, min_blend)

        choices = []
        mcts_weights = []
        for choice, weight in filtered:
            choices.append(choice)
            mcts_weights.append(max(0.0, weight))

        if not choices:
            return ordered[0][0]

        mcts_total = sum(mcts_weights)
        if mcts_total <= 0:
            mcts_norm = [1.0 / len(choices)] * len(choices)
        else:
            mcts_norm = [w / mcts_total for w in mcts_weights]

        selection_path = "blend"
        applied_heuristic_blend = False
        applied_prior_blend = False
        if blend <= 0:
            combined = mcts_norm
            selection_path = "mcts"
        else:
            heuristic_weights = []
            for choice in choices:
                score = self._heuristic_action_score(battle, choice)
                heuristic_weights.append(max(0.0, score or 0.0))
            heur_total = sum(heuristic_weights)
            if heur_total <= 0:
                combined = mcts_norm
                selection_path = "mcts"
            else:
                heur_norm = [w / heur_total for w in heuristic_weights]
                combined = [
                    (1.0 - blend) * m + blend * h for m, h in zip(mcts_norm, heur_norm)
                ]
                applied_heuristic_blend = True

        rl_blend = max(0.0, min(1.0, self.RL_PRIOR_BLEND))
        if rl_blend > 0.0 and (not self.RL_PRIOR_LOWCONF_ONLY or confidence < threshold):
            rl_priors = self._rl_choice_priors(battle, choices)
            if rl_priors:
                rl_total = sum(rl_priors)
                if rl_total > 0:
                    rl_norm = [w / rl_total for w in rl_priors]
                    combined = [
                        (1.0 - rl_blend) * base + rl_blend * rl
                        for base, rl in zip(combined, rl_norm)
                    ]
                    applied_prior_blend = True

        search_prior_blend = max(0.0, min(1.0, self.SEARCH_PRIOR_BLEND))
        if search_prior_blend > 0.0 and (not self.SEARCH_PRIOR_LOWCONF_ONLY or confidence < threshold):
            search_priors = self._search_choice_priors(battle, choices)
            if search_priors:
                prior_total = sum(search_priors)
                if prior_total > 0:
                    prior_norm = [w / prior_total for w in search_priors]
                    combined = [
                        (1.0 - search_prior_blend) * base + search_prior_blend * prior
                        for base, prior in zip(combined, prior_norm)
                    ]
                    applied_prior_blend = True

        if applied_prior_blend and not applied_heuristic_blend:
            selection_path = "policy"
        elif applied_prior_blend or applied_heuristic_blend:
            selection_path = "blend"
        else:
            selection_path = "mcts"

        return _return_choice(_pick_choice(choices, combined), selection_path)

    def _is_damaging_move_choice(self, battle: Battle, choice: str) -> bool:
        move_id = normalize_name(choice.replace("-tera", ""))
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) != move_id:
                continue
            category = getattr(move, "category", None)
            base_power = float(getattr(move, "base_power", 0) or 0)
            damage_attr = getattr(move, "damage", None)
            return bool(
                base_power > 0
                or damage_attr is not None
                or (category is not None and category != MoveCategory.STATUS)
            )
        return False

    def _choose_adaptive_fallback_order(
        self, battle: Battle, active: Pokemon, opponent: Pokemon
    ):
        if battle.force_switch:
            if battle.available_switches:
                best_switch = max(
                    battle.available_switches,
                    key=lambda s: self._score_switch(s, opponent, battle),
                )
                return self.create_order(best_switch)
            return None

        move_candidates: List[Tuple[str, float]] = []
        for move in battle.available_moves or []:
            choice = normalize_name(getattr(move, "id", ""))
            if not choice:
                continue
            score = self._heuristic_action_score(battle, choice)
            move_candidates.append((choice, float(score or 0.0)))
            if getattr(battle, "can_tera", False):
                tera_choice = f"{choice}-tera"
                tera_score = self._heuristic_action_score(battle, tera_choice)
                move_candidates.append((tera_choice, float(tera_score or 0.0)))

        switch_candidates: List[Tuple[str, float]] = []
        for sw in battle.available_switches or []:
            sw_choice = f"switch {normalize_name(sw.species)}"
            score = self._heuristic_action_score(battle, sw_choice)
            switch_candidates.append((sw_choice, float(score or 0.0)))

        if not move_candidates and not switch_candidates:
            return None

        selected: Optional[str] = None
        if move_candidates:
            move_candidates.sort(key=lambda x: x[1], reverse=True)
            best_move, best_score = move_candidates[0]
            if best_score > 0:
                damaging = [
                    (c, s)
                    for c, s in move_candidates
                    if self._is_damaging_move_choice(battle, c) and s >= best_score * 0.85
                ]
                if damaging:
                    damaging.sort(key=lambda x: x[1], reverse=True)
                    selected = damaging[0][0]
                else:
                    selected = best_move

        if selected is None and switch_candidates:
            switch_candidates.sort(key=lambda x: x[1], reverse=True)
            selected = switch_candidates[0][0]

        if not selected:
            return None
        if selected.startswith("switch "):
            switch_name = normalize_name(selected.split("switch ", 1)[1])
            for sw in battle.available_switches or []:
                if normalize_name(sw.species) == switch_name:
                    return self.create_order(sw)
            return None

        tera = False
        if selected.endswith("-tera"):
            selected = selected.replace("-tera", "")
            tera = bool(getattr(battle, "can_tera", False))
        move_id = normalize_name(selected)
        for move in battle.available_moves or []:
            if normalize_name(getattr(move, "id", "")) == move_id:
                return self.create_order(move, terastallize=tera)
        return None

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
