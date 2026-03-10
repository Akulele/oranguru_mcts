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
from src.utils.damage_calc import normalize_name, get_type_effectiveness
from src.utils.damage_belief import score_set_damage_consistency

FP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

from fp.battle import Battle as FPBattle, Battler, Pokemon as FPPokemon, LastUsedMove, StatRange  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.search.random_battles import prepare_random_battles  # noqa: E402
from data import all_move_json as FP_MOVE_JSON  # noqa: E402
from data.pkmn_sets import RandomBattleTeamDatasets  # noqa: E402
import constants  # noqa: E402
from poke_engine import State as PokeEngineState, monte_carlo_tree_search  # noqa: E402


def _run_mcts(state_str: str, search_time_ms: int):
    try:
        state = PokeEngineState.from_string(state_str)
        return monte_carlo_tree_search(state, search_time_ms)
    except Exception:
        return None


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
        self._mcts_stats = {
            "calls": 0,
            "states_sampled": 0,
            "results_kept": 0,
            "result_none": 0,
            "result_errors": 0,
            "empty_results": 0,
            "deterministic_decisions": 0,
            "stochastic_decisions": 0,
            "fallback_super": 0,
            "fallback_random": 0,
            "adaptive_triggered": 0,
            "adaptive_heuristic_used": 0,
            "adaptive_heuristic_failed": 0,
            "adaptive_super_used": 0,
            "adaptive_rerank_used": 0,
            "adaptive_rerank_failed": 0,
            "adaptive_second_pass_used": 0,
            "adaptive_second_pass_failed": 0,
            "diag_turns": 0,
            "diag_low_conf_turns": 0,
            "diag_low_margin_turns": 0,
            "diag_non_top1_choices": 0,
            "diag_choice_delta_sum": 0.0,
            "diag_move_choices": 0,
            "diag_switch_choices": 0,
            "diag_tera_choices": 0,
            "diag_forced_switch_turns": 0,
            "diag_hazard_switch_choices": 0,
            "diag_passive_no_progress_turns": 0,
            "diag_path_mcts": 0,
            "diag_path_adaptive_rerank": 0,
            "diag_path_policy": 0,
            "diag_path_rerank": 0,
            "diag_path_blend": 0,
            "diag_path_fallback_super": 0,
            "diag_path_fallback_random": 0,
            "diag_adaptive_reason_triggered": 0,
            "diag_adaptive_reason_disabled": 0,
            "diag_adaptive_reason_empty": 0,
            "diag_adaptive_reason_early_turn": 0,
            "diag_adaptive_reason_cooldown": 0,
            "diag_adaptive_reason_confidence": 0,
            "diag_adaptive_reason_top_ratio": 0,
            "diag_adaptive_reason_no_heuristics": 0,
            "diag_adaptive_reason_high_heuristic": 0,
            "diag_adaptive_reason_damaging_available": 0,
            "diag_adaptive_reason_not_stallish": 0,
            "diag_battles_finished": 0,
            "diag_battles_won": 0,
            "diag_battles_lost": 0,
            "diag_loss_fast": 0,
            "diag_loss_low_conf": 0,
            "diag_loss_switch_heavy": 0,
            "diag_loss_status_loop": 0,
            "diag_loss_forced_switch": 0,
            "diag_loss_passive": 0,
            "diag_loss_hazard_pivot": 0,
            "diag_loss_tempo": 0,
            "diag_loss_adaptive_used": 0,
            "diag_loss_churn_breaks": 0,
            "diag_loss_other": 0,
        }
        self._rl_prior_ready = False
        self._rl_prior_failed = False
        self._rl_prior_model = None
        self._rl_prior_feature_builder = None
        self._rl_prior_device = "cpu"
        self._rl_prior_torch = None
        self._rl_prior_checkpoint = ""
        self._diag_finished_battle_tags = set()

    def _init_rl_prior(self) -> bool:
        if self._rl_prior_failed:
            return False
        if self._rl_prior_ready and self._rl_prior_model is not None:
            return True
        if not self.RL_PRIOR_ENABLED:
            return False
        try:
            import torch  # local import to avoid hard dependency on non-RL runs
            from src.models.actor_critic import ActorCritic, RecurrentActorCritic
            from src.utils.features import EnhancedFeatureBuilder
            from training.config import RLConfig
        except Exception:
            self._rl_prior_failed = True
            return False

        ckpt_path = Path(self.RL_PRIOR_CHECKPOINT)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parents[2] / ckpt_path
        if not ckpt_path.exists():
            self._rl_prior_failed = True
            return False

        device = self.RL_PRIOR_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        try:
            checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            ckpt_config = checkpoint.get("config", {})
            config = RLConfig()
            config.feature_dim = ckpt_config.get("feature_dim", config.feature_dim)
            config.prediction_features_enabled = ckpt_config.get(
                "prediction_features_enabled", config.prediction_features_enabled
            )
            d_model = ckpt_config.get("d_model", config.d_model)
            n_actions = ckpt_config.get("n_actions", config.n_actions)
            model_type = checkpoint.get("model_type", "feedforward")
            if model_type == "recurrent":
                model = RecurrentActorCritic(
                    feature_dim=config.feature_dim,
                    d_model=d_model,
                    n_actions=n_actions,
                    rnn_hidden=ckpt_config.get("rnn_hidden", config.rnn_hidden),
                    rnn_layers=ckpt_config.get("rnn_layers", config.rnn_layers),
                ).to(device)
            else:
                model = ActorCritic(
                    feature_dim=config.feature_dim,
                    d_model=d_model,
                    n_actions=n_actions,
                ).to(device)

            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._rl_prior_failed = True
                return False
            model.eval()
            self._rl_prior_model = model
            self._rl_prior_feature_builder = EnhancedFeatureBuilder(
                enable_prediction_features=getattr(config, "prediction_features_enabled", False)
            )
            self._rl_prior_device = device
            self._rl_prior_torch = torch
            self._rl_prior_checkpoint = str(ckpt_path)
            self._rl_prior_ready = True
            return True
        except Exception:
            self._rl_prior_failed = True
            return False

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
        if not choices or not self.RL_PRIOR_ENABLED:
            return None
        if not self._init_rl_prior():
            return None
        torch = self._rl_prior_torch
        model = self._rl_prior_model
        feature_builder = self._rl_prior_feature_builder
        if torch is None or model is None or feature_builder is None:
            return None
        try:
            features = feature_builder.build(battle)
            features = [
                0.0 if (not isinstance(f, (int, float)) or f != f or f > 1e6 or f < -1e6) else float(f)
                for f in features
            ]
            mask, move_map, switch_map = self._build_rl_action_mask_and_maps(battle)
            if not any(mask):
                return None
            mask_t = torch.tensor([mask], dtype=torch.bool, device=self._rl_prior_device)
            feat_t = torch.tensor([features], dtype=torch.float32, device=self._rl_prior_device)

            with torch.no_grad():
                if getattr(model, "is_recurrent", False):
                    logits, _, _ = model.forward_step(feat_t, None)
                    logits = logits.masked_fill(~mask_t, -1e9)
                else:
                    logits, _ = model(feat_t, mask_t)
                logits = torch.clamp(logits, min=-1e8, max=1e8)
                probs = torch.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            probs_arr = probs[0].detach().cpu().tolist()

            priors: List[float] = []
            for choice in choices:
                idx = self._choice_to_rl_action_idx(choice, mask, move_map, switch_map)
                if idx is None or idx >= len(probs_arr):
                    priors.append(0.0)
                else:
                    priors.append(max(0.0, float(probs_arr[idx])))
            if sum(priors) <= 0:
                return None
            return priors
        except Exception:
            self._rl_prior_failed = True
            return None

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
        stats = dict(self._mcts_stats)
        calls = max(1, int(stats.get("calls", 0)))
        sampled = max(1, int(stats.get("states_sampled", 0)))
        stats["empty_results_rate"] = float(stats.get("empty_results", 0)) / calls
        stats["fallback_super_rate"] = float(stats.get("fallback_super", 0)) / calls
        stats["fallback_random_rate"] = float(stats.get("fallback_random", 0)) / calls
        stats["state_failure_rate"] = float(
            stats.get("result_none", 0) + stats.get("result_errors", 0)
        ) / sampled
        diag_turns = max(1, int(stats.get("diag_turns", 0)))
        stats["diag_low_conf_rate"] = float(stats.get("diag_low_conf_turns", 0)) / diag_turns
        stats["diag_low_margin_rate"] = float(stats.get("diag_low_margin_turns", 0)) / diag_turns
        stats["diag_non_top1_rate"] = float(stats.get("diag_non_top1_choices", 0)) / diag_turns
        stats["diag_switch_rate"] = float(stats.get("diag_switch_choices", 0)) / diag_turns
        stats["diag_choice_delta_avg"] = float(stats.get("diag_choice_delta_sum", 0.0)) / diag_turns
        return stats

    def _diag_record_adaptive_reason(self, reason: str) -> None:
        if not self.DECISION_DIAG_ENABLED:
            return
        if not reason:
            return
        key = f"diag_adaptive_reason_{reason}"
        self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1

    def _diag_record_choice(
        self,
        battle: Battle,
        ordered: List[Tuple[str, float]],
        chosen: str,
        confidence: float,
        threshold: float,
        path: str,
    ) -> None:
        if not self.DECISION_DIAG_ENABLED:
            return
        mem = self._get_battle_memory(battle)
        self._mcts_stats["diag_turns"] += 1
        mem["diag_turns"] = int(mem.get("diag_turns", 0) or 0) + 1
        mem["diag_status_stall_peak"] = max(
            int(mem.get("diag_status_stall_peak", 0) or 0),
            int(mem.get("status_stall_streak", 0) or 0),
        )

        if confidence < threshold:
            self._mcts_stats["diag_low_conf_turns"] += 1
            mem["diag_low_conf_turns"] = int(mem.get("diag_low_conf_turns", 0) or 0) + 1

        if ordered:
            total = sum(max(0.0, float(w)) for _, w in ordered)
            best_choice = ordered[0][0]
            best_weight = max(0.0, float(ordered[0][1]))
            second_weight = max(0.0, float(ordered[1][1])) if len(ordered) > 1 else 0.0
            margin = ((best_weight - second_weight) / total) if total > 0 else 0.0
            if margin < max(0.0, self.DECISION_DIAG_LOW_MARGIN):
                self._mcts_stats["diag_low_margin_turns"] += 1
                mem["diag_low_margin_turns"] = int(mem.get("diag_low_margin_turns", 0) or 0) + 1
            if chosen and chosen != best_choice:
                self._mcts_stats["diag_non_top1_choices"] += 1
                mem["diag_non_top1_choices"] = int(mem.get("diag_non_top1_choices", 0) or 0) + 1
                best_prob = (best_weight / total) if total > 0 else 0.0
                chosen_weight = 0.0
                for c, w in ordered:
                    if c == chosen:
                        chosen_weight = max(0.0, float(w))
                        break
                chosen_prob = (chosen_weight / total) if total > 0 else 0.0
                delta = max(0.0, best_prob - chosen_prob)
                self._mcts_stats["diag_choice_delta_sum"] += delta
                mem["diag_choice_delta_sum"] = float(mem.get("diag_choice_delta_sum", 0.0) or 0.0) + delta

        if chosen.startswith("switch "):
            self._mcts_stats["diag_switch_choices"] += 1
            mem["diag_switch_choices"] = int(mem.get("diag_switch_choices", 0) or 0) + 1
            if self._side_hazard_pressure(battle) > 0:
                self._mcts_stats["diag_hazard_switch_choices"] += 1
                mem["diag_hazard_switch_choices"] = int(
                    mem.get("diag_hazard_switch_choices", 0) or 0
                ) + 1
        elif chosen:
            self._mcts_stats["diag_move_choices"] += 1
            mem["diag_move_choices"] = int(mem.get("diag_move_choices", 0) or 0) + 1
        if chosen.endswith("-tera"):
            self._mcts_stats["diag_tera_choices"] += 1
            mem["diag_tera_choices"] = int(mem.get("diag_tera_choices", 0) or 0) + 1

        path_key = f"diag_path_{path}"
        self._mcts_stats[path_key] = int(self._mcts_stats.get(path_key, 0) or 0) + 1
        mem[path_key] = int(mem.get(path_key, 0) or 0) + 1

        if self.DECISION_DIAG_ENABLED and self.DECISION_DIAG_LOG and ordered:
            topk = max(1, self.DECISION_DIAG_TOPK)
            head = ", ".join(
                f"{c}:{w:.3f}" for c, w in ordered[:topk]
            )
            print(
                "[diag] turn={} conf={:.3f}/{:.3f} path={} chosen={} top={}".format(
                    int(getattr(battle, "turn", 0) or 0),
                    float(confidence),
                    float(threshold),
                    path,
                    chosen or "<none>",
                    head,
                )
            )

    def _flush_finished_battle_diags(self) -> None:
        if not self.DECISION_DIAG_ENABLED:
            return
        battles = getattr(self, "battles", {}) or {}
        if not battles:
            return
        battle_memory = getattr(self, "_battle_memory", {}) or {}
        for tag, battle in battles.items():
            if tag in self._diag_finished_battle_tags:
                continue
            if not getattr(battle, "finished", False):
                continue
            self._diag_finished_battle_tags.add(tag)
            self._mcts_stats["diag_battles_finished"] += 1
            won = bool(getattr(battle, "won", False))
            lost = bool(getattr(battle, "lost", False))
            if won:
                self._mcts_stats["diag_battles_won"] += 1
            elif lost:
                self._mcts_stats["diag_battles_lost"] += 1

            mem = battle_memory.get(tag, {}) if isinstance(battle_memory, dict) else {}
            if not isinstance(mem, dict):
                mem = {}
            if not lost:
                continue

            tags = 0
            turns = int(getattr(battle, "turn", 0) or 0)
            if 0 < turns <= 12:
                self._mcts_stats["diag_loss_fast"] += 1
                tags += 1

            diag_turns = int(mem.get("diag_turns", 0) or 0)
            low_conf = int(mem.get("diag_low_conf_turns", 0) or 0)
            switch_turns = int(mem.get("diag_switch_choices", 0) or 0)
            low_margin = int(mem.get("diag_low_margin_turns", 0) or 0)
            hazard_switch_turns = int(mem.get("diag_hazard_switch_choices", 0) or 0)
            passive_turns = int(mem.get("diag_passive_no_progress_turns", 0) or 0)
            if diag_turns > 0 and (low_conf / diag_turns) >= 0.5:
                self._mcts_stats["diag_loss_low_conf"] += 1
                tags += 1
            if diag_turns > 0 and (switch_turns / diag_turns) >= 0.45:
                self._mcts_stats["diag_loss_switch_heavy"] += 1
                tags += 1

            passive_ratio = (passive_turns / max(1, diag_turns)) if diag_turns > 0 else 0.0
            if (
                int(mem.get("diag_status_stall_peak", 0) or 0) >= 2
                or passive_ratio >= max(0.0, self.LOSS_PASSIVE_RATIO)
            ):
                self._mcts_stats["diag_loss_status_loop"] += 1
                self._mcts_stats["diag_loss_passive"] += 1
                tags += 1
            forced_turns = int(mem.get("diag_forced_switch_turns", 0) or 0)
            if diag_turns > 0:
                forced_ratio = forced_turns / max(1, diag_turns)
            else:
                forced_ratio = 0.0
            if (
                turns <= 24
                and forced_turns >= 4
                and forced_ratio >= max(0.0, min(1.0, self.LOSS_FORCED_SWITCH_RATIO))
            ):
                self._mcts_stats["diag_loss_forced_switch"] += 1
                tags += 1
            hazard_switch_ratio = (
                hazard_switch_turns / max(1, diag_turns)
                if diag_turns > 0
                else 0.0
            )
            if (
                diag_turns > 0
                and hazard_switch_ratio >= max(0.0, self.LOSS_HAZARD_SWITCH_RATIO)
                and (switch_turns / max(1, diag_turns)) >= 0.25
            ):
                self._mcts_stats["diag_loss_hazard_pivot"] += 1
                tags += 1
            tempo_ratio = max(low_conf, low_margin) / max(1, diag_turns)
            if (
                tempo_ratio >= max(0.0, self.LOSS_TEMPO_RATIO)
                and passive_ratio < max(0.0, self.LOSS_PASSIVE_RATIO)
            ):
                self._mcts_stats["diag_loss_tempo"] += 1
                tags += 1
            if int(mem.get("adaptive_fallback_pending", 0) or 0) == 1 or int(mem.get("diag_adaptive_triggered", 0) or 0) > 0:
                self._mcts_stats["diag_loss_adaptive_used"] += 1
                tags += 1
            if int(mem.get("switch_churn_breaks", 0) or 0) > 0:
                self._mcts_stats["diag_loss_churn_breaks"] += 1
                tags += 1
            if tags == 0:
                self._mcts_stats["diag_loss_other"] += 1

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
        mem = self._get_battle_memory(battle)
        state = mem.get("weather_state")
        if not state or not battle.weather:
            return 0
        if not isinstance(state, dict):
            return -1
        weather_id = state.get("type")
        current_weather = self._map_weather(battle)
        mapped = None
        if weather_id:
            mapping = {
                "raindance": constants.RAIN,
                "rain": constants.RAIN,
                "sunnyday": constants.SUN,
                "sun": constants.SUN,
                "sandstorm": constants.SAND,
                "hail": constants.HAIL,
                "snowscape": constants.SNOW,
                "snow": constants.SNOW,
            }
            mapped = mapping.get(weather_id)
        if mapped and current_weather and mapped != current_weather:
            return -1
        start = state.get("start")
        duration = state.get("duration", 5)
        if not isinstance(start, int):
            return -1
        remaining = duration - max(0, battle.turn - start)
        return max(0, remaining)

    def _terrain_turns_remaining(self, battle: Battle) -> int:
        mem = self._get_battle_memory(battle)
        state = mem.get("terrain_state")
        if not state or not battle.fields:
            return 0
        if not isinstance(state, dict):
            return -1
        terrain_id = state.get("type")
        current_terrain = self._map_terrain(battle)
        mapped = None
        if terrain_id:
            mapping = {
                "electricterrain": constants.ELECTRIC_TERRAIN,
                "grassyterrain": constants.GRASSY_TERRAIN,
                "mistyterrain": constants.MISTY_TERRAIN,
                "psychicterrain": constants.PSYCHIC_TERRAIN,
            }
            mapped = mapping.get(terrain_id)
        if mapped and current_terrain and mapped != current_terrain:
            return -1
        start = state.get("start")
        duration = state.get("duration", 5)
        if not isinstance(start, int):
            return -1
        remaining = duration - max(0, battle.turn - start)
        return max(0, remaining)

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
        mem = self._get_battle_memory(battle)
        pending = mem.get("pending_passive_action")
        if not isinstance(pending, dict):
            return
        pending_turn = int(pending.get("turn", -1) or -1)
        current_turn = int(getattr(battle, "turn", 0) or 0)
        if current_turn <= pending_turn:
            return

        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon
        prev_opp_hp = pending.get("opp_hp")
        cur_opp_hp = getattr(opponent, "current_hp_fraction", None)
        opp_hp_drop = 0.0
        if isinstance(prev_opp_hp, (int, float)) and isinstance(cur_opp_hp, (int, float)):
            opp_hp_drop = max(0.0, float(prev_opp_hp) - float(cur_opp_hp))

        prev_self_hp = pending.get("self_hp")
        cur_self_hp = getattr(active, "current_hp_fraction", None)
        self_hp_gain = 0.0
        if isinstance(prev_self_hp, (int, float)) and isinstance(cur_self_hp, (int, float)):
            self_hp_gain = max(0.0, float(cur_self_hp) - float(prev_self_hp))

        prev_markers = pending.get("opp_markers", {}) or {}
        cur_markers = self._opponent_progress_markers(battle, opponent)
        progress = opp_hp_drop >= 0.06
        progress = progress or cur_markers.get("status") != prev_markers.get("status")
        for key in ("rocks", "web", "spikes", "tspikes"):
            if int(cur_markers.get(key, 0) or 0) > int(prev_markers.get(key, 0) or 0):
                progress = True
                break

        kind = pending.get("kind", "")
        if kind == "recovery" and self_hp_gain >= 0.20:
            progress = True

        if progress:
            mem["passive_no_progress_streak"] = 0
        else:
            mem["passive_no_progress_streak"] = int(mem.get("passive_no_progress_streak", 0) or 0) + 1
            if self.DECISION_DIAG_ENABLED:
                self._mcts_stats["diag_passive_no_progress_turns"] += 1
                mem["diag_passive_no_progress_turns"] = int(
                    mem.get("diag_passive_no_progress_turns", 0) or 0
                ) + 1
        mem.pop("pending_passive_action", None)

    def _passive_choice_kind(self, move) -> str:
        if move is None:
            return ""
        move_id = normalize_name(getattr(move, "id", "") or "")
        if move_id in self.PROTECT_MOVES:
            return "protect"
        if self._is_recovery_move(move):
            return "recovery"
        try:
            if move.category == MoveCategory.STATUS:
                return "status"
        except Exception:
            return ""
        return ""

    def _progress_need_score(
        self,
        battle: Battle,
        active: Pokemon,
        opponent: Pokemon,
        best_damage_score: float,
    ) -> int:
        score = 0
        if self._estimate_matchup(active, opponent) <= self.PROGRESS_NEED_MATCHUP:
            score += 1
        if self._side_hazard_pressure(battle) > 0:
            score += 1
        if self._estimate_best_reply_score(opponent, active, battle) >= self.PROGRESS_NEED_REPLY:
            score += 1
        if best_damage_score >= self.PROGRESS_NEED_DAMAGE and (opponent.current_hp_fraction or 0.0) >= 0.25:
            score += 1
        return score

    # ------------------------------------------------------------------
    # Damage-belief: capture attacker context at action time
    # ------------------------------------------------------------------
    def _record_last_action(self, battle: Battle, order) -> None:
        super()._record_last_action(battle, order)
        mem = self._get_battle_memory(battle)
        order_obj = getattr(order, "order", None)
        passive_kind = self._passive_choice_kind(order_obj)
        if passive_kind:
            mem["pending_passive_action"] = {
                "turn": int(getattr(battle, "turn", 0) or 0),
                "kind": passive_kind,
                "self_hp": getattr(getattr(battle, "active_pokemon", None), "current_hp_fraction", None),
                "opp_hp": getattr(
                    getattr(battle, "opponent_active_pokemon", None), "current_hp_fraction", None
                ),
                "opp_markers": self._opponent_progress_markers(
                    battle, getattr(battle, "opponent_active_pokemon", None)
                ),
            }
        else:
            mem.pop("pending_passive_action", None)
            mem["passive_no_progress_streak"] = 0
        if not self.DAMAGE_BELIEF:
            return
        if not hasattr(order_obj, "category"):
            # Not a move — nothing extra to record
            mem.pop("_dmg_pending", None)
            return
        move = order_obj
        if move.category == MoveCategory.STATUS:
            mem.pop("_dmg_pending", None)
            return
        move_id = normalize_name(getattr(move, "id", "") or "")
        if move_id in self.DAMAGE_BELIEF_UNSTABLE_MOVES:
            mem.pop("_dmg_pending", None)
            return

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            mem.pop("_dmg_pending", None)
            return
        if self._damage_belief_has_unmodeled_state(battle):
            mem.pop("_dmg_pending", None)
            return

        # Base power
        entry = self._get_move_entry(move)
        bp = int(entry.get("basePower", 0) or 0) or (move.base_power or 0)
        if bp <= 0:
            mem.pop("_dmg_pending", None)
            return

        # Skip multi-hit moves (unreliable single-observation scoring)
        multihit = entry.get("multihit")
        if multihit:
            mem.pop("_dmg_pending", None)
            return

        move_cat = "physical" if move.category == MoveCategory.PHYSICAL else "special"
        move_type_str = self._move_type_id(move) or ""
        if not move_type_str:
            mem.pop("_dmg_pending", None)
            return

        # Attacker actual stat (already including base+level+IV+EV+nature, no boosts)
        atk_stats = active.stats or {}
        if move_cat == "physical":
            stat_key = "atk"
        else:
            stat_key = "spa"
        raw_stat = atk_stats.get(stat_key) or 100

        # Apply boost to get effective stat
        boost_val = (active.boosts or {}).get(stat_key, 0) or 0
        if boost_val > 0:
            eff_stat = int(raw_stat * (2 + boost_val) / 2)
        elif boost_val < 0:
            eff_stat = int(raw_stat * 2 / (2 - boost_val))
        else:
            eff_stat = raw_stat

        atk_types = [t.name.lower() for t in active.types if t is not None] if active.types else []
        atk_ability_str = ""
        if active.ability:
            atk_ability_str = normalize_name(str(active.ability))
        atk_item_str = ""
        if active.item:
            atk_item_str = normalize_name(str(active.item))

        atk_status_str = ""
        if active.status:
            atk_status_str = normalize_name(str(active.status))

        opp_types = [t.name.lower() for t in opponent.types if t is not None] if opponent.types else []
        opp_boosts = dict(opponent.boosts) if opponent.boosts else {}

        weather_str = ""
        for w in battle.weather:
            weather_str = w.name.lower() if hasattr(w, "name") else str(w).lower()
            break

        terrain_str = ""
        for f in battle.fields:
            fn = f.name.lower() if hasattr(f, "name") else str(f).lower()
            if "terrain" in fn:
                terrain_str = fn
                break

        mem["_dmg_pending"] = {
            "turn": battle.turn,
            "move_id": move_id,
            "move_bp": bp,
            "move_type": move_type_str,
            "move_category": move_cat,
            "attacker_stat": eff_stat,
            "attacker_level": active.level,
            "attacker_types": atk_types,
            "attacker_boosts": {stat_key: boost_val},
            "attacker_status": atk_status_str,
            "attacker_ability": atk_ability_str,
            "attacker_item": atk_item_str,
            "defender_types": opp_types,
            "defender_boosts": opp_boosts,
            "opponent_species": normalize_name(opponent.species),
            "weather": weather_str,
            "terrain": terrain_str,
            "high_confidence": True,
        }

    # ------------------------------------------------------------------
    # Damage-belief: parse events to find observed damage
    # ------------------------------------------------------------------
    def _update_damage_observation(self, battle: Battle) -> None:
        if not self.DAMAGE_BELIEF:
            return
        mem = self._get_battle_memory(battle)
        mem.setdefault("damage_observations", {})

        pending = mem.get("_dmg_pending")
        if not pending:
            return
        pending_turn = pending.get("turn", -1)
        last_obs_turn = mem.get("_dmg_last_obs_turn", -1)
        if pending_turn <= last_obs_turn:
            return
        # The pending was recorded at commit time (turn T). The events for
        # that turn appear in observations[T] (after the server resolves).
        last_turn = pending_turn
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        if obs is None:
            return

        role = getattr(battle, "player_role", None)
        if not role:
            return

        # Mark processed
        mem["_dmg_last_obs_turn"] = pending_turn

        species = pending["opponent_species"]

        # Walk events to find our move and the resulting damage on the opponent
        opp_prefix = "p2" if role == "p1" else "p1"
        our_prefix = role

        # Track current opponent HP fraction through events
        tracked_hp = None  # will be set as we encounter events
        found_our_move = False
        pre_hit_hp = None
        post_hit_hp = None
        was_crit = False
        skip = False

        for event in obs.events:
            if len(event) < 2:
                continue
            kind = event[1]

            # --- Before our move: track opponent HP changes ---
            if not found_our_move:
                # Switch/drag for opponent: parse HP
                if kind in ("switch", "drag") and len(event) >= 5:
                    who = event[2]
                    if who.startswith(opp_prefix):
                        hp_str = event[4] if len(event) >= 5 else ""
                        tracked_hp = self._parse_hp_fraction(hp_str)
                        # If species changed, this observation doesn't apply
                        ev_species = self._species_from_event(battle, event)
                        if ev_species and ev_species != species:
                            skip = True
                            break

                # HP changes on opponent before our move
                if kind in ("-damage", "-heal") and len(event) >= 4:
                    who = event[2]
                    if who.startswith(opp_prefix):
                        tracked_hp = self._parse_hp_fraction(event[3])

                # Our move event
                if kind == "move" and len(event) >= 4:
                    who = event[2]
                    if who.startswith(our_prefix):
                        if tracked_hp is None:
                            # Infer from opponent's current state at start of turn
                            opp = battle.opponent_active_pokemon
                            if opp and opp.species and normalize_name(opp.species) == species:
                                # We don't have a better source; try the HP we
                                # recorded at commit time
                                tracked_hp = mem.get("last_opponent_hp")
                        pre_hit_hp = tracked_hp
                        found_our_move = True
                        continue

            # --- After our move: look for damage on opponent ---
            if found_our_move:
                if kind == "-miss":
                    skip = True
                    break
                if kind == "-immune":
                    skip = True
                    break
                if kind == "-fail":
                    skip = True
                    break
                if kind == "-activate" and len(event) >= 4:
                    act = event[3].lower() if len(event) >= 4 else ""
                    if "protect" in act or "substitute" in act:
                        skip = True
                        break

                if kind == "-crit":
                    was_crit = True
                    continue

                if kind in ("-damage", "damage") and len(event) >= 4:
                    who = event[2]
                    if who.startswith(opp_prefix):
                        # Check for indirect damage sources (residual, item, etc.)
                        ev_lower = " ".join(event).lower()
                        has_from = "[from]" in ev_lower
                        if has_from and "move:" not in ev_lower:
                            # Residual damage (Stealth Rock, poison, etc.) — skip
                            continue
                        post_hit_hp = self._parse_hp_fraction(event[3])
                        break

                # Another move event means the damage window passed
                if kind == "move":
                    break
                # Switch events also end the window
                if kind in ("switch", "drag"):
                    break

        if skip or pre_hit_hp is None or post_hit_hp is None:
            return

        observed_frac = pre_hit_hp - post_hit_hp
        if observed_frac <= 0.01:
            return

        # Build the observation record
        obs_record = {
            "move_id": pending.get("move_id", ""),
            "move_bp": pending["move_bp"],
            "move_type": pending["move_type"],
            "move_category": pending["move_category"],
            "attacker_stat": pending["attacker_stat"],
            "attacker_level": pending["attacker_level"],
            "attacker_types": pending["attacker_types"],
            "attacker_boosts": pending.get("attacker_boosts", {}),
            "attacker_status": pending.get("attacker_status", ""),
            "attacker_ability": pending.get("attacker_ability", ""),
            "attacker_item": pending.get("attacker_item", ""),
            "defender_boosts": pending.get("defender_boosts", {}),
            "defender_hp_frac": pre_hit_hp,
            "observed_frac": observed_frac,
            "weather": pending.get("weather", ""),
            "terrain": pending.get("terrain", ""),
            "is_crit": was_crit,
            "high_confidence": bool(pending.get("high_confidence", False)),
        }
        mem["damage_observations"].setdefault(species, []).append(obs_record)
        # Cap stored observations per species
        if len(mem["damage_observations"][species]) > 8:
            mem["damage_observations"][species] = mem["damage_observations"][species][-8:]

    @staticmethod
    def _parse_hp_fraction(hp_str: str) -> Optional[float]:
        """Parse HP string like '78/100' or '0 fnt' into a fraction [0..1]."""
        if not hp_str:
            return None
        try:
            hp_str = hp_str.strip().split()[0]  # drop "fnt" or condition tags
            if "/" in hp_str:
                parts = hp_str.split("/")
                cur = float(parts[0])
                mx = float(parts[1])
                if mx <= 0:
                    return 0.0
                return cur / mx
            # Percentage like "78"
            val = float(hp_str)
            if val > 1.0:
                return val / 100.0
            return val
        except Exception:
            return None

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
        last_turn = battle.turn - 1
        observations = getattr(battle, "observations", {})
        obs = observations.get(last_turn)
        role = getattr(battle, "player_role", None)
        if obs is None or not role or last_turn < 1:
            return None
        for event in obs.events:
            if len(event) < 4:
                continue
            if event[1] != "move":
                continue
            who = event[2]
            if who.startswith(role):
                continue
            species = self._species_from_event(battle, event) or ""
            move_id = normalize_name(event[3])
            if move_id:
                return species, move_id, last_turn
        return None

    def _sample_set_for_species(
        self,
        species: str,
        battle: Battle,
        mon: Optional[Pokemon] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[dict]:
        picker = rng or random
        if mon is not None:
            candidates = self._candidate_randombattle_sets(mon, battle)
            if candidates:
                weights = [c for _, c in candidates]
                choice = picker.choices(candidates, weights=weights, k=1)[0][0]
                return choice
        data = self._load_randombattle_sets()
        sets = data.get(species, {})
        if not sets:
            return None
        keys = list(sets.keys())
        weights = [sets[k] for k in keys]
        key = picker.choices(keys, weights=weights, k=1)[0]
        return self._parse_randombattle_set_key(key)

    def _sample_unknown_opponents(
        self,
        battle: Battle,
        taken: set,
        count: int,
        rng: Optional[random.Random] = None,
    ) -> List[FPPokemon]:
        picker = rng or random
        data = self._load_randombattle_sets()
        if not data:
            return []
        names = []
        weights = []
        for name, sets in data.items():
            if name in taken:
                continue
            total = sum(sets.values())
            if total <= 0:
                continue
            names.append(name)
            weights.append(total)
        result = []
        for _ in range(count):
            if not names:
                break
            species = picker.choices(names, weights=weights, k=1)[0]
            set_info = self._sample_set_for_species(species, battle, rng=picker)
            if not set_info:
                continue
            fp_mon = FPPokemon(species, set_info.get("level", 100))
            fp_mon.ability = set_info.get("ability") or fp_mon.ability
            fp_mon.item = set_info.get("item") or fp_mon.item
            fp_mon.tera_type = set_info.get("tera") or fp_mon.tera_type
            self._fill_moves_from_set(fp_mon, set_info, set())
            result.append(fp_mon)
            taken.add(species)
        return result

    def _build_fp_battle(self, battle: Battle, seed: int, fill_opponent_sets: bool = False) -> FPBattle:
        rng = random.Random(seed)
        fp_battle = FPBattle(battle.battle_tag)
        fp_battle.turn = battle.turn
        fp_battle.force_switch = battle.force_switch
        fp_battle.team_preview = getattr(battle, "team_preview", None)
        if fp_battle.team_preview is None:
            fp_battle.team_preview = getattr(battle, "teampreview", False)
        fp_battle.weather = self._map_weather(battle)
        fp_battle.field = self._map_terrain(battle)
        fp_battle.trick_room = Field.TRICK_ROOM in (battle.fields or {})
        fp_battle.trick_room_turns_remaining = -1 if fp_battle.trick_room else 0
        fp_battle.weather_turns_remaining = self._weather_turns_remaining(battle)
        fp_battle.field_turns_remaining = self._terrain_turns_remaining(battle)
        fp_battle.pokemon_format = getattr(battle, "format", None) or "gen9randombattle"
        fp_battle.generation = f"gen{getattr(battle, 'gen', 9) or 9}"
        if fp_battle.pokemon_format and "randombattle" in fp_battle.pokemon_format:
            fp_battle.battle_type = constants.BattleType.RANDOM_BATTLE

        user = Battler()
        opponent = Battler()
        user.last_used_move = LastUsedMove("", "", 0)
        opponent.last_used_move = LastUsedMove("", "", 0)
        user.last_selected_move = LastUsedMove("", "", 0)
        opponent.last_selected_move = LastUsedMove("", "", 0)
        # User team
        active = battle.active_pokemon
        try:
            user.trapped = bool(getattr(battle, "trapped", False))
        except Exception:
            user.trapped = False
        if active and self._is_trapped(active):
            user.trapped = True

        # Side conditions
        self._map_side_conditions(battle.side_conditions, user.side_conditions)
        self._map_side_conditions(battle.opponent_side_conditions, opponent.side_conditions)

        if active:
            user.active = self._poke_env_to_fp(active, battle, None)
            user.active.can_terastallize = bool(getattr(battle, "can_tera", False))
        reserves = []
        for mon in battle.team.values():
            if mon is None or mon == active:
                continue
            reserves.append(self._poke_env_to_fp(mon, battle, None))
        user.reserve = reserves
        if user.active and battle.available_moves:
            available = {normalize_name(m.id) for m in battle.available_moves}
            for mv in user.active.moves:
                move_id = self._fp_move_id(mv)
                mv.disabled = bool(move_id) and move_id not in available
            if self._sleep_clause_blocked(battle):
                for mv in user.active.moves:
                    if self._fp_move_inflicts_sleep(mv.name):
                        mv.disabled = True
            mem = self._get_battle_memory(battle)
            if mem.get("last_action") == "move":
                last_turn = mem.get("last_action_turn", 0)
                move_id = mem.get("last_move_id") or ""
                if last_turn and move_id:
                    user.last_used_move = LastUsedMove(user.active.name, move_id, last_turn)
                    user.last_selected_move = user.last_used_move
        self._clear_invalid_encore(user)

        # Opponent team
        opp_active = battle.opponent_active_pokemon
        if opp_active:
            set_info = None
            if fill_opponent_sets:
                set_info = self._sample_set_for_species(
                    normalize_name(opp_active.species), battle, opp_active, rng=rng
                )
            opponent.active = self._poke_env_to_fp(opp_active, battle, set_info)
            opponent.active.can_terastallize = not bool(getattr(battle, "opponent_used_tera", False))
            self._apply_opponent_item_flags(opponent.active, battle)
            self._apply_opponent_ability_flags(opponent.active, battle)
            self._apply_known_opponent_moves(opponent.active, battle)
            self._apply_opponent_switch_memory(opponent.active, battle)
            self._apply_speed_bounds(opponent.active, battle)
            opponent.trapped = self._is_trapped(opp_active)
        opp_reserves = []
        taken = set()
        if opp_active:
            taken.add(normalize_name(opp_active.species))
        for mon in battle.opponent_team.values():
            if mon is None or mon == opp_active:
                continue
            set_info = None
            if fill_opponent_sets:
                set_info = self._sample_set_for_species(normalize_name(mon.species), battle, mon, rng=rng)
            fp_mon = self._poke_env_to_fp(mon, battle, set_info)
            self._apply_opponent_item_flags(fp_mon, battle)
            self._apply_opponent_ability_flags(fp_mon, battle)
            self._apply_known_opponent_moves(fp_mon, battle)
            self._apply_opponent_switch_memory(fp_mon, battle)
            self._apply_speed_bounds(fp_mon, battle)
            opp_reserves.append(fp_mon)
            taken.add(normalize_name(mon.species))
        if fill_opponent_sets:
            missing = max(0, 6 - (len(opp_reserves) + (1 if opp_active else 0)))
            opp_reserves.extend(self._sample_unknown_opponents(battle, taken, missing, rng=rng))
        opponent.reserve = opp_reserves

        last_opp = self._extract_last_opponent_move(battle)
        if last_opp and opponent.active:
            species, move_id, last_turn = last_opp
            opponent.last_used_move = LastUsedMove(species or opponent.active.name, move_id, last_turn)
            opponent.last_selected_move = opponent.last_used_move
        self._clear_invalid_encore(opponent)

        mem = self._get_battle_memory(battle)
        switch_flags = mem.get("switch_flags", {})
        if isinstance(switch_flags, dict):
            self_flags = switch_flags.get("self", {})
            opp_flags = switch_flags.get("opp", {})
            user.baton_passing = bool(self_flags.get("baton", False))
            user.shed_tailing = bool(self_flags.get("shed", False))
            opponent.baton_passing = bool(opp_flags.get("baton", False))
            opponent.shed_tailing = bool(opp_flags.get("shed", False))
        wish = mem.get("pending_wish", {})
        future_sight = mem.get("pending_future_sight", {})
        if isinstance(wish, dict):
            user.wish = wish.get("self", user.wish)
            opponent.wish = wish.get("opp", opponent.wish)
        if isinstance(future_sight, dict):
            user.future_sight = future_sight.get("self", user.future_sight)
            opponent.future_sight = future_sight.get("opp", opponent.future_sight)

        fp_battle.user = user
        fp_battle.opponent = opponent
        if user.active:
            try:
                user.lock_moves()
            except Exception:
                pass
        if opponent.active:
            try:
                opponent.lock_moves()
            except Exception:
                pass
        return fp_battle

    @staticmethod
    def _fp_move_id(move) -> str:
        if move is None:
            return ""
        for attr in ("id", "move_id"):
            try:
                value = getattr(move, attr)
            except Exception:
                value = None
            if value:
                return normalize_name(str(value))
        try:
            name = getattr(move, "name", "")
        except Exception:
            name = ""
        return normalize_name(str(name)) if name else ""

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
        if battle is None or mon is None:
            return False
        if self._has_heavy_duty_boots(mon):
            return False
        hp_frac = mon.current_hp_fraction if mon.current_hp_fraction is not None else 1.0
        if hp_frac <= 0:
            return True
        side_conditions = getattr(battle, "side_conditions", None) or {}
        if not side_conditions:
            return False

        hazard_fraction = 0.0
        if SideCondition.STEALTH_ROCK in side_conditions:
            def_types = [t.name.lower() for t in (mon.types or []) if t is not None]
            rock_mult = get_type_effectiveness("rock", def_types)
            hazard_fraction += (1.0 / 8.0) * rock_mult

        try:
            grounded = battle.is_grounded(mon)
        except Exception:
            grounded = True

        if grounded:
            spikes_layers = side_conditions.get(SideCondition.SPIKES, 0)
            if spikes_layers == 1:
                hazard_fraction += 1.0 / 8.0
            elif spikes_layers == 2:
                hazard_fraction += 1.0 / 6.0
            elif spikes_layers >= 3:
                hazard_fraction += 1.0 / 4.0

        return hazard_fraction >= max(0.0, hp_frac - 1e-6)

    def _status_choice_is_obviously_bad(
        self,
        battle: Battle,
        move,
        active: Pokemon,
        opponent: Pokemon,
    ) -> bool:
        move_entry = self._get_move_entry(move)
        status_type = self.STATUS_MOVES.get(move.id)
        if status_type is None:
            status_type = self._status_from_move_entry(move_entry)
        if status_type is None:
            return False

        if opponent.status is not None and status_type in {"poison", "burn", "para", "sleep", "yawn"}:
            return True
        if status_type == "sleep" and self._sleep_clause_blocked(battle):
            return True
        if status_type == "poison" and (
            self._opponent_has_type(opponent, "steel") or self._opponent_has_type(opponent, "poison")
        ):
            return True
        if status_type == "burn" and self._opponent_has_type(opponent, "fire"):
            return True
        if status_type == "para":
            if move.id == "thunderwave" and (
                self._opponent_has_type(opponent, "ground")
                or self._opponent_has_type(opponent, "electric")
            ):
                return True
        if status_type == "sap" or move.id == "strengthsap":
            opp_atk = self._stat_estimation(opponent, "atk")
            opp_spa = self._stat_estimation(opponent, "spa")
            hp_frac = active.current_hp_fraction or 0.0
            if opp_atk <= opp_spa and hp_frac > 0.45:
                return True

        mem = self._get_battle_memory(battle)
        if (
            int(mem.get("status_stall_streak", 0) or 0) >= self.STATUS_STALL_MAX
            and mem.get("last_move_id") == move.id
        ):
            return True
        return False

    def _apply_tactical_safety(self, battle: Battle, choice: str, active: Pokemon, opponent: Pokemon) -> str:
        if not self.MOVE_SAFETY_GUARD:
            return choice

        best_damage_move, best_damage_score = self._best_damaging_move(battle, active, opponent)
        reply_score = self._estimate_best_reply_score(opponent, active, battle)
        opp_hp = opponent.current_hp_fraction or 0.0
        active_hp = active.current_hp_fraction or 0.0
        ko_threshold = self.TACTICAL_KO_THRESHOLD * max(opp_hp, 0.05)
        has_ko_window = best_damage_move is not None and best_damage_score >= ko_threshold

        if choice.startswith("switch "):
            switch_name = normalize_name(choice.split("switch ", 1)[1])
            chosen_switch = None
            for sw in battle.available_switches:
                if normalize_name(sw.species) == switch_name:
                    chosen_switch = sw
                    break
            if chosen_switch is not None and self._switch_faints_to_entry_hazards(battle, chosen_switch):
                survivable = [sw for sw in battle.available_switches if not self._switch_faints_to_entry_hazards(battle, sw)]
                if survivable:
                    best_sw = max(survivable, key=lambda s: self._score_switch(s, opponent, battle))
                    return f"switch {normalize_name(best_sw.species)}"
                if best_damage_move is not None and not battle.force_switch:
                    return best_damage_move.id
            return choice

        tera_suffix = choice.endswith("-tera")
        move_id = normalize_name(choice.replace("-tera", ""))
        selected_move = None
        for move in battle.available_moves:
            if move.id == move_id:
                selected_move = move
                break
        if selected_move is None:
            return choice

        try:
            is_status = selected_move.category == MoveCategory.STATUS
        except Exception:
            is_status = False
        is_recovery = self._is_recovery_move(selected_move)
        is_setup = bool(getattr(selected_move, "boosts", None) or {}) and (
            selected_move.target and "self" in str(selected_move.target).lower()
        )
        is_protect = selected_move.id in self.PROTECT_MOVES
        mem = self._get_battle_memory(battle)
        stall = int(mem.get("status_stall_streak", 0) or 0)
        passive_streak = int(mem.get("passive_no_progress_streak", 0) or 0)
        passive_kind = self._passive_choice_kind(selected_move)
        progress_need = self._progress_need_score(
            battle,
            active,
            opponent,
            best_damage_score,
        )

        # Loop breaker: when we repeatedly pick low-progress status/recovery/protect
        # lines in uncertain states, force a high-value damaging move.
        if (
            self.LOOP_BREAKER_ENABLED
            and stall >= max(1, self.LOOP_BREAKER_STALL_STREAK)
            and (is_status or is_recovery or is_protect)
            and best_damage_move is not None
            and best_damage_move.id != move_id
            and not battle.force_switch
        ):
            loop_threshold = max(
                self.LOOP_BREAKER_MIN_SCORE,
                self.TACTICAL_KO_THRESHOLD
                * max(0.0, self.LOOP_BREAKER_KO_FRACTION)
                * max(opp_hp, 0.05),
            )
            if best_damage_score >= loop_threshold:
                return best_damage_move.id

        if is_status and self._status_choice_is_obviously_bad(battle, selected_move, active, opponent):
            if best_damage_move is not None:
                return best_damage_move.id

        if (
            passive_kind
            and best_damage_move is not None
            and best_damage_move.id != move_id
            and not battle.force_switch
        ):
            if passive_kind == "protect" and self._should_use_protect(battle, reply_score):
                if progress_need <= 1 and passive_streak <= 0:
                    pass
                elif best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                    return best_damage_move.id
            elif passive_kind == "recovery":
                safe_low_hp_recover = (
                    active_hp <= self.PASSIVE_BREAK_RECOVERY_HP_MAX and reply_score < self.PROGRESS_NEED_REPLY
                )
                if not safe_low_hp_recover and progress_need >= 2 and (
                    passive_streak >= 1 or active_hp >= 0.55
                ):
                    if best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                        return best_damage_move.id
            elif passive_kind == "status":
                if progress_need >= 2 and (stall >= 1 or passive_streak >= 1):
                    if best_damage_score >= self.PROGRESS_NEED_DAMAGE:
                        return best_damage_move.id

        if has_ko_window and (is_status or is_recovery or is_setup):
            if best_damage_move is not None:
                return best_damage_move.id

        if has_ko_window and is_protect and best_damage_move is not None:
            return best_damage_move.id

        if tera_suffix:
            return f"{move_id}-tera"
        return choice

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
    ) -> List[Tuple[object, float]]:
        fp_battle = base_fp_battle or self._build_fp_battle(
            battle, seed=0, fill_opponent_sets=False
        )
        fp_battles: List[FPBattle] = []
        weights: List[float] = []

        if fp_battle.battle_type == constants.BattleType.RANDOM_BATTLE:
            try:
                self._ensure_randbats_sets(battle)
                samples = prepare_random_battles(fp_battle, sample_states)
                fp_battles = [b for b, _ in samples]
                weights = [w for _, w in samples]
            except Exception:
                fp_battles = [fp_battle]
                weights = [1.0]
        else:
            seeds = [random.randint(1, 1_000_000) for _ in range(sample_states)]
            for seed in seeds:
                fp_battles.append(
                    self._build_fp_battle(battle, seed, fill_opponent_sets=True)
                )
            weights = [1.0 / len(fp_battles)] * len(fp_battles)

        states = [battle_to_poke_engine_state(b).to_string() for b in fp_battles]
        self._mcts_stats["states_sampled"] += len(states)

        results: List[Tuple[object, float]] = []
        workers = min(self.PARALLELISM, len(states))
        if workers > 1:
            executor = self._get_mcts_pool(workers)
            if executor is None:
                workers = 1
        if workers > 1:
            futures = [
                executor.submit(_run_mcts, state, search_time_ms)  # type: ignore[union-attr]
                for state in states
            ]
            for fut, weight in zip(futures, weights):
                try:
                    timeout_sec = max(1.0, (search_time_ms / 1000.0) * 2.0 + 2.0)
                    res = fut.result(timeout=timeout_sec)
                except Exception:
                    self._mcts_stats["result_errors"] += 1
                    continue
                if res is None:
                    self._mcts_stats["result_none"] += 1
                    continue
                results.append((res, weight))
                self._mcts_stats["results_kept"] += 1
        else:
            for state, weight in zip(states, weights):
                res = _run_mcts(state, search_time_ms)
                if res is None:
                    self._mcts_stats["result_none"] += 1
                    continue
                results.append((res, weight))
                self._mcts_stats["results_kept"] += 1
        return results

    def _select_move_from_results(
        self,
        results: List[Tuple[object, float]],
        battle: Battle,
        banned_choices: Optional[set] = None,
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

        results = self._collect_mcts_results(
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
        if self.ADAPTIVE_ESCALATE_ENABLED:
            ordered_pre, _total_pre, conf_pre, th_pre = self._aggregate_policy_from_results(
                results,
                banned_choices=banned_choices,
            )
            if ordered_pre and self._should_trigger_adaptive_fallback(
                battle, ordered_pre, conf_pre, th_pre, record_diag=False
            ):
                boosted_ms = max(
                    search_time_ms,
                    int(search_time_ms * max(1.0, self.ADAPTIVE_ESCALATE_MS_MULT)),
                )
                boosted_ms = min(
                    boosted_ms,
                    max(search_time_ms, self.ADAPTIVE_ESCALATE_MAX_MS),
                )
                boosted_states = max(
                    sample_states,
                    int(sample_states * max(1.0, self.ADAPTIVE_ESCALATE_SAMPLE_MULT)),
                )
                boosted_states = min(
                    boosted_states,
                    max(sample_states, self.ADAPTIVE_ESCALATE_MAX_STATES),
                )
                if boosted_ms > search_time_ms or boosted_states > sample_states:
                    second_results = self._collect_mcts_results(
                        battle,
                        sample_states=boosted_states,
                        search_time_ms=boosted_ms,
                    )
                    if second_results:
                        results = second_results
                        self._mcts_stats["adaptive_second_pass_used"] += 1
                    else:
                        self._mcts_stats["adaptive_second_pass_failed"] += 1

        choice = self._select_move_from_results(results, battle, banned_choices=banned_choices)
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
