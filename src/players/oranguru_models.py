#!/usr/bin/env python3
"""
Model-loading and inference helpers for OranguruEnginePlayer.

This module isolates checkpoint resolution, model initialization, and policy /
value inference helpers from the engine orchestration layer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from poke_env.battle import Battle


def resolve_model_checkpoint(self, path_str: str) -> Path:
    ckpt_path = Path(path_str)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parents[2] / ckpt_path
    return ckpt_path


def resolve_model_device(self, device_name: str, torch_module) -> str:
    device = device_name
    if device == "cuda" and not torch_module.cuda.is_available():
        device = "cpu"
    return device


def load_search_prior_family_model(self, checkpoint_path: str, device_name: str):
    import torch
    from src.models.search_prior_value import SearchPriorValueNet
    from src.utils.features import EnhancedFeatureBuilder

    ckpt_path = self._resolve_model_checkpoint(checkpoint_path)
    if not ckpt_path.exists():
        return None, None, None, None

    device = self._resolve_model_device(device_name, torch)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    board_dim = int(config.get("board_dim", 272))
    action_dim = int(config.get("action_dim", 16))
    n_actions = int(config.get("n_actions", 13))
    hidden_dim = int(config.get("hidden_dim", 256))
    dropout = float(config.get("dropout", 0.1))

    model = SearchPriorValueNet(
        board_dim=board_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
        dropout=dropout,
    ).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    feature_builder = EnhancedFeatureBuilder(enable_prediction_features=False)
    return model, feature_builder, torch, str(ckpt_path)


def init_rl_prior(self) -> bool:
    if self._rl_prior_failed:
        return False
    if self._rl_prior_ready and self._rl_prior_model is not None:
        return True
    if not self.RL_PRIOR_ENABLED:
        return False
    try:
        import torch
        from src.models.actor_critic import ActorCritic, RecurrentActorCritic
        from src.utils.features import EnhancedFeatureBuilder
        from training.config import RLConfig

        ckpt_path = self._resolve_model_checkpoint(self.RL_PRIOR_CHECKPOINT)
        if not ckpt_path.exists():
            self._rl_prior_failed = True
            return False

        device = self._resolve_model_device(self.RL_PRIOR_DEVICE, torch)
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


def _init_search_family_slot(
    self,
    *,
    enabled: bool,
    failed_attr: str,
    ready_attr: str,
    model_attr: str,
    feature_builder_attr: str,
    device_attr: str,
    torch_attr: str,
    checkpoint_attr: str,
    checkpoint_path: str,
    device_name: str,
    stats_fail_key: Optional[str] = None,
) -> bool:
    if getattr(self, failed_attr):
        return False
    if getattr(self, ready_attr) and getattr(self, model_attr) is not None:
        return True
    if not enabled:
        return False
    try:
        model, feature_builder, torch, ckpt_path = self._load_search_prior_family_model(
            checkpoint_path,
            device_name,
        )
        if model is None:
            if stats_fail_key:
                self._mcts_stats[stats_fail_key] = int(self._mcts_stats.get(stats_fail_key, 0) or 0) + 1
            setattr(self, failed_attr, True)
            return False
        setattr(self, model_attr, model)
        setattr(self, feature_builder_attr, feature_builder)
        setattr(self, device_attr, self._resolve_model_device(device_name, torch))
        setattr(self, torch_attr, torch)
        setattr(self, checkpoint_attr, ckpt_path)
        setattr(self, ready_attr, True)
        return True
    except Exception:
        if stats_fail_key:
            self._mcts_stats[stats_fail_key] = int(self._mcts_stats.get(stats_fail_key, 0) or 0) + 1
        setattr(self, failed_attr, True)
        return False


def init_search_prior(self) -> bool:
    return _init_search_family_slot(
        self,
        enabled=self.SEARCH_PRIOR_ENABLED,
        failed_attr="_search_prior_failed",
        ready_attr="_search_prior_ready",
        model_attr="_search_prior_model",
        feature_builder_attr="_search_prior_feature_builder",
        device_attr="_search_prior_device",
        torch_attr="_search_prior_torch",
        checkpoint_attr="_search_prior_checkpoint",
        checkpoint_path=self.SEARCH_PRIOR_CHECKPOINT,
        device_name=self.SEARCH_PRIOR_DEVICE,
        stats_fail_key="search_prior_init_failed",
    )


def init_switch_prior(self) -> bool:
    return _init_search_family_slot(
        self,
        enabled=self.SWITCH_PRIOR_ENABLED,
        failed_attr="_switch_prior_failed",
        ready_attr="_switch_prior_ready",
        model_attr="_switch_prior_model",
        feature_builder_attr="_switch_prior_feature_builder",
        device_attr="_switch_prior_device",
        torch_attr="_switch_prior_torch",
        checkpoint_attr="_switch_prior_checkpoint",
        checkpoint_path=self.SWITCH_PRIOR_CHECKPOINT,
        device_name=self.SWITCH_PRIOR_DEVICE,
    )


def init_passive_breaker(self) -> bool:
    return _init_search_family_slot(
        self,
        enabled=self.PASSIVE_BREAKER_ENABLED,
        failed_attr="_passive_breaker_failed",
        ready_attr="_passive_breaker_ready",
        model_attr="_passive_breaker_model",
        feature_builder_attr="_passive_breaker_feature_builder",
        device_attr="_passive_breaker_device",
        torch_attr="_passive_breaker_torch",
        checkpoint_attr="_passive_breaker_checkpoint",
        checkpoint_path=self.PASSIVE_BREAKER_CHECKPOINT,
        device_name=self.PASSIVE_BREAKER_DEVICE,
    )


def init_tera_pruner(self) -> bool:
    return _init_search_family_slot(
        self,
        enabled=self.TERA_PRUNER_ENABLED,
        failed_attr="_tera_pruner_failed",
        ready_attr="_tera_pruner_ready",
        model_attr="_tera_pruner_model",
        feature_builder_attr="_tera_pruner_feature_builder",
        device_attr="_tera_pruner_device",
        torch_attr="_tera_pruner_torch",
        checkpoint_attr="_tera_pruner_checkpoint",
        checkpoint_path=self.TERA_PRUNER_CHECKPOINT,
        device_name=self.TERA_PRUNER_DEVICE,
    )


def init_world_ranker(self) -> bool:
    if self._world_ranker_failed:
        return False
    if self._world_ranker_ready and self._world_ranker_model is not None:
        return True
    if not self.WORLD_RANKER_ENABLED:
        return False
    try:
        import torch
        from src.models.world_ranker import WorldRankerNet

        ckpt_path = self._resolve_model_checkpoint(self.WORLD_RANKER_CHECKPOINT)
        if not ckpt_path.exists():
            self._world_ranker_failed = True
            return False

        device = self._resolve_model_device(self.WORLD_RANKER_DEVICE, torch)
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        board_dim = int(config.get("board_dim", 272))
        world_dim = int(config.get("world_dim", 101))
        hidden_dim = int(config.get("hidden_dim", 256))
        dropout = float(config.get("dropout", 0.1))

        model = WorldRankerNet(
            board_dim=board_dim,
            world_dim=world_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        self._world_ranker_model = model
        self._world_ranker_device = device
        self._world_ranker_torch = torch
        self._world_ranker_checkpoint = str(ckpt_path)
        self._world_ranker_ready = True
        return True
    except Exception:
        self._world_ranker_failed = True
        return False


def init_leaf_value(self) -> bool:
    if self._leaf_value_failed:
        return False
    if self._leaf_value_ready and self._leaf_value_model is not None:
        return True
    if not self.LEAF_VALUE_ENABLED:
        return False
    try:
        import torch
        from src.models.leaf_value import LeafValueNet

        ckpt_path = self._resolve_model_checkpoint(self.LEAF_VALUE_CHECKPOINT)
        if not ckpt_path.exists():
            self._leaf_value_failed = True
            return False

        device = self._resolve_model_device(self.LEAF_VALUE_DEVICE, torch)
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        board_dim = int(config.get("board_dim", 272))
        extra_dim = int(config.get("extra_dim", 10))
        hidden_dim = int(config.get("hidden_dim", 256))
        dropout = float(config.get("dropout", 0.1))

        model = LeafValueNet(
            board_dim=board_dim,
            extra_dim=extra_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        self._leaf_value_model = model
        self._leaf_value_device = device
        self._leaf_value_torch = torch
        self._leaf_value_checkpoint = str(ckpt_path)
        self._leaf_value_ready = True
        return True
    except Exception:
        self._leaf_value_failed = True
        return False


def rl_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
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


def _search_family_choice_priors(
    self,
    battle: Battle,
    choices: List[str],
    *,
    enabled: bool,
    init_fn_name: str,
    torch_attr: str,
    model_attr: str,
    feature_builder_attr: str,
    device_attr: str,
    failed_attr: str,
    stats_prefix: Optional[str] = None,
) -> Optional[List[float]]:
    if not choices or not enabled:
        return None
    if stats_prefix:
        self._mcts_stats[f"{stats_prefix}_calls"] = int(self._mcts_stats.get(f"{stats_prefix}_calls", 0) or 0) + 1
    if not getattr(self, init_fn_name)():
        return None
    torch = getattr(self, torch_attr)
    model = getattr(self, model_attr)
    feature_builder = getattr(self, feature_builder_attr)
    device = getattr(self, device_attr)
    if torch is None or model is None or feature_builder is None:
        return None
    try:
        board_features = feature_builder.build(battle)
        board_features = [
            0.0 if (not isinstance(f, (int, float)) or f != f or f > 1e6 or f < -1e6) else float(f)
            for f in board_features
        ]
        mask, move_map, switch_map = self._build_rl_action_mask_and_maps(battle)
        if not any(mask):
            if stats_prefix:
                key = f"{stats_prefix}_mask_empty"
                self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1
            return None
        action_features = self._build_search_trace_action_features(battle, mask)
        board_t = torch.tensor([board_features], dtype=torch.float32, device=device)
        action_t = torch.tensor([action_features], dtype=torch.float32, device=device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, _ = model(board_t, action_t, mask_t)
            logits = torch.clamp(logits, min=-1e8, max=1e8)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        probs_arr = probs[0].detach().cpu().tolist()

        priors: List[float] = []
        unmapped = 0
        for choice in choices:
            idx = self._choice_to_rl_action_idx(choice, mask, move_map, switch_map)
            if idx is None or idx >= len(probs_arr):
                unmapped += 1
                priors.append(0.0)
            else:
                priors.append(max(0.0, float(probs_arr[idx])))
        if stats_prefix and unmapped > 0:
            key = f"{stats_prefix}_unmapped_choices"
            self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + int(unmapped)
        if sum(priors) <= 0:
            if stats_prefix:
                key = f"{stats_prefix}_zero_sum"
                self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1
            return None
        if stats_prefix:
            key = f"{stats_prefix}_used"
            self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1
        return priors
    except Exception:
        if stats_prefix:
            key = f"{stats_prefix}_apply_failed"
            self._mcts_stats[key] = int(self._mcts_stats.get(key, 0) or 0) + 1
        setattr(self, failed_attr, True)
        return None


def search_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
    return _search_family_choice_priors(
        self,
        battle,
        choices,
        enabled=self.SEARCH_PRIOR_ENABLED,
        init_fn_name="_init_search_prior",
        torch_attr="_search_prior_torch",
        model_attr="_search_prior_model",
        feature_builder_attr="_search_prior_feature_builder",
        device_attr="_search_prior_device",
        failed_attr="_search_prior_failed",
        stats_prefix="search_prior",
    )


def switch_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
    return _search_family_choice_priors(
        self,
        battle,
        choices,
        enabled=self.SWITCH_PRIOR_ENABLED,
        init_fn_name="_init_switch_prior",
        torch_attr="_switch_prior_torch",
        model_attr="_switch_prior_model",
        feature_builder_attr="_switch_prior_feature_builder",
        device_attr="_switch_prior_device",
        failed_attr="_switch_prior_failed",
    )


def passive_break_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
    return _search_family_choice_priors(
        self,
        battle,
        choices,
        enabled=self.PASSIVE_BREAKER_ENABLED,
        init_fn_name="_init_passive_breaker",
        torch_attr="_passive_breaker_torch",
        model_attr="_passive_breaker_model",
        feature_builder_attr="_passive_breaker_feature_builder",
        device_attr="_passive_breaker_device",
        failed_attr="_passive_breaker_failed",
    )


def tera_choice_priors(self, battle: Battle, choices: List[str]) -> Optional[List[float]]:
    return _search_family_choice_priors(
        self,
        battle,
        choices,
        enabled=self.TERA_PRUNER_ENABLED,
        init_fn_name="_init_tera_pruner",
        torch_attr="_tera_pruner_torch",
        model_attr="_tera_pruner_model",
        feature_builder_attr="_tera_pruner_feature_builder",
        device_attr="_tera_pruner_device",
        failed_attr="_tera_pruner_failed",
    )


def predict_leaf_value(self, battle: Battle) -> Optional[float]:
    if not self.LEAF_VALUE_ENABLED:
        return None
    if not self._init_leaf_value():
        return None
    feature_builder = self._init_search_trace_builder()
    torch = self._leaf_value_torch
    model = self._leaf_value_model
    if feature_builder is None or torch is None or model is None:
        return None
    try:
        board_features = feature_builder.build(battle)
        board_features = [
            0.0 if (not isinstance(f, (int, float)) or f != f or f > 1e6 or f < -1e6) else float(f)
            for f in board_features
        ]
        extra_features = self._build_state_value_features(battle)
        board_t = torch.tensor([board_features], dtype=torch.float32, device=self._leaf_value_device)
        extra_t = torch.tensor([extra_features], dtype=torch.float32, device=self._leaf_value_device)
        with torch.no_grad():
            pred = float(model(board_t, extra_t)[0].detach().cpu().item())
        if not math.isfinite(pred):
            return None
        pred = max(-1.0, min(1.0, pred))
        self._mcts_stats["leaf_value_used"] += 1
        self._mcts_stats["leaf_value_pred_sum"] += pred
        self._mcts_stats["leaf_value_pred_count"] += 1
        return pred
    except Exception:
        self._leaf_value_failed = True
        return None

