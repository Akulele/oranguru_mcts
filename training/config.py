"""
🦧 ORANGURU RL - Configuration

Hyperparameters tuned to prevent collapse.
"""

from dataclasses import dataclass


@dataclass
class RLConfig:
    """RL Training Configuration - tuned to prevent collapse"""
    
    # Model - MUCH LARGER for proper feature learning
    d_model: int = 512  # 2.7x larger (was 192)
    n_heads: int = 8
    n_layers: int = 6   # Deeper network
    n_actions: int = 13
    feature_dim: int = 272  # Updated: added item features
    rnn_hidden: int = 256
    rnn_layers: int = 1
    prediction_features_enabled: bool = False
    
    # PPO - Improved for better learning
    lr: float = 1e-3  # Higher learning rate for faster convergence
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05  # Good exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training - Larger batches for stability
    batch_size: int = 64  # Larger batches
    n_epochs: int = 4  # PPO epochs
    rollout_length: int = 512  # Longer rollouts for better returns
    total_timesteps: int = 1_000_000

    # Offline RL / BC
    offline_batch_size: int = 16
    offline_bc_epochs: int = 80
    offline_rl_epochs: int = 40
    offline_focus_epochs: int = 20
    offline_awbc_beta: float = 1.0
    offline_awbc_max_weight: float = 20.0
    offline_value_coef: float = 0.5
    offline_entropy_coef: float = 0.01
    offline_focus_min_remaining: int = 5
    offline_selfplay_battles: int = 1200
    offline_selfplay_epochs: int = 12
    offline_selfplay_weight_win: float = 1.5
    offline_selfplay_weight_dom: float = 2.0
    offline_selfplay_weight_loss: float = 0.0
    offline_selfplay_min_winrate: float = 0.55
    offline_selfplay_wins_only: bool = False
    offline_selfplay_min_turns: int = 8
    offline_selfplay_min_return: float = 0.0
    offline_selfplay_mode: str = "league"  # "league" or "heuristics"
    offline_selfplay_checkpoints: tuple = ("imitation.pt", "offline_rl.pt", "offline_focus.pt")
    offline_selfplay_include_current: bool = True
    offline_selfplay_include_rulebot: bool = True
    offline_selfplay_include_heuristics: bool = False
    offline_selfplay_include_maxpower: bool = False
    offline_selfplay_round_robin: bool = True

    # Debugging / diagnostics
    debug_eval_enabled: bool = True
    debug_eval_battles: int = 500
    debug_eval_opponents: tuple = ("heuristics",)
    debug_eval_track_illegal: bool = True
    debug_selfplay_stats: bool = True
    debug_log_losses: bool = True

    # Penalize probability mass on illegal actions
    illegal_action_coef: float = 0.2
    # Encourage non-zero switch probability when switches are legal
    switch_mass_target: float = 0.45
    switch_mass_coef: float = 0.60
    switch_mass_bad_matchup_only: bool = True
    switch_mass_bad_matchup_threshold: float = -0.2

    # Matchup-aware switch bias
    switch_bias_enabled: bool = True
    switch_bias_matchup_threshold: float = -0.3
    switch_bias_strength: float = 0.90
    switch_bias_require_good_switch: bool = True
    switch_bias_good_switch_delta: float = 0.3
    # Boost BC loss weight for switch actions to reduce stay-in bias
    bc_switch_action_weight: float = 3.0
    # Penalize low-effectiveness attacks in bad matchups
    attack_eff_penalty_enabled: bool = True
    attack_eff_bad_matchup_only: bool = True
    attack_eff_bad_matchup_threshold: float = -0.2
    attack_eff_low_threshold: float = 0.5
    attack_eff_low_penalty: float = 0.15
    attack_eff_immune_penalty: float = 0.3
    attack_eff_require_switch: bool = True
    attack_eff_require_good_switch: bool = True
    attack_eff_good_switch_delta: float = 0.3

    # Experience replay to prevent forgetting
    replay_buffer_size: int = 10000  # Store past transitions
    replay_sample_ratio: float = 0.25  # Mix 25% old experiences
    
    # Self-play - SLOWER opponent updates
    opponent_update_freq: int = 50  # Much slower (was 10)
    
    # Curriculum - Balanced requirements for steady progression
    curriculum_stages: tuple = (
        # (opponent, min_winrate, min_battles, max_steps)
        # Reasonable targets that ensure learning without being too strict
        ("random", 0.65, 200, 150_000),      # Need 65% over 200 battles
        ("max_power", 0.48, 350, 600_000),   # Need 48% over 350 battles (more achievable)
        ("heuristics", 0.35, 400, 1_000_000), # Need 35% over 400 battles
        ("specialist", 0.0, 0, float('inf')),
    )

    # Specialist pool - mix opponents to avoid overfitting
    # Format: (name, weight)
    specialist_pool: tuple = (
        ("rule_bot", 0.35),
        ("smart_heuristics", 0.25),
        ("heuristics", 0.20),
        ("max_power", 0.10),
        ("random", 0.05),
        ("self_play", 0.05),
    )
    
    # Rewards - TERMINAL-DOMINANT (sparse > dense)
    # Terminal rewards are 10x larger than any intermediate signal
    reward_win: float = 10.0     # Dominant terminal reward
    reward_lose: float = -10.0   # Dominant terminal penalty
    reward_ko: float = 0.5       # KO bonus - still meaningful but can't overwhelm win/lose
    reward_faint: float = -0.5   # Faint penalty
    reward_damage: float = 0.0   # DISABLED - encourages MaxPower behavior
    reward_hp_cap: float = 0.0   # DISABLED

    # Matchup-based rewards (from RuleBot logic)
    reward_good_switch: float = 0.3   # Switching to advantageous matchup
    reward_bad_switch: float = -0.2   # Switching to disadvantageous matchup
    reward_status_land: float = 0.4   # Successfully applying status (burn/para/sleep)
    reward_hazard_set: float = 0.3    # Setting entry hazards
    
    # Monitoring
    log_freq: int = 10
    eval_freq: int = 2000
    save_freq: int = 5000
    
    # Anti-collapse
    min_entropy: float = 0.3  # If entropy drops below this, increase entropy_coef
    max_policy_loss: float = 1.0  # If policy loss exceeds this, skip update
    
    # RuleBot data collection
    rulebot_battles_per_opponent: int = 3500
    rulebot_min_trajectories: int = 15000
    rulebot_collect_random: bool = True
    rulebot_collect_maxpower: bool = True
    rulebot_collect_heuristics: bool = True
    rulebot_collect_wins_only: bool = True
    rulebot_heuristics_multiplier: float = 4.5
    rulebot_wins_multiplier: float = 8.0
    rulebot_weight_random: float = 0.2
    rulebot_weight_maxpower: float = 0.4
    rulebot_weight_heuristics: float = 6.0
    rulebot_weight_wins: float = 10.0

    # BC focus on heuristics-only data
    bc_focus_epochs: int = 90
    bc_focus_lr: float = 5e-5
    bc_log_interval: int = 5
    bc_chunk_log_interval: int = 10
    bc_focus_tags: tuple = ("rulebot_wins",)
    bc_focus_eval_interval: int = 10
    bc_focus_mix_tags: tuple = ("rulebot_wins", "rulebot_vs_heuristics")
    bc_focus_mix_ratio: float = 0.9

    # BC focus on switch-only actions (disabled; corrections merged upstream)
    bc_switch_focus_epochs: int = 0
    bc_switch_focus_lr: float = 5e-5
    bc_switch_focus_weight: float = 2.5
    bc_switch_focus_min_actions: int = 1
    bc_switch_focus_matchup_delta_min: float = 0.2
    bc_switch_focus_skip_forced: bool = True
    bc_switch_focus_tags: tuple = ("heuristics", "rulebot_wins")
    bc_switch_focus_bad_delta_max: float = -0.2
    bc_switch_focus_bad_weight: float = 0.6
    bc_switch_focus_value_coef: float = 0.0

    # Matchup-aware stay penalty during BC
    bc_matchup_switch_penalty: bool = True
    bc_matchup_threshold: float = -0.3
    bc_matchup_penalty_coef: float = 0.4

    # RAM safety
    ram_min_available_gb: float = 3.0
    ram_trim_fraction: float = 0.7
    ram_trim_keep_min: int = 2000
    ram_priority_tags: tuple = ("heuristics", "rulebot_wins")
    ram_pause_available_gb: float = 2.5
    ram_pause_check_seconds: float = 5.0

    # Resume
    resume_checkpoint: str | None = None
    resume_phase_override: str | None = None  # "bc", "bc_focus", "bc_switch_focus", "offline_rl", "offline_focus", "selfplay"

    # Disk streaming
    stream_trajectories: bool = True
    stream_chunk_size: int = 300
    stream_cache_dir: str = "data/trajectory_chunks"
    stream_rebuild: bool = True
    stream_keep_in_memory: bool = False

    # Paths
    checkpoint_dir: str = "checkpoints/rl"
    log_dir: str = "logs"
    replay_trajectories_path: str = "data/replay_trajectories.pkl"
    use_replay_trajectories: bool = True
    replay_weight: float = 0.25
    replay_max_trajectories: int | None = 3000
    # Ladder data (higher-quality online play)
    ladder_trajectories_path: str = "data/ladder_trajectories.pkl"
    use_ladder_trajectories: bool = True
    ladder_weight: float = 1.25
    ladder_max_trajectories: int | None = 3000
    # Switch correction data (from eval opportunity logs; merged into main pool)
    use_switch_corrections: bool = True
    switch_corrections_path: str = "data/switch_corrections.pkl"
    switch_corrections_weight: float = 1.8
    # Move correction data (from eval opportunity logs; merged into main pool)
    use_move_corrections: bool = True
    move_corrections_path: str = "data/move_corrections.pkl"
    move_corrections_weight: float = 0.25

    # Phase skips
    skip_offline_rl: bool = False
    skip_focus: bool = False
    skip_selfplay: bool = False

    # Model selection
    select_best_checkpoint: bool = True
    best_eval_target: str = "heuristics"
    best_checkpoint_path: str = "checkpoints/rl/best_overall.pt"
    rollback_if_worse: bool = True
    rollback_min_delta: float = 0.01
    # Override switch tuning (latest)
    switch_mass_coef: float = 0.60
    switch_bias_matchup_threshold: float = -0.2
    switch_bias_strength: float = 0.90
    switch_bias_require_good_switch: bool = True
    switch_bias_good_switch_delta: float = 0.3
    switch_stay_penalty_strength: float = 0.50
    bc_matchup_threshold: float = -0.2
    bc_matchup_penalty_coef: float = 1.6
    bc_switch_focus_bad_weight: float = 0.2
