import unittest
from types import SimpleNamespace

from poke_env.battle import MoveCategory, PokemonType

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, category=None, base_power=0, accuracy=100):
        self.id = move_id
        self.category = category
        self.base_power = base_power
        self.accuracy = accuracy
        self.damage = None
        self.boosts = {}
        self.target = None


class DummyPokemon:
    def __init__(self, current_hp_fraction=1.0, status=None):
        self.current_hp_fraction = current_hp_fraction
        self.status = status
        self.boosts = {}
        self.fainted = current_hp_fraction <= 0.0


class DummyBattle:
    def __init__(self):
        self.force_switch = False
        self.available_moves = [DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100)]
        self.available_switches = [SimpleNamespace(species="skarmory")]
        self.active_pokemon = DummyPokemon(0.9)
        self.opponent_active_pokemon = DummyPokemon(0.8)
        self.team = {"active": self.active_pokemon}
        self.opponent_team = {"active": self.opponent_active_pokemon}
        self.opponent_side_conditions = {}
        self.turn = 1


class DummyResult:
    def __init__(self, choices):
        self.total_visits = sum(visits for _, visits in choices) or 1
        self.side_one = [
            SimpleNamespace(move_choice=choice, visits=visits)
            for choice, visits in choices
        ]


class OranguruDecisionTests(unittest.TestCase):
    def test_finish_blow_guard_prefers_damage_over_passive_choice(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_damage_score = lambda *_args: 120.0
        engine._heuristic_action_score = lambda _battle, choice: 90.0 if choice == "earthquake" else 10.0
        battle = DummyBattle()
        battle.opponent_active_pokemon = DummyPokemon(0.2)
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("protect", category=MoveCategory.STATUS),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("protect", 60.0), ("earthquake", 45.0)],
            "protect",
        )

        self.assertEqual(adjusted, "earthquake")

    def test_status_move_with_zero_damage_metadata_is_not_damaging(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        battle = DummyBattle()
        move = DummyMove("stealthrock", category=MoveCategory.STATUS, base_power=0)
        move.damage = 0
        battle.available_moves = [move]

        self.assertFalse(engine._is_damaging_move_choice(battle, "stealthrock"))

    def test_finish_blow_guard_ignores_low_policy_when_ko_exists(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_damage_score = lambda *_args: 120.0
        engine._heuristic_action_score = lambda _battle, choice: 95.0 if choice == "earthquake" else 15.0
        battle = DummyBattle()
        battle.opponent_active_pokemon = DummyPokemon(0.2)
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("roost", category=MoveCategory.STATUS),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("roost", 90.0), ("earthquake", 5.0)],
            "roost",
        )

        self.assertEqual(adjusted, "earthquake")

    def test_finish_blow_guard_prefers_safer_guaranteed_ko(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.SAFE_KO_GUARD = True
        engine.SAFE_KO_MIN_OVERKILL = 1.0
        engine.SAFE_KO_MIN_RISK_DELTA = 0.10
        engine.SAFE_KO_MIN_POLICY_RATIO = 0.02
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._move_recoil_rate = lambda _move: 0.0
        engine._get_move_entry = lambda move: {"hasCrashDamage": True} if move.id == "highjumpkick" else {}
        engine._calculate_move_score = lambda move, *_args, **_kwargs: {
            "highjumpkick": 160.0,
            "firepunch": 95.0,
        }.get(move.id, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon = DummyPokemon(0.25)
        battle.available_moves = [
            DummyMove("highjumpkick", category=MoveCategory.PHYSICAL, base_power=130, accuracy=90),
            DummyMove("firepunch", category=MoveCategory.PHYSICAL, base_power=75, accuracy=100),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("highjumpkick", 0.70), ("firepunch", 0.12)],
            "highjumpkick",
        )

        self.assertEqual(adjusted, "firepunch")
        self.assertEqual(memory["finish_blow_last"]["reason"], "take_safe_ko")
        self.assertEqual(memory["finish_blow_last"]["finish_choice"], "firepunch")

    def test_finish_blow_guard_attacks_critical_hp_target_over_recovery(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.FINISH_BLOW_CRITICAL_OPP_HP = 0.10
        engine.FINISH_BLOW_THREAT_OPP_HP = 0.35
        engine.FINISH_BLOW_THREAT_BOOSTS = 2.0
        engine.FINISH_BLOW_CRITICAL_MIN_POLICY_RATIO = 0.05
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._move_recoil_rate = lambda _move: 0.0
        engine._get_move_entry = lambda _move: {}
        engine._calculate_move_score = lambda move, *_args, **_kwargs: {
            "superpower": 5.0,
            "leafstorm": 4.0,
        }.get(move.id, 0.0)
        engine._heuristic_action_score = lambda _battle, choice: {
            "superpower": 46.8,
            "leafstorm": 4.2,
            "synthesis": 0.0,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.742)
        battle.opponent_active_pokemon = DummyPokemon(0.03)
        battle.available_moves = [
            DummyMove("synthesis", category=MoveCategory.STATUS),
            DummyMove("superpower", category=MoveCategory.PHYSICAL, base_power=120),
            DummyMove("leafstorm", category=MoveCategory.SPECIAL, base_power=130),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("synthesis", 0.447), ("superpower", 0.257), ("leafstorm", 0.154)],
            "synthesis",
        )

        self.assertEqual(adjusted, "superpower")
        self.assertEqual(memory["finish_blow_last"]["reason"], "take_critical_hp_attack")

    def test_finish_blow_guard_attacks_boosted_low_target_over_switch(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.FINISH_BLOW_CRITICAL_OPP_HP = 0.10
        engine.FINISH_BLOW_THREAT_OPP_HP = 0.35
        engine.FINISH_BLOW_THREAT_BOOSTS = 2.0
        engine.FINISH_BLOW_CRITICAL_MIN_POLICY_RATIO = 0.05
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._move_recoil_rate = lambda _move: 0.0
        engine._get_move_entry = lambda _move: {}
        engine._calculate_move_score = lambda move, *_args, **_kwargs: {
            "playrough": 8.0,
            "behemothblade": 6.0,
            "closecombat": 7.0,
        }.get(move.id, 0.0)
        engine._heuristic_action_score = lambda _battle, choice: {
            "playrough": 4.8,
            "behemothblade": 3.0,
            "closecombat": 1.4,
            "switch gholdengo": 1.0,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.191)
        battle.opponent_active_pokemon = DummyPokemon(0.28)
        battle.opponent_active_pokemon.boosts = {
            "attack": 2,
            "special-attack": 2,
            "speed": 2,
        }
        battle.available_moves = [
            DummyMove("playrough", category=MoveCategory.PHYSICAL, base_power=90),
            DummyMove("behemothblade", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("closecombat", category=MoveCategory.PHYSICAL, base_power=120),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [
                ("playrough", 0.346),
                ("behemothblade", 0.198),
                ("closecombat", 0.194),
                ("switch gholdengo", 0.036),
            ],
            "switch gholdengo",
        )

        self.assertEqual(adjusted, "playrough")
        self.assertEqual(memory["finish_blow_last"]["reason"], "take_boosted_threat_attack")

    def test_finish_blow_guard_preserves_strong_protect_over_low_policy_ko(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.FINISH_BLOW_CRITICAL_OPP_HP = 0.10
        engine.FINISH_BLOW_THREAT_OPP_HP = 0.35
        engine.FINISH_BLOW_THREAT_BOOSTS = 2.0
        engine.FINISH_BLOW_CRITICAL_MIN_POLICY_RATIO = 0.05
        engine.FINISH_BLOW_PASSIVE_MAX_SCORE_DROP = 0.28
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._estimate_best_damage_score = lambda *_args: 60.0
        engine._calculate_move_score = lambda move, *_args, **_kwargs: 60.0 if move.id == "aurawheel" else 0.0
        engine._heuristic_action_score = lambda _battle, choice: {
            "protect": 0.0,
            "aurawheel": 8.6,
            "rapidspin": 1.3,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.184)
        battle.opponent_active_pokemon = DummyPokemon(0.20)
        battle.available_moves = [
            DummyMove("protect", category=MoveCategory.STATUS),
            DummyMove("aurawheel", category=MoveCategory.PHYSICAL, base_power=110),
            DummyMove("rapidspin", category=MoveCategory.PHYSICAL, base_power=50),
        ]

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("protect", 0.4656), ("rapidspin", 0.1907), ("aurawheel", 0.1819)],
            "protect",
        )

        self.assertEqual(adjusted, "protect")
        self.assertEqual(memory["finish_blow_last"]["reason"], "passive_policy_drop")

    def test_fatal_reply_guard_switches_when_attack_does_not_ko(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.FATAL_REPLY_GUARD_ENABLED = True
        engine.FATAL_REPLY_KO_THRESHOLD = 185.0
        engine.FATAL_REPLY_MIN_REPLY = 45.0
        engine.FATAL_REPLY_MIN_POLICY_RATIO = 0.10
        engine.FATAL_REPLY_MIN_SWITCH_SCORE = 0.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.PROTECT_MOVES = set()
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._calculate_move_score = lambda move, *_args, **_kwargs: 150.0 if move.id == "fireblast" else 0.0
        engine._estimate_best_reply_score = lambda *_args: 240.0
        engine._switch_faints_to_entry_hazards = lambda *_args: False
        engine._score_switch = lambda sw, *_args: 85.0 if sw.species == "screamtail" else 10.0
        engine._is_recovery_move = lambda _move: False
        engine._should_use_protect = lambda *_args: False
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(1.0)
        battle.opponent_active_pokemon = DummyPokemon(0.8)
        battle.available_moves = [DummyMove("fireblast", category=MoveCategory.SPECIAL, base_power=110)]
        battle.available_switches = [SimpleNamespace(species="screamtail")]

        adjusted = engine._maybe_avoid_fatal_reply_choice(
            battle,
            [("fireblast", 0.70), ("switch screamtail", 0.12)],
            "fireblast",
        )

        self.assertEqual(adjusted, "switch screamtail")
        self.assertEqual(memory["fatal_reply_last"]["reason"], "avoid_fatal_reply")

    def test_fatal_reply_guard_attacks_over_low_hp_setup(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.FATAL_REPLY_GUARD_ENABLED = True
        engine.FATAL_REPLY_KO_THRESHOLD = 185.0
        engine.FATAL_REPLY_MIN_REPLY = 45.0
        engine.FATAL_REPLY_MIN_POLICY_RATIO = 0.10
        engine.FATAL_REPLY_MIN_SWITCH_SCORE = 0.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.PROTECT_MOVES = set()
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._calculate_move_score = lambda move, *_args, **_kwargs: {
            "focusblast": 80.0,
            "thunderbolt": 70.0,
        }.get(move.id, 0.0)
        engine._estimate_best_reply_score = lambda *_args: 80.0
        engine._switch_faints_to_entry_hazards = lambda *_args: False
        engine._score_switch = lambda *_args: -10.0
        engine._is_recovery_move = lambda _move: False
        engine._should_use_protect = lambda *_args: False
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.054)
        battle.opponent_active_pokemon = DummyPokemon(1.0)
        setup = DummyMove("nastyplot", category=MoveCategory.STATUS)
        setup.boosts = {"spa": 2}
        setup.target = "self"
        battle.available_moves = [
            DummyMove("focusblast", category=MoveCategory.SPECIAL, base_power=120),
            DummyMove("thunderbolt", category=MoveCategory.SPECIAL, base_power=90),
            setup,
        ]
        battle.available_switches = []

        adjusted = engine._maybe_avoid_fatal_reply_choice(
            battle,
            [("focusblast", 0.29), ("thunderbolt", 0.25), ("nastyplot", 0.23)],
            "nastyplot",
        )

        self.assertEqual(adjusted, "focusblast")
        self.assertEqual(memory["fatal_reply_last"]["safe_kind"], "attack")

    def test_fatal_reply_guard_blocks_setup_into_strong_reply(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.FATAL_REPLY_GUARD_ENABLED = True
        engine.FATAL_REPLY_KO_THRESHOLD = 185.0
        engine.FATAL_REPLY_MIN_REPLY = 45.0
        engine.FATAL_REPLY_MIN_POLICY_RATIO = 0.10
        engine.FATAL_REPLY_MIN_SWITCH_SCORE = 0.0
        engine.SETUP_FATAL_REPLY_MIN_REPLY = 120.0
        engine.SETUP_FATAL_REPLY_MAX_HP = 0.90
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.PROTECT_MOVES = set()
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._calculate_move_score = lambda move, *_args, **_kwargs: 110.0 if move.id == "earthquake" else 0.0
        engine._estimate_best_reply_score = lambda *_args: 130.0
        engine._switch_faints_to_entry_hazards = lambda *_args: False
        engine._score_switch = lambda *_args: -10.0
        engine._is_recovery_move = lambda _move: False
        engine._should_use_protect = lambda *_args: False
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.80)
        battle.opponent_active_pokemon = DummyPokemon(0.70)
        setup = DummyMove("swordsdance", category=MoveCategory.STATUS)
        setup.boosts = {"atk": 2}
        setup.target = "self"
        battle.available_moves = [
            setup,
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
        ]
        battle.available_switches = []

        adjusted = engine._maybe_avoid_fatal_reply_choice(
            battle,
            [("swordsdance", 0.60), ("earthquake", 0.12)],
            "swordsdance",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(memory["fatal_reply_last"]["reason"], "avoid_fatal_reply")
        self.assertTrue(memory["fatal_reply_last"]["setup_reply_danger"])

    def test_anti_sweeper_guard_uses_trick_room_into_shell_smash(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.ANTI_SWEEPER_CONTROL_GUARD_ENABLED = True
        engine.ANTI_SWEEPER_CONTROL_MIN_TURN = 8
        engine.ANTI_SWEEPER_CONTROL_MIN_BOOST_PRESSURE = 2.0
        engine.ANTI_SWEEPER_CONTROL_HIGH_PRESSURE = 5.0
        engine.ANTI_SWEEPER_CONTROL_MAX_MY_ALIVE = 3
        engine.ANTI_SWEEPER_CONTROL_MAX_OPP_ALIVE = 3
        engine.ANTI_SWEEPER_CONTROL_MIN_POLICY_RATIO = 0.35
        engine.ANTI_SWEEPER_CONTROL_MAX_SCORE_DROP = 0.35
        engine.STATUS_MOVES = OranguruEnginePlayer.STATUS_MOVES
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._get_effective_speed = lambda mon: 50 if mon is battle.active_pokemon else 400
        engine._get_move_entry = lambda _move: {}
        engine._status_from_move_entry = lambda _entry: None
        engine._status_choice_is_obviously_bad = lambda *_args: False
        engine._sleep_clause_blocked = lambda _battle: False
        engine._move_inflicts_sleep = lambda _move: False
        engine._heuristic_action_score = lambda _battle, choice: 0.0 if choice == "trickroom" else 4.0
        battle = DummyBattle()
        battle.turn = 22
        battle.active_pokemon = DummyPokemon(1.0)
        battle.opponent_active_pokemon = DummyPokemon(0.18)
        battle.opponent_active_pokemon.boosts = {
            "attack": 2,
            "special-attack": 2,
            "speed": 2,
        }
        battle.team = {
            "active": battle.active_pokemon,
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
        }
        battle.opponent_team = {
            "active": battle.opponent_active_pokemon,
            "bench": DummyPokemon(1.0),
        }
        battle.available_moves = [
            DummyMove("trickroom", category=MoveCategory.STATUS),
            DummyMove("psychic", category=MoveCategory.SPECIAL, base_power=90),
            DummyMove("bugbuzz", category=MoveCategory.SPECIAL, base_power=90),
        ]

        adjusted = engine._maybe_take_anti_sweeper_control_choice(
            battle,
            [("trickroom", 0.4805), ("psychic", 0.1312), ("bugbuzz", 0.1308)],
            "psychic",
        )

        self.assertEqual(adjusted, "trickroom")
        self.assertEqual(memory["anti_sweeper_last"]["reason"], "take_anti_sweeper_control")
        self.assertEqual(memory["anti_sweeper_last"]["control_kind"], "trickroom")

    def test_anti_sweeper_guard_phazes_boosted_multi_mon_threat(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.ANTI_SWEEPER_CONTROL_GUARD_ENABLED = True
        engine.ANTI_SWEEPER_CONTROL_MIN_TURN = 8
        engine.ANTI_SWEEPER_CONTROL_MIN_BOOST_PRESSURE = 2.0
        engine.ANTI_SWEEPER_CONTROL_HIGH_PRESSURE = 5.0
        engine.ANTI_SWEEPER_CONTROL_MAX_MY_ALIVE = 3
        engine.ANTI_SWEEPER_CONTROL_MAX_OPP_ALIVE = 3
        engine.ANTI_SWEEPER_CONTROL_MIN_POLICY_RATIO = 0.35
        engine.ANTI_SWEEPER_CONTROL_MAX_SCORE_DROP = 0.35
        engine.STATUS_MOVES = OranguruEnginePlayer.STATUS_MOVES
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._get_effective_speed = lambda _mon: 100
        engine._get_move_entry = lambda _move: {}
        engine._status_from_move_entry = lambda _entry: None
        engine._status_choice_is_obviously_bad = lambda *_args: False
        engine._sleep_clause_blocked = lambda _battle: False
        engine._move_inflicts_sleep = lambda _move: False
        engine._heuristic_action_score = lambda _battle, choice: 3.0 if choice == "dragontail" else 1.0
        battle = DummyBattle()
        battle.turn = 21
        battle.active_pokemon = DummyPokemon(0.30)
        battle.opponent_active_pokemon = DummyPokemon(1.0)
        battle.opponent_active_pokemon.boosts = {
            "defense": 4,
            "special-attack": 2,
            "special-defense": 2,
            "speed": 4,
        }
        battle.team = {
            "active": battle.active_pokemon,
            "bench": DummyPokemon(1.0),
        }
        battle.opponent_team = {
            "active": battle.opponent_active_pokemon,
            "bench": DummyPokemon(1.0),
        }
        battle.available_moves = [
            DummyMove("dragontail", category=MoveCategory.PHYSICAL, base_power=60),
            DummyMove("flashcannon", category=MoveCategory.SPECIAL, base_power=80),
        ]
        battle.available_switches = [SimpleNamespace(species="wugtrio")]

        adjusted = engine._maybe_take_anti_sweeper_control_choice(
            battle,
            [("dragontail", 0.4262), ("flashcannon", 0.4168), ("switch wugtrio", 0.1286)],
            "switch wugtrio",
        )

        self.assertEqual(adjusted, "dragontail")
        self.assertEqual(memory["anti_sweeper_last"]["reason"], "take_anti_sweeper_control")
        self.assertEqual(memory["anti_sweeper_last"]["control_kind"], "phaze")

    def test_recovery_guard_blocks_large_policy_drop_from_top_attack(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.RECOVERY_WINDOW_MAX_HP = 0.40
        engine.RECOVERY_WINDOW_MIN_OPP_HP = 0.25
        engine.RECOVERY_WINDOW_MAX_REPLY = 110.0
        engine.RECOVERY_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.RECOVERY_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.55
        engine.RECOVERY_WINDOW_CRITICAL_HP = 0.30
        engine.RECOVERY_WINDOW_CRITICAL_MIN_POLICY_RATIO = 0.33
        engine.RECOVERY_WINDOW_MIN_HEUR_GAIN = 1.0
        engine.RECOVERY_WINDOW_HIGH_HEUR_GAIN = 10.0
        engine.RECOVERY_WINDOW_MAX_SCORE_DROP = 0.28
        engine.TACTICAL_KO_THRESHOLD = 220.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._estimate_best_reply_score = lambda *_args: 40.0
        engine._estimate_best_damage_score = lambda *_args: 80.0
        engine._is_recovery_move = lambda move: move.id == "synthesis"
        engine._heuristic_action_score = lambda _battle, choice: {
            "bodypress": 0.0,
            "synthesis": 115.4,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.175)
        battle.opponent_active_pokemon = DummyPokemon(0.53)
        battle.available_moves = [
            DummyMove("bodypress", category=MoveCategory.PHYSICAL, base_power=80),
            DummyMove("synthesis", category=MoveCategory.STATUS),
        ]

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("bodypress", 0.5529), ("synthesis", 0.2081)],
            "bodypress",
        )

        self.assertEqual(adjusted, "bodypress")
        self.assertEqual(memory["recovery_window_last"]["reason"], "score_drop")

    def test_tactical_rerank_blocks_huge_top1_policy_drop(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_RERANK_MAX_TOP1_DROP = 0.35
        engine.RERANK_GATE_ENABLED = False
        engine._rerank_gate_allows = OranguruEnginePlayer._rerank_gate_allows.__get__(engine)
        engine._get_battle_memory = lambda _battle: {
            "recovery_window_last": {
                "reason": "take_recovery",
                "chosen_choice": "scald",
                "recovery_choice": "rest",
            }
        }
        battle = DummyBattle()

        adjusted = engine._maybe_accept_rerank_choice(
            battle,
            [("scald", 0.7797), ("rest", 0.0278)],
            "scald",
            "rest",
            0.7,
            0.6,
        )

        self.assertEqual(adjusted, "scald")

    def test_contact_risk_guard_prefers_close_non_contact_attack(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.CONTACT_RISK_GUARD_ENABLED = True
        engine.CONTACT_RISK_MIN_RISK = 0.12
        engine.CONTACT_RISK_MIN_DAMAGE_RATIO = 0.85
        engine.CONTACT_RISK_MIN_POLICY_RATIO = 0.30
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._move_makes_contact = OranguruEnginePlayer._move_makes_contact.__get__(engine)
        engine._contact_punish_risk = OranguruEnginePlayer._contact_punish_risk.__get__(engine)
        engine._get_ability_id = lambda _mon: "flamebody"
        engine._get_move_entry = lambda move: {
            "megahorn": {"flags": {"contact": 1}},
            "stoneedge": {"flags": {}},
        }.get(move.id, {})
        engine._calculate_move_score = lambda move, *_args, **_kwargs: {
            "megahorn": 100.0,
            "stoneedge": 92.0,
        }.get(move.id, 0.0)
        engine._heuristic_action_score = lambda _battle, choice: {
            "megahorn": 100.0,
            "stoneedge": 96.0,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.7)
        battle.opponent_active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon.ability = "flamebody"
        battle.available_moves = [
            DummyMove("megahorn", category=MoveCategory.PHYSICAL, base_power=120),
            DummyMove("stoneedge", category=MoveCategory.PHYSICAL, base_power=100),
        ]

        adjusted = engine._maybe_avoid_contact_risk_choice(
            battle,
            [("megahorn", 0.55), ("stoneedge", 0.28)],
            "megahorn",
        )

        self.assertEqual(adjusted, "stoneedge")
        self.assertEqual(memory["contact_risk_last"]["reason"], "avoid_contact_risk")

    def test_tera_sanity_rejects_new_stab_weakness_without_ko(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TERA_DEFENSIVE_SANITY_ENABLED = True
        engine.TACTICAL_KO_THRESHOLD = 220.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._calculate_move_score = lambda *_args, **_kwargs: 40.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)
        battle.active_pokemon.types = [PokemonType.PSYCHIC]
        battle.active_pokemon.tera_type = PokemonType.FIGHTING
        battle.opponent_active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon.types = [PokemonType.DARK, PokemonType.FLYING]
        move = DummyMove("psychicfangs", category=MoveCategory.PHYSICAL, base_power=85)

        self.assertTrue(engine._tera_defensive_sanity_reject(battle, move))
        self.assertEqual(memory["tera_sanity_last"]["stab_type"], "flying")

    def test_setup_window_guard_prefers_setup_when_safe(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SETUP_WINDOW_MIN_HP = 0.65
        engine.SETUP_WINDOW_MAX_REPLY = 110.0
        engine.SETUP_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.SETUP_WINDOW_MIN_HEUR_GAIN = 15.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_reply_score = lambda *_args: 40.0
        engine._estimate_best_damage_score = lambda *_args: 60.0
        engine._heuristic_action_score = lambda _battle, choice: 90.0 if choice == "calmmind" else 40.0
        engine._should_setup_move = lambda move, active, opponent: move.id == "calmmind"
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon = DummyPokemon(0.8)
        battle.available_moves = [
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
        ]
        battle.available_moves[0].boosts = {"spa": 1}
        battle.available_moves[0].target = "self"

        adjusted = engine._maybe_take_setup_window_choice(
            battle,
            [("earthquake", 60.0), ("calmmind", 45.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "calmmind")

    def test_negative_matchup_switch_guard_prefers_nearby_damage(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._heuristic_action_score = lambda _battle, choice: 25.0 if choice.startswith("switch ") else 80.0
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 60.0 if choice.startswith("switch ") else 0.0
        battle = DummyBattle()

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 60.0), ("earthquake", 55.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "earthquake")

    def test_lategame_attack_guard_commits_direct_damage_over_switch(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROTECT_MOVES = set()
        engine.LATEGAME_ATTACK_MIN_TURN = 12
        engine.LATEGAME_ATTACK_ALLOW_HIDDEN = False
        engine.LATEGAME_ATTACK_MIN_BASE_POWER = 50.0
        engine.LATEGAME_ATTACK_MIN_HEURISTIC = 0.75
        engine.LATEGAME_ATTACK_MIN_POLICY_RATIO = 0.05
        engine.LATEGAME_ATTACK_MAX_SCORE_DROP = 0.40
        engine.LATEGAME_ATTACK_MAX_RISK = 35.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_recovery_move = lambda _move: False
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._adaptive_choice_risk_penalty = lambda *_args: 0.0
        engine._heuristic_action_score = lambda _battle, choice: {
            "switch talonflame": 0.0,
            "uturn": 18.0,
            "nuzzle": 7.0,
            "thunderbolt": 2.1,
            "dazzlinggleam": 1.8,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.turn = 53
        battle.active_pokemon = DummyPokemon(0.6)
        battle.opponent_active_pokemon = DummyPokemon(0.74)
        battle.team = {
            "ded": battle.active_pokemon,
            "talonflame": DummyPokemon(0.35),
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
            "fainted4": DummyPokemon(0.0),
        }
        battle.opponent_team = {
            "chansey": battle.opponent_active_pokemon,
            "alomomola": DummyPokemon(1.0),
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
            "fainted4": DummyPokemon(0.0),
        }
        battle.available_moves = [
            DummyMove("uturn", category=MoveCategory.PHYSICAL, base_power=70),
            DummyMove("nuzzle", category=MoveCategory.PHYSICAL, base_power=20),
            DummyMove("thunderbolt", category=MoveCategory.SPECIAL, base_power=90),
            DummyMove("dazzlinggleam", category=MoveCategory.SPECIAL, base_power=80),
        ]

        adjusted = engine._maybe_commit_late_game_attack_choice(
            battle,
            [
                ("switch talonflame", 0.3589),
                ("uturn", 0.3420),
                ("nuzzle", 0.2396),
                ("thunderbolt", 0.0354),
                ("dazzlinggleam", 0.0241),
            ],
            "switch talonflame",
        )

        self.assertEqual(adjusted, "thunderbolt")
        self.assertEqual(memory["late_game_attack_guard_last"]["reason"], "take_late_attack")

    def test_lategame_attack_guard_prefers_damage_selected_attack_over_raw_heuristic(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROTECT_MOVES = set()
        engine.LATEGAME_ATTACK_MIN_TURN = 12
        engine.LATEGAME_ATTACK_ALLOW_HIDDEN = False
        engine.LATEGAME_ATTACK_MIN_BASE_POWER = 50.0
        engine.LATEGAME_ATTACK_MIN_HEURISTIC = 0.75
        engine.LATEGAME_ATTACK_MIN_POLICY_RATIO = 0.05
        engine.LATEGAME_ATTACK_MAX_SCORE_DROP = 0.40
        engine.LATEGAME_ATTACK_MAX_RISK = 35.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_recovery_move = lambda move: move.id == "rest"
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._adaptive_choice_risk_penalty = lambda *_args: 0.0
        engine._heuristic_action_score = lambda _battle, choice: {
            "rest": 115.1,
            "psychocut": 2.945,
            "knockoff": 2.735,
            "superpower": 2.726,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.turn = 36
        battle.active_pokemon = DummyPokemon(0.178)
        battle.active_pokemon.boosts = {"attack": 2, "defense": 2}
        battle.opponent_active_pokemon = DummyPokemon(1.0)
        battle.team = {
            "malamar": battle.active_pokemon,
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
            "fainted4": DummyPokemon(0.0),
            "fainted5": DummyPokemon(0.0),
        }
        battle.opponent_team = {
            "arceusgrass": battle.opponent_active_pokemon,
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
            "fainted4": DummyPokemon(0.0),
            "fainted5": DummyPokemon(0.0),
        }
        knockoff = DummyMove("knockoff", category=MoveCategory.PHYSICAL, base_power=65)
        battle.available_moves = [
            DummyMove("superpower", category=MoveCategory.PHYSICAL, base_power=120),
            knockoff,
            DummyMove("psychocut", category=MoveCategory.PHYSICAL, base_power=70),
            DummyMove("rest", category=MoveCategory.STATUS, base_power=0),
        ]
        engine._best_damaging_move = lambda *_args: (knockoff, 100.0)

        adjusted = engine._maybe_commit_late_game_attack_choice(
            battle,
            [
                ("rest", 0.3258),
                ("psychocut", 0.2296),
                ("knockoff", 0.2258),
                ("superpower", 0.2189),
            ],
            "rest",
        )

        self.assertEqual(adjusted, "knockoff")
        self.assertEqual(memory["late_game_attack_guard_last"]["reason"], "take_late_attack")
        self.assertEqual(memory["late_game_attack_guard_last"]["attack_damage_score"], 100.0)

    def test_lategame_attack_guard_does_not_force_attack_when_materially_behind(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROTECT_MOVES = set()
        engine.LATEGAME_ATTACK_MIN_TURN = 12
        engine.LATEGAME_ATTACK_ALLOW_HIDDEN = False
        engine.LATEGAME_ATTACK_MIN_BASE_POWER = 50.0
        engine.LATEGAME_ATTACK_MIN_HEURISTIC = 0.75
        engine.LATEGAME_ATTACK_MIN_POLICY_RATIO = 0.05
        engine.LATEGAME_ATTACK_MAX_SCORE_DROP = 0.40
        engine.LATEGAME_ATTACK_MAX_RISK = 35.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_recovery_move = lambda move: move.id == "moonlight"
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._adaptive_choice_risk_penalty = lambda *_args: 0.0
        engine._heuristic_action_score = lambda _battle, choice: {
            "moonlight": 66.0,
            "calmmind": 67.5,
            "photongeyser": 4.0,
            "earthpower": 0.0,
        }.get(choice, 0.0)
        battle = DummyBattle()
        battle.turn = 31
        battle.active_pokemon = DummyPokemon(0.528)
        battle.opponent_active_pokemon = DummyPokemon(0.42)
        battle.team = {
            "necrozma": battle.active_pokemon,
            "girafarig": DummyPokemon(0.141),
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
            "fainted4": DummyPokemon(0.0),
        }
        battle.opponent_team = {
            "landorus": battle.opponent_active_pokemon,
            "probopass": DummyPokemon(0.09),
            "amoonguss": DummyPokemon(0.45),
            "furret": DummyPokemon(0.17),
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
        }
        photongeyser = DummyMove("photongeyser", category=MoveCategory.SPECIAL, base_power=100)
        battle.available_moves = [
            photongeyser,
            DummyMove("earthpower", category=MoveCategory.SPECIAL, base_power=90),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
            DummyMove("moonlight", category=MoveCategory.STATUS, base_power=0),
        ]
        engine._best_damaging_move = lambda *_args: (photongeyser, 50.0)

        adjusted = engine._maybe_commit_late_game_attack_choice(
            battle,
            [
                ("moonlight", 0.4722),
                ("earthpower", 0.1706),
                ("photongeyser", 0.1192),
                ("calmmind", 0.0441),
            ],
            "moonlight",
        )

        self.assertEqual(adjusted, "moonlight")
        self.assertEqual(memory["late_game_attack_guard_last"]["reason"], "not_lategame")

    def test_lategame_attack_guard_ignores_hidden_opponents_by_default(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROTECT_MOVES = set()
        engine.LATEGAME_ATTACK_MIN_TURN = 12
        engine.LATEGAME_ATTACK_ALLOW_HIDDEN = False
        engine.LATEGAME_ATTACK_MIN_BASE_POWER = 50.0
        engine.LATEGAME_ATTACK_MIN_HEURISTIC = 0.75
        engine.LATEGAME_ATTACK_MIN_POLICY_RATIO = 0.05
        engine.LATEGAME_ATTACK_MAX_SCORE_DROP = 0.40
        engine.LATEGAME_ATTACK_MAX_RISK = 35.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._is_recovery_move = lambda _move: False
        engine._is_damaging_move_choice = OranguruEnginePlayer._is_damaging_move_choice.__get__(engine)
        engine._adaptive_choice_risk_penalty = lambda *_args: 0.0
        engine._heuristic_action_score = lambda _battle, choice: 10.0 if choice == "behemothblade" else 0.0
        battle = DummyBattle()
        battle.turn = 6
        battle.active_pokemon = DummyPokemon(1.0)
        battle.opponent_active_pokemon = DummyPokemon(0.37)
        battle.team = {
            "active": battle.active_pokemon,
            "ally1": DummyPokemon(1.0),
            "ally2": DummyPokemon(1.0),
            "fainted1": DummyPokemon(0.0),
            "fainted2": DummyPokemon(0.0),
            "fainted3": DummyPokemon(0.0),
        }
        battle.opponent_team = {"active": battle.opponent_active_pokemon}
        battle.available_moves = [
            DummyMove("behemothblade", category=MoveCategory.PHYSICAL, base_power=100),
        ]

        adjusted = engine._maybe_commit_late_game_attack_choice(
            battle,
            [("switch hydrapple", 0.2026), ("behemothblade", 0.2077)],
            "switch hydrapple",
        )

        self.assertEqual(adjusted, "switch hydrapple")
        self.assertEqual(memory["late_game_attack_guard_last"]["reason"], "not_lategame")

    def test_negative_matchup_switch_guard_breaks_near_tie_for_damage(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._heuristic_action_score = lambda _battle, choice: 25.0 if choice.startswith("switch ") else 30.0
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 10.0 if choice.startswith("switch ") else 0.0
        battle = DummyBattle()

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 60.0), ("earthquake", 59.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "earthquake")

    def test_negative_matchup_switch_guard_suppresses_risk_branch_by_default(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._heuristic_action_score = lambda _battle, choice: 0.0 if choice.startswith("switch ") else -0.3
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 60.0 if choice.startswith("switch ") else 0.0
        battle = DummyBattle()

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 100.0), ("earthquake", 70.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "switch skarmory")
        self.assertEqual(memory["switch_guard_last"]["reason"], "policy_or_heuristic")

    def test_negative_matchup_switch_guard_risk_branch_remains_env_tunable(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SWITCH_GUARD_RISK_MIN_RISK = 20.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._heuristic_action_score = lambda _battle, choice: 0.0 if choice.startswith("switch ") else -0.3
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 60.0 if choice.startswith("switch ") else 0.0
        battle = DummyBattle()

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 100.0), ("earthquake", 70.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(memory["switch_guard_last"]["reason"], "take_risk_attack")

    def test_negative_matchup_switch_guard_matches_live_presence_audit(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SWITCH_GUARD_MIN_ACTIVE_HP = 0.45
        engine.SWITCH_GUARD_POLICY_RATIO = 0.70
        engine.SWITCH_GUARD_HEUR_GAIN = 1.0
        engine.SWITCH_GUARD_RISK_POLICY_RATIO = 0.60
        engine.SWITCH_GUARD_RISK_MIN_RISK = 20.0
        engine.SWITCH_GUARD_RISK_HEUR_FLOOR = -0.5
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._heuristic_action_score = lambda _battle, choice: 10.0 if choice.startswith("switch ") else 12.0
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 0.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 100.0), ("earthquake", 72.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(memory["switch_guard_last"]["reason"], "take_live_attack")

    def test_negative_matchup_switch_guard_allows_low_hp_escape(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SWITCH_GUARD_MIN_ACTIVE_HP = 0.45
        engine.SWITCH_GUARD_LOW_HP_TARGET_MIN_HP = 0.35
        engine.SWITCH_GUARD_LOW_HP_MIN_HP_GAIN = 0.15
        engine.SWITCH_GUARD_LOW_HP_POLICY_RATIO = 0.45
        engine.SWITCH_GUARD_LOW_HP_HEUR_FLOOR = 0.0
        engine.SWITCH_GUARD_POLICY_RATIO = 0.70
        engine.SWITCH_GUARD_HEUR_GAIN = 1.0
        engine.SWITCH_GUARD_RISK_POLICY_RATIO = 0.60
        engine.SWITCH_GUARD_RISK_MIN_RISK = 20.0
        engine.SWITCH_GUARD_RISK_HEUR_FLOOR = -0.5
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._heuristic_action_score = lambda _battle, choice: 10.0 if choice.startswith("switch ") else 30.0
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 0.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.2)

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 100.0), ("earthquake", 90.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "switch skarmory")
        self.assertEqual(memory["switch_guard_last"]["reason"], "low_active_hp")

    def test_negative_matchup_switch_guard_rejects_low_hp_chain_switch(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SWITCH_GUARD_MIN_ACTIVE_HP = 0.45
        engine.SWITCH_GUARD_LOW_HP_TARGET_MIN_HP = 0.35
        engine.SWITCH_GUARD_LOW_HP_MIN_HP_GAIN = 0.15
        engine.SWITCH_GUARD_LOW_HP_POLICY_RATIO = 0.45
        engine.SWITCH_GUARD_LOW_HP_HEUR_FLOOR = 0.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._heuristic_action_score = lambda _battle, choice: 10.0 if choice.startswith("switch ") else 20.0
        engine._adaptive_choice_risk_penalty = lambda _battle, choice: 0.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.2)
        battle.available_switches = [SimpleNamespace(species="skarmory", current_hp_fraction=0.25)]

        adjusted = engine._maybe_reduce_negative_matchup_switch(
            battle,
            [("switch skarmory", 100.0), ("earthquake", 50.0)],
            "switch skarmory",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(memory["switch_guard_last"]["reason"], "take_low_hp_attack")
        self.assertLess(memory["switch_guard_last"]["switch_target_hp"], 0.35)

    def test_progress_window_takes_setup_when_behind_and_safe(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROGRESS_WINDOW_MIN_ACTIVE_HP = 0.50
        engine.PROGRESS_WINDOW_MIN_OPP_HP = 0.55
        engine.PROGRESS_WINDOW_MAX_REPLY = 110.0
        engine.PROGRESS_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.PROGRESS_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.45
        engine.PROGRESS_WINDOW_MIN_HEUR_GAIN = 1.0
        engine.PROGRESS_WINDOW_HIGH_HEUR_GAIN = 10.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._estimate_best_reply_score = lambda *_args: 60.0
        engine._estimate_best_damage_score = lambda *_args: 40.0
        engine._heuristic_action_score = lambda _battle, choice: 30.0 if choice == "calmmind" else 10.0
        engine._should_setup_move = lambda move, _active, _opponent: move.id == "calmmind"
        engine._is_recovery_move = lambda _move: False
        engine._status_choice_is_obviously_bad = lambda *_args: False
        engine._should_use_status_move = lambda *_args: 0.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon = DummyPokemon(0.9)
        battle.team = {"active": battle.active_pokemon}
        battle.opponent_team = {
            "active": battle.opponent_active_pokemon,
            "bench1": DummyPokemon(1.0),
        }
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        battle.available_moves[1].boosts = {"spa": 1}
        battle.available_moves[1].target = "self"

        adjusted = engine._maybe_take_progress_when_behind_choice(
            battle,
            [("earthquake", 80.0), ("calmmind", 40.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "calmmind")
        self.assertEqual(memory["progress_window_last"]["reason"], "take_progress")
        self.assertEqual(memory["progress_window_last"]["progress_kind"], "setup")

    def test_progress_window_counts_unrevealed_opponents_alive(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROGRESS_WINDOW_MIN_ACTIVE_HP = 0.50
        engine.PROGRESS_WINDOW_MIN_OPP_HP = 0.55
        engine.PROGRESS_WINDOW_MAX_REPLY = 110.0
        engine.PROGRESS_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.PROGRESS_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.45
        engine.PROGRESS_WINDOW_MIN_HEUR_GAIN = 1.0
        engine.PROGRESS_WINDOW_HIGH_HEUR_GAIN = 10.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        engine._estimate_best_reply_score = lambda *_args: 60.0
        engine._estimate_best_damage_score = lambda *_args: 40.0
        engine._heuristic_action_score = lambda _battle, choice: 30.0 if choice == "calmmind" else 10.0
        engine._should_setup_move = lambda move, _active, _opponent: move.id == "calmmind"
        engine._is_recovery_move = lambda _move: False
        engine._status_choice_is_obviously_bad = lambda *_args: False
        engine._should_use_status_move = lambda *_args: 0.0
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(0.8)
        battle.opponent_active_pokemon = DummyPokemon(0.9)
        battle.team = {
            "active": battle.active_pokemon,
            "bench1": DummyPokemon(1.0),
            "bench2": DummyPokemon(1.0),
            "bench3": DummyPokemon(1.0),
            "bench4": DummyPokemon(1.0),
        }
        battle.opponent_team = {"active": battle.opponent_active_pokemon}
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        battle.available_moves[1].boosts = {"spa": 1}
        battle.available_moves[1].target = "self"

        adjusted = engine._maybe_take_progress_when_behind_choice(
            battle,
            [("earthquake", 80.0), ("calmmind", 40.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "calmmind")
        self.assertEqual(memory["progress_window_last"]["opp_alive"], 6)

    def test_progress_window_requires_being_behind(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.PROGRESS_WINDOW_MIN_ACTIVE_HP = 0.50
        memory = {}
        engine._get_battle_memory = lambda _battle: memory
        battle = DummyBattle()
        battle.team = {
            "active": battle.active_pokemon,
            "bench1": DummyPokemon(1.0),
            "bench2": DummyPokemon(1.0),
            "bench3": DummyPokemon(1.0),
            "bench4": DummyPokemon(1.0),
            "bench5": DummyPokemon(1.0),
        }
        battle.opponent_team = {"active": battle.opponent_active_pokemon}

        adjusted = engine._maybe_take_progress_when_behind_choice(
            battle,
            [("earthquake", 80.0), ("calmmind", 80.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(memory["progress_window_last"]["reason"], "not_behind")

    def test_final_finish_pass_overrides_late_setup_rerank(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._get_battle_memory = lambda _battle: {}
        engine._diag_record_choice = lambda *_args, **_kwargs: None
        engine._append_search_trace_example = lambda *_args, **_kwargs: None
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, choice: choice
        engine._estimate_best_damage_score = lambda *_args: 120.0
        engine._heuristic_action_score = lambda _battle, choice: 100.0 if choice == "calmmind" else 90.0
        battle = DummyBattle()
        battle.opponent_active_pokemon = DummyPokemon(0.2)
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        results = [(DummyResult([("earthquake", 60.0), ("calmmind", 30.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "earthquake")

    def test_tactical_rerank_master_switch_preserves_mcts_choice(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.0
        engine.TACTICAL_RERANKS_ENABLED = False
        engine.FINISH_BLOW_GUARD_ENABLED = False
        engine.SETUP_WINDOW_ENABLED = False
        engine.RECOVERY_WINDOW_ENABLED = False
        engine.PROGRESS_WINDOW_ENABLED = False
        engine.SWITCH_GUARD_ENABLED = False
        engine.SELECTION_MODE = "blend"
        engine.GATE_MODE = "hard"
        engine.HEURISTIC_BLEND = 0.0
        engine.MIN_HEURISTIC_BLEND = 0.0
        engine.POLICY_CUTOFF = 0.75
        engine.RL_PRIOR_BLEND = 0.0
        engine.RL_PRIOR_LOWCONF_ONLY = True
        engine.SEARCH_PRIOR_BLEND = 0.0
        engine.SEARCH_PRIOR_LOWCONF_ONLY = True
        engine._apply_switch_prior_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._apply_tera_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._maybe_passive_break_choice = lambda *_args: "calmmind"
        engine._should_trigger_adaptive_fallback = lambda *_args: False
        engine._maybe_force_finish_blow_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_take_safe_recovery_choice = lambda _battle, _ordered, _choice: "recover"
        engine._maybe_take_progress_when_behind_choice = lambda _battle, _ordered, _choice: "swordsdance"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, _choice: "earthquake"
        engine._maybe_accept_rerank_choice = lambda _battle, _ordered, _current, candidate, _confidence, _threshold: candidate
        diag = {}
        engine._diag_record_choice = lambda _battle, _ordered, choice, _confidence, _threshold, path: diag.update(choice=choice, path=path)
        engine._append_search_trace_example = lambda *_args, **_kwargs: None

        battle = DummyBattle()
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        results = [(DummyResult([("earthquake", 60.0), ("calmmind", 40.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "earthquake")
        self.assertEqual(diag["choice"], "earthquake")
        self.assertEqual(diag["path"], "mcts")

    def test_passive_breaker_can_run_with_tactical_reranks_disabled(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.6
        engine.TACTICAL_RERANKS_ENABLED = False
        engine.FINISH_BLOW_GUARD_ENABLED = False
        engine.SETUP_WINDOW_ENABLED = False
        engine.RECOVERY_WINDOW_ENABLED = False
        engine.PROGRESS_WINDOW_ENABLED = False
        engine.SWITCH_GUARD_ENABLED = False
        engine.PASSIVE_BREAKER_ENABLED = True
        engine.SELECTION_MODE = "blend"
        engine.GATE_MODE = "hard"
        engine.HEURISTIC_BLEND = 0.0
        engine.MIN_HEURISTIC_BLEND = 0.0
        engine.POLICY_CUTOFF = 0.75
        engine.RL_PRIOR_BLEND = 0.0
        engine.RL_PRIOR_LOWCONF_ONLY = True
        engine.SEARCH_PRIOR_BLEND = 0.0
        engine.SEARCH_PRIOR_LOWCONF_ONLY = True
        engine._apply_switch_prior_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._apply_tera_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._maybe_passive_break_choice = lambda *_args: "earthquake"
        engine._should_trigger_adaptive_fallback = lambda *_args: False
        engine._maybe_force_finish_blow_choice = lambda _battle, _ordered, _choice: _choice
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_take_safe_recovery_choice = lambda _battle, _ordered, _choice: "recover"
        engine._maybe_take_progress_when_behind_choice = lambda _battle, _ordered, _choice: "swordsdance"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, _choice: "switch skarmory"
        engine._maybe_accept_rerank_choice = lambda _battle, _ordered, _current, candidate, _confidence, _threshold: candidate
        diag = {}
        engine._diag_record_choice = lambda _battle, _ordered, choice, _confidence, _threshold, path: diag.update(choice=choice, path=path)
        engine._append_search_trace_example = lambda *_args, **_kwargs: None

        battle = DummyBattle()
        battle.available_moves = [
            DummyMove("recover", category=MoveCategory.STATUS, base_power=0),
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
        ]
        results = [(DummyResult([("recover", 60.0), ("earthquake", 40.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "earthquake")
        self.assertEqual(diag["choice"], "earthquake")
        self.assertEqual(diag["path"], "rerank")

    def test_shadow_windows_record_without_changing_choice(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.0
        engine.TACTICAL_RERANKS_ENABLED = False
        engine.TACTICAL_SHADOW_WINDOWS_ENABLED = True
        engine.FINISH_BLOW_GUARD_ENABLED = False
        engine.SETUP_WINDOW_ENABLED = False
        engine.RECOVERY_WINDOW_ENABLED = False
        engine.PROGRESS_WINDOW_ENABLED = False
        engine.SWITCH_GUARD_ENABLED = False
        engine.PASSIVE_BREAKER_ENABLED = False
        engine.SELECTION_MODE = "blend"
        engine.GATE_MODE = "hard"
        engine.HEURISTIC_BLEND = 0.0
        engine.MIN_HEURISTIC_BLEND = 0.0
        engine.POLICY_CUTOFF = 0.75
        engine.RL_PRIOR_BLEND = 0.0
        engine.RL_PRIOR_LOWCONF_ONLY = True
        engine.SEARCH_PRIOR_BLEND = 0.0
        engine.SEARCH_PRIOR_LOWCONF_ONLY = True
        engine._apply_switch_prior_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._apply_tera_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._should_trigger_adaptive_fallback = lambda *_args: False
        engine._maybe_force_finish_blow_choice = lambda _battle, _ordered, _choice: _choice
        calls = []
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, choice: calls.append(("setup", choice)) or "calmmind"
        engine._maybe_take_safe_recovery_choice = lambda _battle, _ordered, choice: calls.append(("recovery", choice)) or "recover"
        engine._maybe_take_progress_when_behind_choice = lambda _battle, _ordered, choice: calls.append(("progress", choice)) or "swordsdance"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, choice: calls.append(("switch", choice)) or "switch skarmory"
        engine._maybe_accept_rerank_choice = lambda _battle, _ordered, _current, candidate, _confidence, _threshold: candidate
        diag = {}
        engine._diag_record_choice = lambda _battle, _ordered, choice, _confidence, _threshold, path: diag.update(choice=choice, path=path)
        engine._append_search_trace_example = lambda *_args, **_kwargs: None

        battle = DummyBattle()
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        results = [(DummyResult([("earthquake", 60.0), ("calmmind", 40.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "earthquake")
        self.assertEqual(diag["choice"], "earthquake")
        self.assertEqual(diag["path"], "mcts")
        self.assertEqual(
            calls,
            [
                ("setup", "earthquake"),
                ("recovery", "earthquake"),
                ("progress", "earthquake"),
                ("switch", "earthquake"),
            ],
        )

    def test_finish_blow_guard_can_run_with_tactical_reranks_disabled(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.0
        engine.TACTICAL_RERANKS_ENABLED = False
        engine.FINISH_BLOW_GUARD_ENABLED = True
        engine.SETUP_WINDOW_ENABLED = False
        engine.RECOVERY_WINDOW_ENABLED = False
        engine.PROGRESS_WINDOW_ENABLED = False
        engine.SWITCH_GUARD_ENABLED = False
        engine.SELECTION_MODE = "blend"
        engine.GATE_MODE = "hard"
        engine.HEURISTIC_BLEND = 0.0
        engine.MIN_HEURISTIC_BLEND = 0.0
        engine.POLICY_CUTOFF = 0.75
        engine.RL_PRIOR_BLEND = 0.0
        engine.RL_PRIOR_LOWCONF_ONLY = True
        engine.SEARCH_PRIOR_BLEND = 0.0
        engine.SEARCH_PRIOR_LOWCONF_ONLY = True
        engine._apply_switch_prior_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._apply_tera_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._maybe_passive_break_choice = lambda *_args: "calmmind"
        engine._should_trigger_adaptive_fallback = lambda *_args: False
        engine._maybe_force_finish_blow_choice = lambda _battle, _ordered, _choice: "earthquake"
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_take_safe_recovery_choice = lambda _battle, _ordered, _choice: "recover"
        engine._maybe_take_progress_when_behind_choice = lambda _battle, _ordered, _choice: "swordsdance"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, _choice: "switch skarmory"
        engine._maybe_accept_rerank_choice = lambda _battle, _ordered, _current, candidate, _confidence, _threshold: candidate
        diag = {}
        engine._diag_record_choice = lambda _battle, _ordered, choice, _confidence, _threshold, path: diag.update(choice=choice, path=path)
        engine._append_search_trace_example = lambda *_args, **_kwargs: None

        battle = DummyBattle()
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("roost", category=MoveCategory.STATUS, base_power=0),
        ]
        results = [(DummyResult([("roost", 60.0), ("earthquake", 40.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "earthquake")
        self.assertEqual(diag["choice"], "earthquake")
        self.assertEqual(diag["path"], "rerank")

    def test_setup_window_can_run_without_other_tactical_reranks(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine._mcts_stats = {"deterministic_decisions": 0, "stochastic_decisions": 0}
        engine.MCTS_DETERMINISTIC = True
        engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        engine.MCTS_CONFIDENCE_THRESHOLD = 0.0
        engine.TACTICAL_RERANKS_ENABLED = False
        engine.FINISH_BLOW_GUARD_ENABLED = False
        engine.SETUP_WINDOW_ENABLED = True
        engine.RECOVERY_WINDOW_ENABLED = False
        engine.PROGRESS_WINDOW_ENABLED = False
        engine.SWITCH_GUARD_ENABLED = False
        engine.SELECTION_MODE = "blend"
        engine.GATE_MODE = "hard"
        engine.HEURISTIC_BLEND = 0.0
        engine.MIN_HEURISTIC_BLEND = 0.0
        engine.POLICY_CUTOFF = 0.75
        engine.RL_PRIOR_BLEND = 0.0
        engine.RL_PRIOR_LOWCONF_ONLY = True
        engine.SEARCH_PRIOR_BLEND = 0.0
        engine.SEARCH_PRIOR_LOWCONF_ONLY = True
        engine._apply_switch_prior_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._apply_tera_prune = lambda _battle, filtered, _confidence, _threshold: filtered
        engine._maybe_passive_break_choice = lambda *_args: "protect"
        engine._should_trigger_adaptive_fallback = lambda *_args: False
        engine._maybe_force_finish_blow_choice = lambda _battle, _ordered, _choice: "earthquake"
        engine._maybe_take_setup_window_choice = lambda _battle, _ordered, _choice: "calmmind"
        engine._maybe_take_safe_recovery_choice = lambda _battle, _ordered, _choice: "recover"
        engine._maybe_take_progress_when_behind_choice = lambda _battle, _ordered, _choice: "swordsdance"
        engine._maybe_reduce_negative_matchup_switch = lambda _battle, _ordered, _choice: "switch skarmory"
        engine._maybe_accept_rerank_choice = lambda _battle, _ordered, _current, candidate, _confidence, _threshold: candidate
        diag = {}
        engine._diag_record_choice = lambda _battle, _ordered, choice, _confidence, _threshold, path: diag.update(choice=choice, path=path)
        engine._append_search_trace_example = lambda *_args, **_kwargs: None

        battle = DummyBattle()
        battle.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0),
        ]
        results = [(DummyResult([("earthquake", 60.0), ("calmmind", 40.0)]), 1.0)]

        choice = engine._select_move_from_results(results, battle)

        self.assertEqual(choice, "calmmind")
        self.assertEqual(diag["choice"], "calmmind")
        self.assertEqual(diag["path"], "rerank")


if __name__ == "__main__":
    unittest.main()
