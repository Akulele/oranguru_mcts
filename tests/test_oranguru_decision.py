import unittest
from types import SimpleNamespace

from poke_env.battle import MoveCategory

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, category=None, base_power=0):
        self.id = move_id
        self.category = category
        self.base_power = base_power
        self.damage = None


class DummyPokemon:
    def __init__(self, current_hp_fraction=1.0, status=None):
        self.current_hp_fraction = current_hp_fraction
        self.status = status
        self.boosts = {}


class DummyBattle:
    def __init__(self):
        self.force_switch = False
        self.available_moves = [DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100)]
        self.available_switches = [SimpleNamespace(species="skarmory")]
        self.active_pokemon = DummyPokemon(0.9)
        self.opponent_active_pokemon = DummyPokemon(0.8)


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


if __name__ == "__main__":
    unittest.main()
