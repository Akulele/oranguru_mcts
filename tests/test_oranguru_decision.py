import unittest
from types import SimpleNamespace

from poke_env.battle import MoveCategory

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, category=None, base_power=0):
        self.id = move_id
        self.category = category
        self.base_power = base_power


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


class OranguruDecisionTests(unittest.TestCase):
    def test_finish_blow_guard_prefers_damage_over_passive_choice(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_damage_score = lambda *_args: 120.0
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


if __name__ == "__main__":
    unittest.main()
