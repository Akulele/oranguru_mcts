import unittest

from poke_env.battle import MoveCategory

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, category=None, base_power=0):
        self.id = move_id
        self.category = category
        self.base_power = base_power


class DummyPokemon:
    def __init__(self, current_hp_fraction=1.0):
        self.current_hp_fraction = current_hp_fraction
        self.boosts = {}


class DummyBattle:
    def __init__(self):
        self.force_switch = False
        self.active_pokemon = DummyPokemon(0.9)
        self.opponent_active_pokemon = DummyPokemon(0.2)
        self.available_moves = [
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            DummyMove("recover", category=MoveCategory.STATUS, base_power=0),
        ]


class FinishBlowDiagnosticTests(unittest.TestCase):
    def _engine(self, *, best_damage_score=120.0, finish_heuristic=90.0, passive_heuristic=10.0):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine.FINISH_BLOW_NEAR_KO_RATIO = 0.90
        engine._estimate_best_damage_score = lambda *_args: best_damage_score
        engine._heuristic_action_score = (
            lambda _battle, choice: finish_heuristic if choice == "earthquake" else passive_heuristic
        )
        return engine

    def test_records_finish_taken_from_passive_choice(self):
        engine = self._engine()
        mem = {}
        engine._get_battle_memory = lambda _battle: mem
        battle = DummyBattle()

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("recover", 80.0), ("earthquake", 20.0)],
            "recover",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(mem["finish_blow_last"]["reason"], "take_passive_finish")
        self.assertEqual(mem["finish_blow_last"]["finish_choice"], "earthquake")

    def test_records_no_ko_window(self):
        engine = self._engine(best_damage_score=20.0)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem
        battle = DummyBattle()

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("recover", 80.0), ("earthquake", 20.0)],
            "recover",
        )

        self.assertEqual(adjusted, "recover")
        self.assertEqual(mem["finish_blow_last"]["reason"], "no_ko_window")
        self.assertLess(
            mem["finish_blow_last"]["best_damage_score"],
            mem["finish_blow_last"]["near_ko_threshold"],
        )

    def test_takes_near_ko_finish_from_passive_choice(self):
        engine = self._engine(best_damage_score=40.0)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem
        battle = DummyBattle()

        adjusted = engine._maybe_force_finish_blow_choice(
            battle,
            [("irondefense", 80.0), ("earthquake", 20.0)],
            "irondefense",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(mem["finish_blow_last"]["reason"], "take_passive_finish")
        self.assertLess(
            mem["finish_blow_last"]["best_damage_score"],
            mem["finish_blow_last"]["ko_threshold"],
        )
        self.assertGreaterEqual(
            mem["finish_blow_last"]["best_damage_score"],
            mem["finish_blow_last"]["near_ko_threshold"],
        )


if __name__ == "__main__":
    unittest.main()
