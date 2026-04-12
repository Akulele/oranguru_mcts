import unittest

from poke_env.battle import MoveCategory

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, category=None, base_power=0):
        self.id = move_id
        self.category = category
        self.base_power = base_power
        self.boosts = {}
        self.target = None


class DummyPokemon:
    def __init__(self, current_hp_fraction=1.0):
        self.current_hp_fraction = current_hp_fraction
        self.boosts = {}


class DummyBattle:
    def __init__(self):
        self.force_switch = False
        self.active_pokemon = DummyPokemon(0.8)
        self.opponent_active_pokemon = DummyPokemon(0.8)
        setup = DummyMove("calmmind", category=MoveCategory.STATUS, base_power=0)
        setup.boosts = {"spa": 1}
        setup.target = "self"
        self.available_moves = [setup, DummyMove("scald", category=MoveCategory.SPECIAL, base_power=80)]


class SetupWindowHighGainTests(unittest.TestCase):
    def test_high_heuristic_gain_can_override_normal_policy_floor(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SETUP_WINDOW_MIN_HP = 0.65
        engine.SETUP_WINDOW_MAX_REPLY = 110.0
        engine.SETUP_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.SETUP_WINDOW_MIN_HEUR_GAIN = 15.0
        engine.SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.20
        engine.SETUP_WINDOW_HIGH_HEUR_GAIN = 60.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_reply_score = lambda *_args: 40.0
        engine._estimate_best_damage_score = lambda *_args: 60.0
        engine._heuristic_action_score = lambda _battle, choice: 95.0 if choice == "calmmind" else 3.0
        engine._should_setup_move = lambda move, _active, _opponent: move.id == "calmmind"
        battle = DummyBattle()

        adjusted = engine._maybe_take_setup_window_choice(
            battle,
            [("scald", 80.0), ("calmmind", 20.0)],
            "scald",
        )

        self.assertEqual(adjusted, "calmmind")

    def test_low_policy_setup_still_rejected(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.SETUP_WINDOW_MIN_HP = 0.65
        engine.SETUP_WINDOW_MAX_REPLY = 110.0
        engine.SETUP_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.SETUP_WINDOW_MIN_HEUR_GAIN = 15.0
        engine.SETUP_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.20
        engine.SETUP_WINDOW_HIGH_HEUR_GAIN = 60.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_reply_score = lambda *_args: 40.0
        engine._estimate_best_damage_score = lambda *_args: 60.0
        engine._heuristic_action_score = lambda _battle, choice: 120.0 if choice == "calmmind" else 3.0
        engine._should_setup_move = lambda move, _active, _opponent: move.id == "calmmind"
        battle = DummyBattle()

        adjusted = engine._maybe_take_setup_window_choice(
            battle,
            [("scald", 80.0), ("calmmind", 10.0)],
            "scald",
        )

        self.assertEqual(adjusted, "scald")


if __name__ == "__main__":
    unittest.main()
