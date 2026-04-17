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
        self.active_pokemon = DummyPokemon(0.3)
        self.opponent_active_pokemon = DummyPokemon(0.8)
        self.available_moves = [
            DummyMove("recover", category=MoveCategory.STATUS, base_power=0),
            DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
        ]


class SafeRecoveryWindowTests(unittest.TestCase):
    def _engine(self, *, active_hp=0.3, reply_score=40.0, recover_heuristic=120.0):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        engine.RECOVERY_WINDOW_MAX_HP = 0.4
        engine.RECOVERY_WINDOW_MAX_REPLY = 110.0
        engine.RECOVERY_WINDOW_MIN_OPP_HP = 0.25
        engine.RECOVERY_WINDOW_MIN_POLICY_RATIO = 0.65
        engine.RECOVERY_WINDOW_HIGH_GAIN_MIN_POLICY_RATIO = 0.42
        engine.RECOVERY_WINDOW_CRITICAL_HP = 0.30
        engine.RECOVERY_WINDOW_CRITICAL_MIN_POLICY_RATIO = 0.33
        engine.RECOVERY_WINDOW_MIN_HEUR_GAIN = 1.0
        engine.RECOVERY_WINDOW_HIGH_HEUR_GAIN = 10.0
        engine.TACTICAL_KO_THRESHOLD = 220.0
        engine._estimate_best_reply_score = lambda *_args: reply_score
        engine._estimate_best_damage_score = lambda *_args: 40.0
        engine._heuristic_action_score = (
            lambda _battle, choice: recover_heuristic if choice == "recover" else 2.0
        )
        engine._is_recovery_move = lambda move: move.id == "recover"
        battle = DummyBattle()
        battle.active_pokemon = DummyPokemon(active_hp)
        return engine, battle

    def test_takes_safe_recovery_at_low_hp(self):
        engine, battle = self._engine()
        mem = {}
        engine._get_battle_memory = lambda _battle: mem

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("earthquake", 80.0), ("recover", 40.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "recover")
        self.assertEqual(mem["recovery_window_last"]["reason"], "take_recovery")

    def test_rejects_high_gain_recovery_with_weak_policy_support(self):
        engine, battle = self._engine(active_hp=0.35)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("earthquake", 80.0), ("recover", 30.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(mem["recovery_window_last"]["reason"], "policy_ratio")

    def test_accepts_borderline_high_gain_recovery_support(self):
        engine, battle = self._engine(active_hp=0.35)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("earthquake", 80.0), ("recover", 34.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "recover")
        self.assertEqual(mem["recovery_window_last"]["reason"], "take_recovery")

    def test_critical_hp_allows_lower_policy_ratio_for_recovery(self):
        engine, battle = self._engine(active_hp=0.2)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("earthquake", 80.0), ("recover", 30.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "recover")
        self.assertEqual(mem["recovery_window_last"]["reason"], "take_recovery")
        self.assertTrue(mem["recovery_window_last"]["critical_hp"])

    def test_rejects_recovery_when_reply_is_unsafe(self):
        engine, battle = self._engine(reply_score=160.0)
        mem = {}
        engine._get_battle_memory = lambda _battle: mem

        adjusted = engine._maybe_take_safe_recovery_choice(
            battle,
            [("earthquake", 80.0), ("recover", 30.0)],
            "earthquake",
        )

        self.assertEqual(adjusted, "earthquake")
        self.assertEqual(mem["recovery_window_last"]["reason"], "unsafe_reply")


if __name__ == "__main__":
    unittest.main()
