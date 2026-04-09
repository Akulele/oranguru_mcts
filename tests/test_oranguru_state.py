import unittest
from types import SimpleNamespace

from src.players import oranguru_state
from src.players.oranguru_engine import OranguruEnginePlayer
import constants


class DummyMove:
    def __init__(self, move_id):
        self.id = move_id


class OranguruStateTests(unittest.TestCase):
    def test_status_to_fp_maps_common_statuses(self):
        engine = SimpleNamespace()
        self.assertEqual(oranguru_state.status_to_fp(engine, "slp"), constants.SLEEP)
        self.assertEqual(oranguru_state.status_to_fp(engine, "par"), constants.PARALYZED)
        self.assertEqual(oranguru_state.status_to_fp(engine, "tox"), constants.TOXIC)
        self.assertIsNone(oranguru_state.status_to_fp(engine, None))

    def test_side_hazard_pressure_accumulates_layers(self):
        battle = SimpleNamespace(
            side_conditions={
                oranguru_state.SideCondition.STEALTH_ROCK: 1,
                oranguru_state.SideCondition.SPIKES: 2,
                oranguru_state.SideCondition.TOXIC_SPIKES: 1,
            }
        )
        pressure = oranguru_state.side_hazard_pressure(SimpleNamespace(), battle)
        self.assertAlmostEqual(pressure, 0.125 + (1.0 / 6.0) + 0.08)

    def test_engine_fp_move_id_binding(self):
        engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        self.assertEqual(engine._fp_move_id(DummyMove("Thunderbolt")), "thunderbolt")

    def test_sleep_clause_banned_choices_bans_sleep_move_and_tera_variant(self):
        engine = SimpleNamespace(
            _sleep_clause_blocked=lambda _battle: True,
            _move_inflicts_sleep=lambda move: move.id == "spore",
        )
        battle = SimpleNamespace(available_moves=[DummyMove("spore"), DummyMove("gigadrain")])

        banned = oranguru_state.sleep_clause_banned_choices(engine, battle)

        self.assertEqual(banned, {"spore", "spore-tera"})


if __name__ == "__main__":
    unittest.main()
