import unittest
from types import SimpleNamespace

from src.players import oranguru_searchflow


class SearchflowBudgetTests(unittest.TestCase):
    def test_compute_search_budget_expands_early_uncertain_turns(self):
        engine = SimpleNamespace(
            SAMPLE_STATES=3,
            SEARCH_TIME_MS=200,
            DYNAMIC_SAMPLING=True,
            PARALLELISM=2,
            MAX_SAMPLE_STATES=10,
            _get_known_moves=lambda _opponent: [],
            _apply_world_budget_controls=lambda _battle, states: states,
        )
        opponent = SimpleNamespace(current_hp_fraction=1.0)
        battle = SimpleNamespace(
            opponent_team={"a": object(), "b": None},
            time_remaining=120,
        )

        sample_states, search_time_ms = oranguru_searchflow._compute_search_budget(engine, battle, opponent)

        self.assertEqual(sample_states, 8)
        self.assertEqual(search_time_ms, 100)

    def test_compute_search_budget_respects_time_pressure_and_cap(self):
        engine = SimpleNamespace(
            SAMPLE_STATES=2,
            SEARCH_TIME_MS=150,
            DYNAMIC_SAMPLING=True,
            PARALLELISM=3,
            MAX_SAMPLE_STATES=5,
            _get_known_moves=lambda _opponent: ["thunderbolt"],
            _apply_world_budget_controls=lambda _battle, states: min(states, 4),
        )
        opponent = SimpleNamespace(current_hp_fraction=0.8)
        battle = SimpleNamespace(
            opponent_team={"a": object(), "b": object(), "c": None, "d": None},
            time_remaining=45,
        )

        sample_states, search_time_ms = oranguru_searchflow._compute_search_budget(engine, battle, opponent)

        self.assertEqual(sample_states, 3)
        self.assertEqual(search_time_ms, 150)


if __name__ == "__main__":
    unittest.main()
