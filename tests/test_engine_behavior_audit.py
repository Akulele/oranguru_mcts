import unittest

from evaluation.audit_engine_behavior import analyze_examples


def _fp_battle(
    *,
    user_active=None,
    opponent_active=None,
    opponent_reserve=None,
):
    return {
        "user": {"active": user_active or {}, "reserve": []},
        "opponent": {
            "active": opponent_active or {},
            "reserve": opponent_reserve or [],
        },
    }


class EngineBehaviorAuditTests(unittest.TestCase):
    def test_flags_status_into_statused_target(self):
        rows = [
            {
                "battle_id": "b1",
                "turn": 5,
                "chosen_choice": "thunderwave",
                "selection_path": "mcts",
                "source": "test",
                "tag": "test",
                "fp_oracle_battle": _fp_battle(
                    user_active={"name": "jolteon", "hp": 100, "max_hp": 100},
                    opponent_active={"name": "snorlax", "status": "slp", "types": ["normal"], "hp": 100, "max_hp": 100},
                ),
            }
        ]
        moves = {"thunderwave": {"name": "Thunder Wave", "status": "par", "category": "Status"}}
        summary = analyze_examples(rows, moves_data=moves, sample_limit=5)
        self.assertEqual(summary["issue_counts"].get("status_into_statused_target"), 1)

    def test_flags_repeat_passive_move(self):
        battle = _fp_battle(
            user_active={"name": "blissey", "hp": 300, "max_hp": 400},
            opponent_active={"name": "tinglu", "hp": 200, "max_hp": 300, "types": ["ground", "dark"]},
        )
        rows = [
            {
                "battle_id": "b2",
                "turn": 8,
                "chosen_choice": "softboiled",
                "selection_path": "mcts",
                "source": "test",
                "tag": "test",
                "fp_oracle_battle": battle,
            },
            {
                "battle_id": "b2",
                "turn": 9,
                "chosen_choice": "softboiled",
                "selection_path": "mcts",
                "source": "test",
                "tag": "test",
                "fp_oracle_battle": battle,
            },
        ]
        moves = {"softboiled": {"name": "Soft-Boiled", "heal": [1, 2], "category": "Status"}}
        summary = analyze_examples(rows, moves_data=moves, passive_repeat_streak=2, sample_limit=5)
        self.assertEqual(summary["issue_counts"].get("repeat_passive_move"), 1)

    def test_flags_sleep_clause_risk(self):
        rows = [
            {
                "battle_id": "b3",
                "turn": 4,
                "chosen_choice": "spore",
                "selection_path": "mcts",
                "source": "test",
                "tag": "test",
                "fp_oracle_battle": _fp_battle(
                    user_active={"name": "breloom", "hp": 100, "max_hp": 100},
                    opponent_active={"name": "slowbro", "status": "", "types": ["water", "psychic"], "hp": 100, "max_hp": 100},
                    opponent_reserve=[{"name": "zapdos", "status": "slp", "types": ["electric", "flying"], "hp": 100, "max_hp": 100}],
                ),
            }
        ]
        moves = {"spore": {"name": "Spore", "status": "slp", "category": "Status"}}
        summary = analyze_examples(rows, moves_data=moves, sample_limit=5)
        self.assertEqual(summary["issue_counts"].get("sleep_clause_risk"), 1)


if __name__ == "__main__":
    unittest.main()
