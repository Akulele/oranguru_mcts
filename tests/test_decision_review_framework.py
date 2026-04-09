import unittest

from evaluation.build_decision_review_pack import build_review_pack
from evaluation.mine_bad_decisions import mine_examples


class DecisionReviewFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.moves_data = {
            "tackle": {"category": "Physical", "type": "Normal"},
            "thunderwave": {"category": "Status", "type": "Electric", "status": "par"},
            "calmmind": {"category": "Status", "type": "Psychic", "target": "self", "boosts": {"spa": 1, "spd": 1}},
            "recover": {"category": "Status", "type": "Normal"},
        }

    def test_mine_examples_flags_strategic_buckets(self):
        rows = [
            {
                "battle_id": "b1",
                "turn": 5,
                "chosen_choice": "thunderwave",
                "top_actions": [
                    {"choice": "thunderwave", "score": 80.0},
                    {"choice": "tackle", "score": 70.0},
                ],
                "action_labels": ["thunderwave", "tackle"],
                "winner": "opp",
                "bot_id": "bot",
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Pikachu", "hp": 100, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Bulbasaur", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Starmie", "hp": 20, "max_hp": 100, "types": ["Water"], "boosts": {}},
                        "reserve": [{"name": "Zapdos", "hp": 100, "max_hp": 100}],
                    },
                },
            },
            {
                "battle_id": "b2",
                "turn": 7,
                "chosen_choice": "tackle",
                "top_actions": [
                    {"choice": "tackle", "score": 75.0},
                    {"choice": "recover", "score": 74.0},
                ],
                "action_labels": ["tackle", "recover"],
                "winner": "opp",
                "bot_id": "bot",
                "best_reply_score": 50.0,
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Slowbro", "hp": 30, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Chansey", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Garchomp", "hp": 80, "max_hp": 100, "types": ["Dragon", "Ground"], "boosts": {}},
                        "reserve": [{"name": "Skarmory", "hp": 100, "max_hp": 100}, {"name": "Rotom", "hp": 100, "max_hp": 100}],
                    },
                },
            },
        ]
        summary = mine_examples(rows, moves_data=self.moves_data)
        self.assertGreaterEqual(summary["issue_counts"].get("missed_ko", 0), 1)
        self.assertGreaterEqual(summary["issue_counts"].get("ignored_safe_recovery", 0), 1)

    def test_build_review_pack_sorts_by_priority(self):
        summary = {
            "samples": {
                "underused_setup_window": [
                    {"battle_id": "b2", "turn": 8, "choice": "tackle", "active_species": "mew", "opponent_species": "blissey", "priority": 70.0, "lost_battle": False},
                ],
                "missed_ko": [
                    {"battle_id": "b1", "turn": 5, "choice": "thunderwave", "active_species": "pikachu", "opponent_species": "starmie", "priority": 120.0, "lost_battle": True, "best_choice": "tackle"},
                ],
            }
        }
        pack = build_review_pack(summary, limit=5)
        self.assertEqual(pack[0]["category"], "missed_ko")
        self.assertIn("KO", pack[0]["review_blurb"])
        losses_only = build_review_pack(summary, limit=5, losses_only=True)
        self.assertEqual(len(losses_only), 1)
        self.assertEqual(losses_only[0]["category"], "missed_ko")


if __name__ == "__main__":
    unittest.main()
