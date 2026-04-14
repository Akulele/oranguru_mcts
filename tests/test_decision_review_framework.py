import unittest

from evaluation.build_decision_review_pack import build_review_pack
from evaluation.mine_bad_decisions import mine_examples


class DecisionReviewFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.moves_data = {
            "tackle": {"category": "Physical", "type": "Normal"},
            "thunderwave": {"category": "Status", "type": "Electric", "status": "par"},
            "willowisp": {"category": "Status", "type": "Fire", "status": "brn"},
            "calmmind": {"category": "Status", "type": "Psychic", "target": "self", "boosts": {"spa": 1, "spd": 1}},
            "protect": {"category": "Status", "type": "Normal"},
            "recover": {"category": "Status", "type": "Normal"},
            "stealthrock": {"category": "Status", "type": "Rock"},
            "dragondance": {"category": "Status", "type": "Dragon", "target": "self", "boosts": {"atk": 1, "spe": 1}},
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

    def test_setup_window_records_setup_alternative(self):
        rows = [
            {
                "battle_id": "b3",
                "turn": 11,
                "chosen_choice": "tackle",
                "top_actions": [
                    {"choice": "tackle", "weight": 72.0, "heuristic_score": 50.0},
                    {"choice": "dragondance", "weight": 80.0, "heuristic_score": 90.0},
                ],
                "action_labels": ["tackle", "dragondance"],
                "winner": "opp",
                "bot_id": "bot",
                "best_reply_score": 60.0,
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Dragonite", "hp": 90, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Scizor", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Blissey", "hp": 90, "max_hp": 100, "types": ["Normal"], "boosts": {}},
                        "reserve": [{"name": "Corviknight", "hp": 100, "max_hp": 100}],
                    },
                },
                "setup_window": {"reason": "policy_ratio"},
            }
        ]
        summary = mine_examples(rows, moves_data=self.moves_data)
        self.assertEqual(summary["setup_window_reasons"], {"policy_ratio": 1})
        sample = summary["samples"]["underused_setup_window"][0]
        self.assertEqual(sample["alternative"], "dragondance")
        self.assertEqual(sample["alternative_heuristic_score"], 90.0)
        pack = build_review_pack(summary, limit=5)
        setup_row = next(row for row in pack if row["category"] == "underused_setup_window")
        self.assertIn("dragondance", setup_row["review_blurb"])

    def test_setup_window_requires_supported_setup_alternative(self):
        def row(battle_id, choice, setup_weight, attack_heur, setup_heur):
            return {
                "battle_id": battle_id,
                "turn": 11,
                "chosen_choice": choice,
                "top_actions": [
                    {"choice": choice, "weight": 80.0, "score": 80.0, "heuristic_score": attack_heur},
                    {"choice": "dragondance", "weight": setup_weight, "score": setup_weight, "heuristic_score": setup_heur},
                ],
                "action_labels": [choice, "dragondance"],
                "winner": "opp",
                "bot_id": "bot",
                "best_reply_score": 60.0,
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Dragonite", "hp": 90, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Scizor", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Blissey", "hp": 90, "max_hp": 100, "types": ["Normal"], "boosts": {}},
                        "reserve": [{"name": "Corviknight", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("protect-choice", "protect", setup_weight=80.0, attack_heur=0.0, setup_heur=120.0),
                row("low-policy", "tackle", setup_weight=10.0, attack_heur=0.0, setup_heur=120.0),
                row("no-heur-regret", "tackle", setup_weight=80.0, attack_heur=10.0, setup_heur=10.0),
                row("regret", "tackle", setup_weight=30.0, attack_heur=0.0, setup_heur=20.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["underused_setup_window"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])

    def test_status_window_records_supported_status_alternative(self):
        def row(battle_id, status_weight, attack_heur, status_heur):
            return {
                "battle_id": battle_id,
                "turn": 11,
                "chosen_choice": "tackle",
                "top_actions": [
                    {"choice": "tackle", "weight": 80.0, "score": 80.0, "heuristic_score": attack_heur},
                    {"choice": "willowisp", "weight": status_weight, "score": status_weight, "heuristic_score": status_heur},
                ],
                "action_labels": ["tackle", "willowisp"],
                "winner": "opp",
                "bot_id": "bot",
                "best_reply_score": 60.0,
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Rotom", "hp": 90, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Scizor", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Tauros", "hp": 90, "max_hp": 100, "types": ["Normal"], "boosts": {}},
                        "reserve": [{"name": "Corviknight", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("low-policy", status_weight=10.0, attack_heur=0.0, status_heur=120.0),
                row("no-heur-regret", status_weight=80.0, attack_heur=10.0, status_heur=10.0),
                row("regret", status_weight=30.0, attack_heur=0.0, status_heur=20.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["underused_status_window"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])
        self.assertEqual(samples[0]["alternative"], "willowisp")

    def test_safe_recovery_requires_supported_recovery_alternative(self):
        def row(battle_id, recovery_weight, choice_heur, recovery_heur):
            return {
                "battle_id": battle_id,
                "turn": 7,
                "chosen_choice": "tackle",
                "top_actions": [
                    {"choice": "tackle", "weight": 80.0, "score": 80.0, "heuristic_score": choice_heur},
                    {"choice": "recover", "weight": recovery_weight, "score": recovery_weight, "heuristic_score": recovery_heur},
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
                        "reserve": [{"name": "Skarmory", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("low-policy", recovery_weight=10.0, choice_heur=0.0, recovery_heur=120.0),
                row("no-heur-regret", recovery_weight=80.0, choice_heur=10.0, recovery_heur=10.0),
                row("regret", recovery_weight=30.0, choice_heur=0.0, recovery_heur=20.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["ignored_safe_recovery"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])
        self.assertEqual(samples[0]["alternative"], "recover")

    def test_failed_progress_requires_supported_progress_alternative(self):
        def row(battle_id, progress_weight, choice_heur, progress_heur):
            return {
                "battle_id": battle_id,
                "turn": 12,
                "chosen_choice": "tackle",
                "top_actions": [
                    {"choice": "tackle", "weight": 80.0, "score": 80.0, "heuristic_score": choice_heur},
                    {"choice": "calmmind", "weight": progress_weight, "score": progress_weight, "heuristic_score": progress_heur},
                ],
                "action_labels": ["tackle", "calmmind"],
                "winner": "opp",
                "bot_id": "bot",
                "best_reply_score": 60.0,
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Mew", "hp": 90, "max_hp": 100, "boosts": {}},
                        "reserve": [],
                    },
                    "opponent": {
                        "active": {"name": "Blissey", "hp": 90, "max_hp": 100, "types": ["Normal"], "boosts": {}},
                        "reserve": [{"name": "Corviknight", "hp": 100, "max_hp": 100}, {"name": "Rotom", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("low-policy", progress_weight=10.0, choice_heur=0.0, progress_heur=120.0),
                row("no-heur-regret", progress_weight=80.0, choice_heur=10.0, progress_heur=10.0),
                row("regret", progress_weight=30.0, choice_heur=0.0, progress_heur=20.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["failed_to_progress_when_behind"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])
        self.assertEqual(samples[0]["alternative"], "calmmind")
        self.assertEqual(samples[0]["progress_kind"], "setup")

    def test_over_switch_requires_heuristic_regret_when_available(self):
        def row(battle_id, switch_heur, attack_heur, attack_weight=79.0):
            return {
                "battle_id": battle_id,
                "turn": 1,
                "chosen_choice": "switch skarmory",
                "top_actions": [
                    {"choice": "switch skarmory", "weight": 80.0, "score": 80.0, "heuristic_score": switch_heur, "risk_penalty": 0.0},
                    {"choice": "tackle", "weight": attack_weight, "score": attack_weight, "heuristic_score": attack_heur, "risk_penalty": 0.0},
                ],
                "action_labels": ["tackle", "switch skarmory"],
                "winner": "opp",
                "bot_id": "bot",
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Pikachu", "hp": 80, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Skarmory", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Starmie", "hp": 80, "max_hp": 100, "types": ["Water"], "boosts": {}},
                        "reserve": [{"name": "Zapdos", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("no-regret", switch_heur=2.5, attack_heur=2.0),
                row("low-policy", switch_heur=1.0, attack_heur=8.0, attack_weight=10.0),
                row("regret", switch_heur=1.0, attack_heur=3.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["over_switched_negative_matchup"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])

    def test_missed_ko_requires_policy_and_heuristic_support_when_available(self):
        def row(battle_id, choice, chosen_heur, attack_heur, attack_weight=70.0):
            return {
                "battle_id": battle_id,
                "turn": 1,
                "chosen_choice": choice,
                "top_actions": [
                    {"choice": choice, "weight": 80.0, "score": 80.0, "heuristic_score": chosen_heur},
                    {"choice": "tackle", "weight": attack_weight, "score": attack_weight, "heuristic_score": attack_heur},
                ],
                "action_labels": [choice, "tackle"],
                "winner": "opp",
                "bot_id": "bot",
                "fp_oracle_battle": {
                    "user": {
                        "active": {"name": "Pikachu", "hp": 80, "max_hp": 100, "boosts": {}},
                        "reserve": [{"name": "Skarmory", "hp": 100, "max_hp": 100}],
                    },
                    "opponent": {
                        "active": {"name": "Starmie", "hp": 20, "max_hp": 100, "types": ["Water"], "boosts": {}},
                        "reserve": [{"name": "Zapdos", "hp": 100, "max_hp": 100}],
                    },
                },
            }

        summary = mine_examples(
            [
                row("defensive-no-regret", "recover", chosen_heur=90.0, attack_heur=3.0),
                row("low-policy", "thunderwave", chosen_heur=0.0, attack_heur=8.0, attack_weight=10.0),
                row("regret", "thunderwave", chosen_heur=0.0, attack_heur=3.0),
            ],
            moves_data=self.moves_data,
        )
        samples = summary["samples"]["missed_ko"]
        self.assertEqual([sample["battle_id"] for sample in samples], ["regret"])

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
        self.assertIn("tackle", pack[0]["review_blurb"])
        losses_only = build_review_pack(summary, limit=5, losses_only=True)
        self.assertEqual(len(losses_only), 1)
        self.assertEqual(losses_only[0]["category"], "missed_ko")


if __name__ == "__main__":
    unittest.main()
