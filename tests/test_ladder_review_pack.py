import unittest

from evaluation.build_ladder_review_pack import build_ladder_review_pack


class LadderReviewPackTests(unittest.TestCase):
    def test_prioritizes_negative_residual_low_confidence_loss(self):
        ladder_rows = [
            {
                "schema_version": 1,
                "battle_tag": "battle-1",
                "result": "loss",
                "expected_score": 0.65,
                "rating_residual": -0.65,
                "opponent_rating_pre": 1500,
                "decision_ms_avg": 2400.0,
                "opp_remaining": 1,
            },
            {
                "schema_version": 1,
                "battle_tag": "battle-2",
                "result": "win",
                "expected_score": 0.50,
                "rating_residual": 0.50,
                "opponent_rating_pre": 1500,
                "decision_ms_avg": 2300.0,
                "opp_remaining": 3,
            },
        ]
        trace_rows = [
            {
                "battle_id": "battle-1",
                "turn": 28,
                "chosen_choice": "recover",
                "selection_path": "mcts",
                "policy_confidence": 0.18,
                "phase": "end",
                "top_actions": [
                    {"choice": "earthquake", "kind": "attack", "score": 0.55, "heuristic_score": 20.0},
                    {"choice": "recover", "kind": "recovery", "score": 0.35, "heuristic_score": 40.0},
                ],
                "fp_oracle_battle": {
                    "user": {
                        "active": {
                            "name": "Hippowdon",
                            "hp": 60,
                            "max_hp": 100,
                            "status": "brn",
                            "moves": [{"name": "earthquake"}, {"name": "slackoff"}],
                        },
                        "reserve": [{"name": "Rotom", "hp": 0, "max_hp": 100}],
                        "side_conditions": {"stealthrock": 1},
                    },
                    "opponent": {
                        "active": {"name": "Pikachu", "hp": 20, "max_hp": 100},
                        "reserve": [{"name": "Blissey", "hp": 100, "max_hp": 100}],
                        "side_conditions": {"spikes": 1},
                    },
                    "weather": "raindance",
                },
            },
            {
                "battle_id": "battle-2",
                "turn": 4,
                "chosen_choice": "earthquake",
                "selection_path": "mcts",
                "policy_confidence": 0.80,
                "phase": "opening",
                "top_actions": [
                    {"choice": "earthquake", "kind": "attack", "score": 0.80, "heuristic_score": 20.0},
                    {"choice": "recover", "kind": "recovery", "score": 0.10, "heuristic_score": 40.0},
                ],
            },
        ]
        teacher_rows = [
            {
                "battle_id": "battle-1",
                "turn": 28,
                "teacher_source": "fp_oracle",
                "action_labels": ["earthquake", "recover"],
                "action_mask": [True, True],
                "policy_target": [0.72, 0.28],
                "teacher_entropy": 0.59,
                "teacher_samples_used": 4,
                "teacher_total_visits": 1200,
            }
        ]

        pack = build_ladder_review_pack(
            ladder_rows=ladder_rows,
            trace_rows=trace_rows,
            teacher_rows=teacher_rows,
            limit=5,
            max_per_battle=3,
        )

        self.assertEqual(pack["summary"]["ladder_battles"], 2)
        self.assertEqual(pack["summary"]["trace_rows"], 2)
        self.assertEqual(pack["rows"][0]["battle_id"], "battle-1")
        self.assertEqual(pack["rows"][0]["choice"], "recover")
        self.assertIn("negative residual", " ".join(pack["rows"][0]["reasons"]))
        self.assertEqual(pack["rows"][0]["active_species"], "hippowdon")
        self.assertEqual(pack["rows"][0]["opponent_species"], "pikachu")
        self.assertEqual(pack["rows"][0]["board_context"]["user"]["alive"], 1)
        self.assertEqual(pack["rows"][0]["board_context"]["opponent"]["alive"], 2)
        self.assertEqual(pack["rows"][0]["board_context"]["opponent"]["hidden_estimate"], 4)
        self.assertEqual(pack["rows"][0]["board_context"]["field"]["weather"], "raindance")
        self.assertEqual(pack["rows"][0]["nearby_turns"][0]["turn"], 28)
        self.assertEqual(pack["rows"][0]["teacher"]["source"], "fp_oracle")
        self.assertEqual(pack["rows"][0]["teacher"]["top_choice"], "earthquake")
        self.assertEqual(pack["summary"]["teacher_rows"], 1)
        self.assertEqual(pack["summary"]["teacher_disagreements"], 1)


if __name__ == "__main__":
    unittest.main()
