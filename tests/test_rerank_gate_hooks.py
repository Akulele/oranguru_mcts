import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.players.oranguru_engine import OranguruEnginePlayer
from src.players.oranguru_rerank_gate import build_trace_rerank_gate_example


class RerankGateHookTests(unittest.TestCase):
    def test_builds_trace_gate_example_for_accepted_setup_rerank(self):
        row = {
            "battle_id": "b1",
            "turn": 7,
            "value_target": -1.0,
            "selection_path": "rerank",
            "chosen_choice": "calmmind",
            "policy_confidence": 0.31,
            "policy_threshold": 0.60,
            "matchup_score": 0.25,
            "best_reply_score": 80.0,
            "hazard_load": 0.1,
            "top_actions": [
                {"choice": "earthquake", "kind": "attack", "score": 0.60, "heuristic_score": 20.0, "risk_penalty": 4.0},
                {"choice": "calmmind", "kind": "setup", "score": 0.42, "heuristic_score": 100.0, "risk_penalty": 8.0},
            ],
            "setup_window": {
                "reason": "take_setup",
                "chosen_choice": "earthquake",
                "setup_choice": "calmmind",
                "active_hp": 0.9,
                "opp_hp": 0.8,
                "reply_score": 70.0,
                "chosen_weight": 0.60,
                "setup_weight": 0.42,
                "chosen_heuristic": 20.0,
                "setup_heuristic": 100.0,
            },
        }

        example = build_trace_rerank_gate_example(row)

        self.assertIsNotNone(example)
        self.assertEqual(example["label"], 0)
        self.assertEqual(example["source"], "setup_window:take_setup")
        self.assertEqual(example["candidate_choice"], "calmmind")
        self.assertEqual(example["top1_choice"], "earthquake")
        self.assertAlmostEqual(example["features"]["source_setup_window"], 1.0)
        self.assertAlmostEqual(example["features"]["score_drop_top1_minus_candidate"], 0.18)
        self.assertAlmostEqual(example["features"]["heuristic_delta_candidate_minus_top1"], 80.0)
        self.assertEqual(len(example["feature_vector"]), len(example["feature_names"]))

    def test_runtime_gate_can_block_tactical_rerank_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "gate.json"
            model_path.write_text(
                json.dumps(
                    {
                        "feature_names": ["source_setup_window"],
                        "weights": [-10.0],
                        "bias": 0.0,
                        "threshold": 0.50,
                    }
                ),
                encoding="utf-8",
            )

            engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
            engine.RERANK_GATE_ENABLED = True
            engine.RERANK_GATE_MODEL = str(model_path)
            engine.RERANK_GATE_THRESHOLD = 0.50
            engine.RERANK_GATE_FAIL_OPEN = True
            engine._rerank_gate_model = None
            engine._rerank_gate_failed = False
            engine._mcts_stats = {}
            memory = {
                "setup_window_last": {
                    "reason": "take_setup",
                    "chosen_choice": "earthquake",
                    "setup_choice": "calmmind",
                    "active_hp": 0.9,
                    "opp_hp": 0.8,
                    "reply_score": 70.0,
                    "chosen_weight": 0.60,
                    "setup_weight": 0.42,
                    "chosen_heuristic": 20.0,
                    "setup_heuristic": 100.0,
                }
            }
            engine._get_battle_memory = lambda _battle: memory
            engine._heuristic_action_score = lambda _battle, choice: 100.0 if choice == "calmmind" else 20.0
            engine._adaptive_choice_risk_penalty = lambda _battle, _choice: 0.0
            engine._search_trace_choice_kind = lambda _battle, choice: "setup" if choice == "calmmind" else "attack"
            engine._estimate_matchup = lambda *_args: 0.25
            engine._estimate_best_reply_score = lambda *_args: 80.0
            engine._side_hazard_pressure = lambda _battle: 0.0
            battle = SimpleNamespace(active_pokemon=object(), opponent_active_pokemon=object())

            accepted = engine._maybe_accept_rerank_choice(
                battle,
                [("earthquake", 0.60), ("calmmind", 0.42)],
                "earthquake",
                "calmmind",
                0.31,
                0.60,
            )

        self.assertEqual(accepted, "earthquake")
        self.assertEqual(memory["rerank_gate_last"]["reason"], "block")
        self.assertEqual(memory["rerank_gate_last"]["source"], "setup_window:take_setup")
        self.assertEqual(engine._mcts_stats["rerank_gate_blocked"], 1)

    def test_runtime_bucket_rule_gate_blocks_matching_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "gate.json"
            model_path.write_text(
                json.dumps(
                    {
                        "mode": "bucket_rules",
                        "default": "allow",
                        "rules": [
                            {
                                "source": "setup_window:take_setup",
                                "score_drop_max": 0.05,
                                "heuristic_delta_min": 60.0,
                                "heuristic_delta_max": 100.0,
                                "action": "block",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
            engine.RERANK_GATE_ENABLED = True
            engine.RERANK_GATE_MODEL = str(model_path)
            engine.RERANK_GATE_THRESHOLD = 0.50
            engine.RERANK_GATE_FAIL_OPEN = True
            engine._rerank_gate_model = None
            engine._rerank_gate_failed = False
            engine._mcts_stats = {}
            memory = {
                "setup_window_last": {
                    "reason": "take_setup",
                    "chosen_choice": "earthquake",
                    "setup_choice": "calmmind",
                    "chosen_weight": 0.60,
                    "setup_weight": 0.58,
                    "chosen_heuristic": 20.0,
                    "setup_heuristic": 90.0,
                }
            }
            engine._get_battle_memory = lambda _battle: memory
            engine._heuristic_action_score = lambda _battle, choice: 90.0 if choice == "calmmind" else 20.0
            engine._adaptive_choice_risk_penalty = lambda _battle, _choice: 0.0
            engine._search_trace_choice_kind = lambda _battle, choice: "setup" if choice == "calmmind" else "attack"
            engine._estimate_matchup = lambda *_args: 0.25
            engine._estimate_best_reply_score = lambda *_args: 80.0
            engine._side_hazard_pressure = lambda _battle: 0.0
            battle = SimpleNamespace(active_pokemon=object(), opponent_active_pokemon=object())

            accepted = engine._maybe_accept_rerank_choice(
                battle,
                [("earthquake", 0.60), ("calmmind", 0.58)],
                "earthquake",
                "calmmind",
                0.31,
                0.60,
            )

        self.assertEqual(accepted, "earthquake")
        self.assertEqual(memory["rerank_gate_last"]["reason"], "block")
        self.assertEqual(memory["rerank_gate_last"]["mode"], "bucket_rules")
        self.assertEqual(memory["rerank_gate_last"]["rule_idx"], 0)
        self.assertAlmostEqual(memory["rerank_gate_last"]["score_drop"], 0.02)
        self.assertAlmostEqual(memory["rerank_gate_last"]["heuristic_delta"], 70.0)
        self.assertEqual(engine._mcts_stats["rerank_gate_blocked"], 1)


if __name__ == "__main__":
    unittest.main()
