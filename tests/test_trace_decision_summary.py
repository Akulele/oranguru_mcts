import tempfile
import unittest
from pathlib import Path
import json

from evaluation.summarize_trace_decisions import summarize


class TraceDecisionSummaryTests(unittest.TestCase):
    def test_summarizes_paths_and_non_top1_loss_samples(self):
        rows = [
            {
                "battle_id": "b1",
                "turn": 1,
                "value_target": -1.0,
                "selection_path": "rerank",
                "chosen_choice": "recover",
                "policy_confidence": 0.2,
                "top_actions": [
                    {"choice": "earthquake", "kind": "attack", "score": 0.6, "heuristic_score": 20.0},
                    {"choice": "recover", "kind": "status", "score": 0.3, "heuristic_score": 60.0},
                ],
                "recovery_window": {"reason": "take_recovery"},
            },
            {
                "battle_id": "b2",
                "turn": 2,
                "value_target": 1.0,
                "selection_path": "mcts",
                "chosen_choice": "earthquake",
                "policy_confidence": 0.7,
                "top_actions": [
                    {"choice": "earthquake", "kind": "attack", "score": 0.7, "heuristic_score": 20.0},
                    {"choice": "recover", "kind": "status", "score": 0.2, "heuristic_score": 60.0},
                ],
                "finish_blow": {"reason": "chosen_damaging"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = summarize([str(path)], sample_limit=10)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["battles"], 2)
        self.assertEqual(summary["path_counts"], {"rerank": 1, "mcts": 1})
        self.assertEqual(summary["by_path"]["rerank"]["losses"], 1)
        self.assertEqual(summary["by_path"]["rerank"]["non_top1_rate"], 1.0)
        self.assertEqual(summary["window_reasons"]["recovery_window"], {"take_recovery": 1})
        self.assertEqual(summary["window_reasons"]["finish_blow"], {"chosen_damaging": 1})
        self.assertEqual(summary["non_top1_loss_samples"][0]["chosen"], "recover")
        self.assertEqual(summary["rerank_loss_samples"][0]["top1"], "earthquake")


if __name__ == "__main__":
    unittest.main()
