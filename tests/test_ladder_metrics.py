import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.ladder_metrics import (
    LadderMetricsLogger,
    expected_score,
    extract_ladder_ratings,
    parse_rating_transitions_from_text,
)


class LadderMetricsTest(unittest.TestCase):
    def test_parse_rating_transition_html_arrow(self):
        text = "A_Jar_Of_Water's rating: 1242 &rarr; <strong>1266</strong>"
        parsed = parse_rating_transitions_from_text(text)
        row = parsed["ajarofwater"]
        self.assertEqual(row.pre, 1242)
        self.assertEqual(row.post, 1266)
        self.assertEqual(row.delta, 24)

    def test_expected_score_even_rating(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)

    def test_extract_ladder_ratings_from_battle_observations(self):
        obs = SimpleNamespace(events=[[
            "",
            "raw",
            "Bot Name's rating: 1500 -> 1516\nOpponent's rating: 1500 -> 1484",
        ]])
        battle = SimpleNamespace(
            player_username="Bot Name",
            opponent_username="Opponent",
            _observations={1: obs},
            rating=None,
            opponent_rating=None,
        )
        ratings = extract_ladder_ratings(battle)
        self.assertEqual(ratings["player_rating_pre"], 1500)
        self.assertEqual(ratings["player_rating_post"], 1516)
        self.assertEqual(ratings["player_rating_delta"], 16)
        self.assertEqual(ratings["opponent_rating_pre"], 1500)
        self.assertEqual(ratings["opponent_rating_post"], 1484)

    def test_logger_writes_rating_residual(self):
        obs = SimpleNamespace(events=[[
            "Bot's rating: 1500 -> 1516\nOpp's rating: 1500 -> 1484",
        ]])
        battle = SimpleNamespace(
            won=True,
            lost=False,
            battle_tag="battle-gen9randombattle-1",
            player_username="Bot",
            opponent_username="Opp",
            _observations={1: obs},
            rating=None,
            opponent_rating=None,
            turn=12,
            team={},
            opponent_team={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.jsonl"
            logger = LadderMetricsLogger(path, bot_version="test")
            payload = logger.log_battle(
                battle,
                account="Bot",
                player_type="oranguru_engine",
                battle_format="gen9randombattle",
            )
            self.assertEqual(payload["result"], "win")
            self.assertAlmostEqual(payload["expected_score"], 0.5)
            self.assertAlmostEqual(payload["rating_residual"], 0.5)
            saved = json.loads(path.read_text().strip())
            self.assertEqual(saved["bot_version"], "test")
            self.assertEqual(saved["player_rating_delta"], 16)


if __name__ == "__main__":
    unittest.main()
