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

    def test_parse_showdown_raw_rating_prefix(self):
        text = "|raw|A_Jar_Of_Water's rating: 1499 &rarr; <strong>1518</strong>"
        parsed = parse_rating_transitions_from_text(text)
        self.assertIn("ajarofwater", parsed)
        self.assertEqual(parsed["ajarofwater"].delta, 19)

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

    def test_logger_patches_late_rating_lines(self):
        battle = SimpleNamespace(
            won=False,
            lost=True,
            battle_tag="battle-gen9randombattle-2",
            player_username="Bot",
            opponent_username="Opp",
            _observations={},
            rating=None,
            opponent_rating=None,
            turn=20,
            team={},
            opponent_team={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.jsonl"
            logger = LadderMetricsLogger(path, bot_version="test")
            logger.log_battle(
                battle,
                account="Bot",
                player_type="oranguru_engine",
                battle_format="gen9randombattle",
            )

            updated = logger.update_battle_ratings_from_text(
                battle_tag="battle-gen9randombattle-2",
                text="Bot's rating: 1510 -> 1494\nOpp's rating: 1510 -> 1526",
            )

            self.assertTrue(updated)
            saved = json.loads(path.read_text().strip())
            self.assertEqual(saved["player_rating_pre"], 1510)
            self.assertEqual(saved["player_rating_delta"], -16)
            self.assertAlmostEqual(saved["expected_score"], 0.5)
            self.assertAlmostEqual(saved["rating_residual"], -0.5)

    def test_logger_applies_pending_rating_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.jsonl"
            logger = LadderMetricsLogger(path, bot_version="test")
            logger.update_battle_ratings_from_text(
                battle_tag="battle-gen9randombattle-5",
                text="Bot's rating: 1500 -> 1516\nOpp's rating: 1500 -> 1484",
            )
            battle = SimpleNamespace(
                won=True,
                lost=False,
                battle_tag="battle-gen9randombattle-5",
                player_username="Bot",
                opponent_username="Opp",
                _observations={},
                rating=None,
                opponent_rating=None,
                turn=11,
                team={},
                opponent_team={},
            )
            logger.log_battle(
                battle,
                account="Bot",
                player_type="oranguru_engine",
                battle_format="gen9randombattle",
            )

            saved = json.loads(path.read_text().strip())
            self.assertEqual(saved["player_rating_delta"], 16)
            self.assertAlmostEqual(saved["rating_residual"], 0.5)

    def test_late_rating_patch_preserves_existing_log_rows(self):
        battle = SimpleNamespace(
            won=True,
            lost=False,
            battle_tag="battle-gen9randombattle-3",
            player_username="Bot",
            opponent_username="Opp",
            _observations={},
            rating=None,
            opponent_rating=None,
            turn=5,
            team={},
            opponent_team={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.jsonl"
            path.write_text('{"schema_version":1,"result":"loss"}\n', encoding="utf-8")
            logger = LadderMetricsLogger(path, bot_version="test")
            logger.log_battle(
                battle,
                account="Bot",
                player_type="oranguru_engine",
                battle_format="gen9randombattle",
            )
            logger.update_battle_ratings_from_text(
                battle_tag="battle-gen9randombattle-3",
                text="Bot's rating: 1500 -> 1516\nOpp's rating: 1500 -> 1484",
            )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["result"], "loss")
            self.assertEqual(json.loads(lines[1])["player_rating_delta"], 16)

    def test_logger_patches_from_battle_observations(self):
        battle = SimpleNamespace(
            won=True,
            lost=False,
            battle_tag="battle-gen9randombattle-4",
            player_username="Bot",
            opponent_username="Opp",
            _observations={},
            rating=None,
            opponent_rating=None,
            turn=9,
            team={},
            opponent_team={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ladder.jsonl"
            logger = LadderMetricsLogger(path, bot_version="test")
            logger.log_battle(
                battle,
                account="Bot",
                player_type="oranguru_engine",
                battle_format="gen9randombattle",
            )

            battle._observations = {
                1: SimpleNamespace(events=[[
                    "Bot's rating: 1500 -> 1516\nOpp's rating: 1500 -> 1484",
                ]])
            }
            updated = logger.update_battle_ratings_from_battle(battle)

            self.assertTrue(updated)
            saved = json.loads(path.read_text().strip())
            self.assertEqual(saved["player_rating_delta"], 16)
            self.assertAlmostEqual(saved["rating_residual"], 0.5)


if __name__ == "__main__":
    unittest.main()
