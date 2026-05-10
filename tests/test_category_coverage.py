import ast
import json
import unittest
from pathlib import Path

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import ITEM_EFFECTS
from src.utils.damage_belief import _TYPE_BOOST_ITEMS


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


class CategoryCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moves = _load_json("data/moves.json")
        cls.items = _load_json("data/items.json")

    def test_recoil_move_set_matches_moves_data(self):
        expected = {move_id for move_id, entry in self.moves.items() if "recoil" in entry}
        # Showdown's current items data omits a recoil key for Chloroblast, but
        # in-game it consumes half the user's max HP after damage.
        expected.add("chloroblast")

        self.assertEqual(RuleBotPlayer.HIGH_RECOIL_MOVES, expected)

    def test_pivot_move_set_matches_moves_data(self):
        expected = {
            move_id
            for move_id, entry in self.moves.items()
            if entry.get("selfSwitch") or entry.get("selfSwitch") == "copyvolatile"
        }
        expected.discard("revivalblessing")

        self.assertEqual(RuleBotPlayer.PIVOT_MOVES, expected)

    def test_priority_move_set_includes_current_damaging_priority(self):
        expected = {
            move_id
            for move_id, entry in self.moves.items()
            if isinstance(entry.get("priority"), (int, float))
            and entry.get("priority") > 0
            and str(entry.get("category", "")).lower() != "status"
        }
        # Bide has positive priority in legacy data but is a lock/damage-return
        # move, not a normal priority attack to value like Aqua Jet.
        expected.discard("bide")

        self.assertEqual(RuleBotPlayer.PRIORITY_MOVES, expected)

    def test_safe_ko_crash_detection_covers_moves_data(self):
        source = (ROOT / "src/players/oranguru_decision.py").read_text()
        tree = ast.parse(source)
        crash_literal = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "move_id"
                and any(isinstance(op, ast.In) for op in node.ops)
                and isinstance(node.comparators[0], ast.Set)
            ):
                values = {
                    elt.value
                    for elt in node.comparators[0].elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
                if {"highjumpkick", "jumpkick"}.issubset(values):
                    crash_literal = values
                    break

        expected = {
            move_id
            for move_id, entry in self.moves.items()
            if entry.get("hasCrashDamage") or "crash" in entry
        }
        self.assertEqual(crash_literal, expected)

    def test_damage_calc_has_current_type_booster_items(self):
        expected = {
            item_id
            for item_id, entry in self.items.items()
            if entry.get("onBasePowerPriority") == 15
            and (
                "onPlate" in entry
                or item_id
                in {
                    "blackbelt",
                    "blackglasses",
                    "charcoal",
                    "dragonfang",
                    "fairyfeather",
                    "hardstone",
                    "magnet",
                    "metalcoat",
                    "miracleseed",
                    "mysticwater",
                    "nevermeltice",
                    "oddincense",
                    "poisonbarb",
                    "rockincense",
                    "roseincense",
                    "seaincense",
                    "sharpbeak",
                    "silkscarf",
                    "silverpowder",
                    "softsand",
                    "spelltag",
                    "twistedspoon",
                    "waveincense",
                }
            )
        }
        for item_id in expected:
            with self.subTest(item=item_id):
                self.assertIn(item_id, ITEM_EFFECTS)
                self.assertIn("type_boost", ITEM_EFFECTS[item_id])
                self.assertIn(item_id, _TYPE_BOOST_ITEMS)

    def test_damage_calc_has_generic_damage_boosters(self):
        for item_id in ("muscleband", "wiseglasses"):
            with self.subTest(item=item_id):
                self.assertIn(item_id, ITEM_EFFECTS)


if __name__ == "__main__":
    unittest.main()
