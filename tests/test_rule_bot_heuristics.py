import unittest
from types import SimpleNamespace

from poke_env.battle import PokemonType, MoveCategory
from poke_env.battle.field import Field
from poke_env.battle.effect import Effect
from poke_env.data.gen_data import GenData

from src.players.rule_bot import RuleBotPlayer


class DummyMove:
    def __init__(self, move_id, accuracy=100, move_type=None, boosts=None, category=MoveCategory.STATUS, base_power=0):
        self.id = move_id
        self.accuracy = accuracy
        self.type = move_type
        self.category = category
        self.boosts = boosts or {}
        self.base_power = base_power


class DummyPokemon:
    def __init__(
        self,
        *,
        stats=None,
        boosts=None,
        status=None,
        current_hp_fraction=1.0,
        type_1=None,
        type_2=None,
        tera_type=None,
        is_terastallized=False,
        types=None,
        effects=None,
        data=None,
    ):
        self.stats = stats or {}
        self.boosts = boosts or {}
        self.status = status
        self.current_hp_fraction = current_hp_fraction
        self.type_1 = type_1
        self.type_2 = type_2
        self.tera_type = tera_type
        self.is_terastallized = is_terastallized
        self.types = types if types is not None else [type_1, type_2]
        self.effects = effects or {}
        self._data = data


class RuleBotHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.bot = RuleBotPlayer(start_listening=False)
        self.bot.STATUS_KO_GUARD = True
        self.bot.STATUS_KO_THRESHOLD = 200.0

    def test_strength_sap_prefers_boosted_attackers(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 100, "def": 120, "spa": 100, "spd": 120, "spe": 80},
            current_hp_fraction=0.5,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 320, "atk": 200, "def": 120, "spa": 90, "spd": 110, "spe": 80},
            boosts={"atk": 2},
            current_hp_fraction=0.9,
            type_1=PokemonType.GRASS,
            data=data,
        )
        move = DummyMove("strengthsap", accuracy=100)
        score = self.bot._should_use_status_move(move, active, opponent, SimpleNamespace(available_moves=[]))
        self.assertGreaterEqual(score, 220.0)

    def test_status_moves_blocked_when_target_already_statused(self):
        active = DummyPokemon(stats={"hp": 300, "atk": 100, "def": 120, "spa": 100, "spd": 120, "spe": 80})
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 110, "spa": 120, "spd": 110, "spe": 80},
            status="par",
        )
        move = DummyMove("sleeppowder", accuracy=75)
        score = self.bot._should_use_status_move(move, active, opponent, SimpleNamespace())
        self.assertEqual(score, 0.0)

    def test_tera_defensive_types_used_in_type_checks(self):
        data = GenData.from_gen(9)
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 110, "spa": 120, "spd": 110, "spe": 80},
            type_1=PokemonType.GRASS,
            type_2=None,
            tera_type=PokemonType.WATER,
            is_terastallized=True,
            data=data,
        )
        self.assertTrue(self.bot._opponent_has_type(opponent, "water"))
        self.assertFalse(self.bot._opponent_has_type(opponent, "grass"))

    def test_tera_defensive_types_with_string_flag(self):
        data = GenData.from_gen(9)
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 110, "spa": 120, "spd": 110, "spe": 80},
            type_1=PokemonType.GRASS,
            type_2=None,
            tera_type=PokemonType.WATER,
            is_terastallized=False,
            data=data,
        )
        opponent.terastallized = "water"
        self.assertTrue(self.bot._opponent_has_type(opponent, "water"))

    def test_tera_matchup_penalizes_bad_switch(self):
        data = GenData.from_gen(9)
        attacker = DummyPokemon(
            stats={"hp": 300, "atk": 160, "def": 120, "spa": 120, "spd": 120, "spe": 95},
            type_1=PokemonType.FIRE,
            type_2=None,
            current_hp_fraction=1.0,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 320, "atk": 120, "def": 110, "spa": 120, "spd": 110, "spe": 80},
            type_1=PokemonType.GRASS,
            type_2=None,
            tera_type=PokemonType.WATER,
            is_terastallized=True,
            current_hp_fraction=1.0,
            data=data,
        )
        matchup = self.bot._estimate_matchup(attacker, opponent)
        self.assertLess(matchup, 0.0)

    def test_psychic_noise_boosts_vs_recovery(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 80, "def": 120, "spa": 140, "spd": 120, "spe": 90},
            type_1=PokemonType.PSYCHIC,
            current_hp_fraction=0.8,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 320, "atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 80},
            type_1=PokemonType.DRAGON,
            current_hp_fraction=0.9,
            data=data,
        )
        move = DummyMove(
            "psychicnoise",
            accuracy=100,
            move_type=PokemonType.PSYCHIC,
            category=MoveCategory.SPECIAL,
            base_power=75,
        )
        self.bot._opponent_is_stallish = lambda *_: False
        self.bot._damage_multiplier_against = lambda *_: 1.0
        self.bot._opponent_known_move_ids = lambda *_: set()
        battle = SimpleNamespace(battle_tag="test", side_conditions=None)
        baseline = self.bot._calculate_move_score(move, active, opponent, battle)
        self.bot._opponent_known_move_ids = lambda *_: {"recover"}
        boosted = self.bot._calculate_move_score(move, active, opponent, battle)
        self.assertGreater(boosted, baseline)

    def test_canonicalize_move_id_handles_prefix(self):
        self.assertEqual(self.bot._canonicalize_move_id("Move: Thunder Wave"), "thunderwave")

    def test_trick_room_prefers_slower(self):
        data = GenData.from_gen(9)
        slow = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 80, "spd": 120, "spe": 50},
            type_1=PokemonType.WATER,
            current_hp_fraction=1.0,
            data=data,
        )
        fast = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 80, "spd": 120, "spe": 120},
            type_1=PokemonType.WATER,
            current_hp_fraction=1.0,
            data=data,
        )
        battle = SimpleNamespace(fields={Field.TRICK_ROOM: 3})
        self.bot._current_battle = battle
        matchup = self.bot._estimate_matchup(slow, fast)
        self.assertGreater(matchup, 0.0)

    def test_status_moves_avoided_when_ko_available(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 80},
            current_hp_fraction=0.8,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 80},
            current_hp_fraction=0.2,
            data=data,
        )
        battle = SimpleNamespace(available_moves=[DummyMove("sludgebomb")])
        self.bot._estimate_best_damage_score = lambda *_: 500.0
        score = self.bot._should_use_status_move(DummyMove("toxic"), active, opponent, battle)
        self.assertEqual(score, 0.0)

    def test_no_retreat_not_recommended_twice(self):
        active = DummyPokemon(
            stats={"hp": 300, "atk": 150, "def": 120, "spa": 80, "spd": 110, "spe": 90},
            effects={Effect.NO_RETREAT: 1},
        )
        opponent = DummyPokemon(stats={"hp": 300, "atk": 120, "def": 110, "spa": 100, "spd": 100, "spe": 80})
        move = DummyMove("noretreat", accuracy=100, boosts={"atk": 1, "def": 1, "spe": 1})
        self.assertFalse(self.bot._should_setup_move(move, active, opponent))


if __name__ == "__main__":
    unittest.main()
