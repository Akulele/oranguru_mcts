import unittest

from poke_env.battle import PokemonType
from poke_env.battle.effect import Effect
from poke_env.data.gen_data import GenData

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, move_type=None):
        self.id = move_id
        self.type = move_type


class DummyPokemon:
    def __init__(
        self,
        *,
        status=None,
        effects=None,
        type_1=None,
        type_2=None,
        tera_type=None,
        is_terastallized=False,
        types=None,
        data=None,
    ):
        self.status = status
        self.effects = effects or {}
        self.type_1 = type_1
        self.type_2 = type_2
        self.tera_type = tera_type
        self.is_terastallized = is_terastallized
        self.types = types if types is not None else [type_1, type_2]
        self._data = data


class DummyBattle:
    def __init__(self, *, available_moves, opponent_active_pokemon, active_pokemon=None):
        self.available_moves = available_moves
        self.opponent_active_pokemon = opponent_active_pokemon
        self.active_pokemon = active_pokemon


class EngineSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.engine = OranguruEnginePlayer(start_listening=False)

    def test_status_banned_choices_blocks_sleep_on_statused_target(self):
        opp = DummyPokemon(status="par")
        battle = DummyBattle(
            available_moves=[DummyMove("sleeppowder"), DummyMove("thunderwave")],
            opponent_active_pokemon=opp,
        )
        banned = self.engine._status_banned_choices(battle)
        self.assertIn("sleeppowder", banned)
        self.assertIn("thunderwave", banned)

    def test_status_banned_choices_allows_unstatused(self):
        opp = DummyPokemon(status=None)
        battle = DummyBattle(
            available_moves=[DummyMove("sleeppowder")],
            opponent_active_pokemon=opp,
        )
        banned = self.engine._status_banned_choices(battle)
        self.assertEqual(banned, set())

    def test_no_retreat_banned_when_effect_active(self):
        active = DummyPokemon(effects={Effect.NO_RETREAT: 1})
        battle = DummyBattle(
            available_moves=[DummyMove("noretreat")],
            opponent_active_pokemon=DummyPokemon(status=None),
            active_pokemon=active,
        )
        banned = self.engine._self_effect_banned_choices(battle)
        self.assertIn("noretreat", banned)

    def test_tera_defensive_types_used_for_multiplier(self):
        data = GenData.from_gen(9)
        opp = DummyPokemon(
            type_1=PokemonType.GRASS,
            type_2=None,
            tera_type=PokemonType.WATER,
            is_terastallized=True,
            data=data,
        )
        mult = self.engine._damage_multiplier_against(opp, PokemonType.FIRE)
        # Fire vs Water should be resisted (< 1.0), not super-effective vs Grass (> 1.0).
        self.assertLess(mult, 1.0)


if __name__ == "__main__":
    unittest.main()
