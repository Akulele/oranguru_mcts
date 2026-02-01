import unittest

from poke_env.battle import PokemonType, MoveCategory
from poke_env.battle.effect import Effect
from poke_env.data.gen_data import GenData

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, move_type=None, category=None, base_power=0):
        self.id = move_id
        self.type = move_type
        self.category = category
        self.base_power = base_power


class DummyFPMove:
    def __init__(self, *, move_id=None, name=None):
        self.id = move_id
        self.name = name


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
        current_hp_fraction=1.0,
    ):
        self.status = status
        self.effects = effects or {}
        self.type_1 = type_1
        self.type_2 = type_2
        self.tera_type = tera_type
        self.is_terastallized = is_terastallized
        self.types = types if types is not None else [type_1, type_2]
        self._data = data
        self.current_hp_fraction = current_hp_fraction


class DummyBattle:
    def __init__(self, *, available_moves, opponent_active_pokemon, active_pokemon=None):
        self.available_moves = available_moves
        self.opponent_active_pokemon = opponent_active_pokemon
        self.active_pokemon = active_pokemon


class EngineSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.engine = OranguruEnginePlayer(start_listening=False)

    def test_recovery_guard_avoids_heal_when_ko_available(self):
        active = DummyPokemon(status=None)
        opponent = DummyPokemon(status=None)
        battle = DummyBattle(
            available_moves=[DummyMove("recover")],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        active.current_hp_fraction = 0.4
        opponent.current_hp_fraction = 0.2
        self.engine.RECOVERY_KO_GUARD = True
        self.engine._estimate_best_reply_score = lambda *_: 0
        self.engine._estimate_matchup = lambda *_: 0.5
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        score = self.engine._heuristic_action_score(battle, "recover")
        self.assertEqual(score, 0.0)

    def test_fp_move_id_prefers_id_when_present(self):
        move = DummyFPMove(move_id="thunderwave", name="Thunder Wave")
        self.assertEqual(self.engine._fp_move_id(move), "thunderwave")

    def test_fp_move_id_falls_back_to_name(self):
        move = DummyFPMove(move_id=None, name="Sleep Powder")
        self.assertEqual(self.engine._fp_move_id(move), "sleeppowder")

    def test_action_dominance_scales_non_damaging(self):
        active = DummyPokemon(status=None)
        opponent = DummyPokemon(status=None, current_hp_fraction=0.4)
        battle = DummyBattle(
            available_moves=[
                DummyMove("recover", category=MoveCategory.STATUS),
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.ACTION_DOMINANCE = True
        self.engine.ACTION_DOMINANCE_THRESHOLD = 100.0
        self.engine.ACTION_DOMINANCE_SCALE = 0.1
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        policy = {"recover": 1.0, "earthquake": 0.5}
        scaled = self.engine._apply_action_dominance(policy, battle)
        self.assertLess(scaled["recover"], scaled["earthquake"])


if __name__ == "__main__":
    unittest.main()
