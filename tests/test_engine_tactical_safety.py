import unittest

from poke_env.battle import MoveCategory, PokemonType, SideCondition

from src.players.oranguru_engine import OranguruEnginePlayer


class DummyMove:
    def __init__(self, move_id, *, category=MoveCategory.STATUS, base_power=0, boosts=None, target=None):
        self.id = move_id
        self.category = category
        self.base_power = base_power
        self.boosts = boosts or {}
        self.target = target


class DummyPokemon:
    def __init__(
        self,
        *,
        species="dummy",
        status=None,
        current_hp_fraction=1.0,
        types=None,
        boosts=None,
        item=None,
        grounded=True,
    ):
        self.species = species
        self.status = status
        self.current_hp_fraction = current_hp_fraction
        self.types = types or []
        self.boosts = boosts or {}
        self.item = item
        self.grounded = grounded


class DummyBattle:
    def __init__(
        self,
        *,
        available_moves,
        active_pokemon,
        opponent_active_pokemon,
        available_switches=None,
        side_conditions=None,
        turn=1,
    ):
        self.available_moves = available_moves
        self.active_pokemon = active_pokemon
        self.opponent_active_pokemon = opponent_active_pokemon
        self.available_switches = available_switches or []
        self.side_conditions = side_conditions or {}
        self.force_switch = False
        self.battle_tag = "test-battle"
        self.turn = turn

    def is_grounded(self, mon):
        return bool(getattr(mon, "grounded", True))


class TacticalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.engine = OranguruEnginePlayer.__new__(OranguruEnginePlayer)
        self.engine.MOVE_SAFETY_GUARD = True
        self.engine.TACTICAL_KO_THRESHOLD = 220.0
        self.engine.STATUS_STALL_MAX = 2
        self.engine.PASSIVE_REPEAT_HIGH_HP_MAX = 0.70
        self.engine.SETUP_PRESSURE_REPLY = 160.0
        self.engine.SETUP_PRESSURE_HP_MAX = 0.55
        self.engine._battle_memory = {}
        self.engine._estimate_best_reply_score = lambda *_: 0.0
        self.engine._estimate_matchup = lambda *_: 0.0
        self.engine._side_hazard_pressure = lambda *_: 0.0

    def test_invalid_status_is_replaced_by_damage(self):
        active = DummyPokemon(species="volbeat", current_hp_fraction=0.9)
        opponent = DummyPokemon(species="drifblim", status="par", current_hp_fraction=0.6)
        t_wave = DummyMove("thunderwave", category=MoveCategory.STATUS)
        tbolt = DummyMove("thunderbolt", category=MoveCategory.SPECIAL, base_power=90)
        battle = DummyBattle(
            available_moves=[t_wave, tbolt],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        )
        self.engine._best_damaging_move = lambda *_: (tbolt, 120.0)
        choice = self.engine._apply_tactical_safety(battle, "thunderwave", active, opponent)
        self.assertEqual(choice, "thunderbolt")

    def test_setup_is_replaced_in_ko_window(self):
        active = DummyPokemon(species="palafin", current_hp_fraction=0.8)
        opponent = DummyPokemon(species="duraludon", current_hp_fraction=0.2)
        setup = DummyMove("bulkup", category=MoveCategory.STATUS, boosts={"atk": 1}, target="self")
        eq = DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100)
        battle = DummyBattle(
            available_moves=[setup, eq],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        )
        self.engine._best_damaging_move = lambda *_: (eq, 300.0)
        choice = self.engine._apply_tactical_safety(battle, "bulkup", active, opponent)
        self.assertEqual(choice, "earthquake")

    def test_strength_sap_vs_special_is_replaced(self):
        active = DummyPokemon(species="drifblim", current_hp_fraction=0.8)
        opponent = DummyPokemon(species="duraludon", current_hp_fraction=0.8)
        sap = DummyMove("strengthsap", category=MoveCategory.STATUS)
        shadow_ball = DummyMove("shadowball", category=MoveCategory.SPECIAL, base_power=80)
        battle = DummyBattle(
            available_moves=[sap, shadow_ball],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
        )
        self.engine._best_damaging_move = lambda *_: (shadow_ball, 120.0)
        self.engine._stat_estimation = lambda mon, stat: 90 if stat == "atk" else 130
        choice = self.engine._apply_tactical_safety(battle, "strengthsap", active, opponent)
        self.assertEqual(choice, "shadowball")

    def test_repeat_high_hp_recovery_is_replaced(self):
        active = DummyPokemon(species="blissey", current_hp_fraction=0.92)
        opponent = DummyPokemon(species="tinglu", current_hp_fraction=0.75)
        recover = DummyMove("recover", category=MoveCategory.STATUS)
        moonblast = DummyMove("moonblast", category=MoveCategory.SPECIAL, base_power=95)
        battle = DummyBattle(
            available_moves=[recover, moonblast],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            turn=8,
        )
        mem = self.engine._get_battle_memory(battle)
        mem.update(
            {
                "last_action": "move",
                "last_action_turn": 7,
                "last_move_id": "recover",
                "last_active_species": "blissey",
                "last_opponent_species": "tinglu",
                "same_move_repeat_streak": 1,
            }
        )
        self.engine._best_damaging_move = lambda *_: (moonblast, 170.0)
        self.engine._estimate_best_reply_score = lambda *_: 120.0
        choice = self.engine._apply_tactical_safety(battle, "recover", active, opponent)
        self.assertEqual(choice, "moonblast")

    def test_repeat_status_same_matchup_is_replaced(self):
        active = DummyPokemon(species="grimmsnarl", current_hp_fraction=0.85)
        opponent = DummyPokemon(species="tinglu", current_hp_fraction=0.88, types=[PokemonType.GROUND, PokemonType.DARK])
        t_wave = DummyMove("thunderwave", category=MoveCategory.STATUS)
        spirit_break = DummyMove("spiritbreak", category=MoveCategory.PHYSICAL, base_power=75)
        battle = DummyBattle(
            available_moves=[t_wave, spirit_break],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            turn=6,
        )
        mem = self.engine._get_battle_memory(battle)
        mem.update(
            {
                "last_action": "move",
                "last_action_turn": 5,
                "last_move_id": "thunderwave",
                "last_active_species": "grimmsnarl",
                "last_opponent_species": "tinglu",
                "same_move_repeat_streak": 1,
                "status_stall_streak": 1,
            }
        )
        self.engine._best_damaging_move = lambda *_: (spirit_break, 150.0)
        choice = self.engine._apply_tactical_safety(battle, "thunderwave", active, opponent)
        self.assertEqual(choice, "spiritbreak")

    def test_setup_under_pressure_is_replaced(self):
        active = DummyPokemon(species="gyarados", current_hp_fraction=0.45)
        opponent = DummyPokemon(species="raikou", current_hp_fraction=0.80)
        dance = DummyMove("dragondance", category=MoveCategory.STATUS, boosts={"atk": 1, "spe": 1}, target="self")
        eq = DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100)
        battle = DummyBattle(
            available_moves=[dance, eq],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            turn=9,
        )
        self.engine._best_damaging_move = lambda *_: (eq, 180.0)
        self.engine._estimate_best_reply_score = lambda *_: 200.0
        choice = self.engine._apply_tactical_safety(battle, "dragondance", active, opponent)
        self.assertEqual(choice, "earthquake")

    def test_hazard_ko_switch_uses_survivable_switch(self):
        active = DummyPokemon(species="uxie")
        opponent = DummyPokemon(species="coalossal")
        doomed = DummyPokemon(
            species="charizard",
            current_hp_fraction=0.49,
            types=[PokemonType.FIRE, PokemonType.FLYING],
            grounded=False,
        )
        safe = DummyPokemon(
            species="toxapex",
            current_hp_fraction=0.8,
            types=[PokemonType.WATER, PokemonType.POISON],
            grounded=True,
        )
        battle = DummyBattle(
            available_moves=[],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            available_switches=[doomed, safe],
            side_conditions={SideCondition.STEALTH_ROCK: 1},
        )
        self.engine._score_switch = lambda mon, *_: 10.0 if mon.species == "toxapex" else 1.0
        choice = self.engine._apply_tactical_safety(battle, "switch charizard", active, opponent)
        self.assertEqual(choice, "switch toxapex")


if __name__ == "__main__":
    unittest.main()
