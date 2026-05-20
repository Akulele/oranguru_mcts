import unittest
from types import SimpleNamespace

from poke_env.battle import PokemonType, MoveCategory
from poke_env.battle.field import Field
from poke_env.battle.effect import Effect
from poke_env.data.gen_data import GenData

from src.players.rule_bot import RuleBotPlayer
from src.utils.damage_calc import get_type_effectiveness


class DummyMove:
    def __init__(
        self,
        move_id,
        accuracy=100,
        move_type=None,
        boosts=None,
        category=MoveCategory.STATUS,
        base_power=0,
        target="normal",
    ):
        self.id = move_id
        self.accuracy = accuracy
        self.type = move_type
        self.category = category
        self.boosts = boosts or {}
        self.base_power = base_power
        self.target = target


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
        species="dummy",
        level=None,
        base_stats=None,
        item=None,
    ):
        self.species = species
        self.stats = stats or {}
        self.base_stats = base_stats or {}
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
        self.moves = {}
        self.level = level
        self.item = item

    def damage_multiplier(self, move_type):
        if move_type is None:
            return 1.0
        move_id = move_type.name.lower() if hasattr(move_type, "name") else str(move_type).lower()
        terastallized = getattr(self, "terastallized", None)
        if terastallized:
            def_types = [str(terastallized).lower()]
        elif getattr(self, "is_terastallized", False) and getattr(self, "tera_type", None) is not None:
            tera_type = self.tera_type
            def_types = [tera_type.name.lower() if hasattr(tera_type, "name") else str(tera_type).lower()]
        else:
            def_types = [
                (t.name.lower() if hasattr(t, "name") else str(t).lower())
                for t in (self.types or [])
                if t is not None
            ]
        return get_type_effectiveness(move_id, def_types)


class SwitchChurnHeuristicTests(unittest.TestCase):
    def _bot_with_memory(self, mem, *, damage=0.0, hazard_pressure=0.0):
        bot = RuleBotPlayer.__new__(RuleBotPlayer)
        bot.ANTI_SWITCH_CHURN = True
        bot.SWITCH_CHURN_MIN_STREAK = 1
        bot.SWITCH_CHURN_MIN_DAMAGE = 40.0
        bot.SWITCH_CHURN_MIN_HAZARD_HP = 0.08
        bot._get_battle_memory = lambda _battle: mem
        bot._estimate_best_damage_score = lambda *_args: damage
        bot._side_hazard_pressure = lambda *_args: hazard_pressure
        bot._estimate_matchup = lambda *_args: 0.0
        bot._estimate_best_reply_score = lambda *_args: 0.0
        bot._score_switch = lambda *_args: 0.0
        bot._get_effective_speed = lambda *_args: 100
        bot._opponent_is_set_up = lambda *_args: False
        return bot

    def _same_board_battle(self, move):
        active = DummyPokemon(current_hp_fraction=0.8)
        active.species = "heracross"
        opponent = DummyPokemon(current_hp_fraction=0.7)
        opponent.species = "grumpig"
        return SimpleNamespace(
            force_switch=False,
            available_switches=[SimpleNamespace(species="toxapex", current_hp_fraction=0.8)],
            available_moves=[move],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            battle_tag="switch-churn-test",
        )

    def test_switch_churn_detects_repeated_switch_into_same_board_with_damage_available(self):
        mem = {
            "self_switch_streak": 1,
            "last_opponent_species": "grumpig",
            "last_opponent_hp": 0.7,
        }
        bot = self._bot_with_memory(mem, damage=75.0)
        battle = self._same_board_battle(DummyMove("megahorn", category=MoveCategory.PHYSICAL, base_power=120))

        self.assertTrue(bot._is_switch_churn_risk(battle))

    def test_switch_churn_detects_repeated_switch_when_hazards_punish_cycling(self):
        mem = {
            "self_switch_streak": 1,
            "last_opponent_species": "grumpig",
            "last_opponent_hp": 0.7,
        }
        bot = self._bot_with_memory(mem, damage=5.0, hazard_pressure=0.125)
        battle = self._same_board_battle(DummyMove("tackle", category=MoveCategory.PHYSICAL, base_power=40))

        self.assertTrue(bot._is_switch_churn_risk(battle))


class RuleBotHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.bot = RuleBotPlayer.__new__(RuleBotPlayer)
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

    def test_strength_sap_blocked_by_revealed_sap_sipper(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 100, "def": 120, "spa": 100, "spd": 120, "spe": 80},
            current_hp_fraction=0.5,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 320, "atk": 200, "def": 120, "spa": 90, "spd": 110, "spe": 80},
            boosts={"attack": 4},
            current_hp_fraction=0.9,
            type_1=PokemonType.WATER,
            data=data,
        )
        opponent.ability = "Sap Sipper"
        move = DummyMove("strengthsap", accuracy=100, move_type=PokemonType.GRASS)

        score = self.bot._should_use_status_move(move, active, opponent, SimpleNamespace(available_moves=[]))

        self.assertEqual(score, 0.0)

    def test_status_moves_blocked_when_target_already_statused(self):
        active = DummyPokemon(stats={"hp": 300, "atk": 100, "def": 120, "spa": 100, "spd": 120, "spe": 80})
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 110, "spa": 120, "spd": 110, "spe": 80},
            status="par",
        )
        move = DummyMove("sleeppowder", accuracy=75)
        score = self.bot._should_use_status_move(move, active, opponent, SimpleNamespace())
        self.assertEqual(score, 0.0)

    def test_status_pivot_absorber_blocks_repeated_glare(self):
        self.bot.STATUS_PIVOT_ABSORBER_GUARD = True
        self.bot.STATUS_PIVOT_ABSORBER_TURNS = 8
        active = DummyPokemon(
            species="dudunsparce",
            stats={"hp": 343, "atk": 214, "def": 180, "spa": 189, "spd": 172, "spe": 139},
        )
        donphan = DummyPokemon(
            species="donphan",
            stats={"hp": 300, "atk": 220, "def": 220, "spa": 80, "spd": 120, "spe": 180},
            status=None,
        )
        slither = DummyPokemon(
            species="slitherwing",
            stats={"hp": 300, "atk": 220, "def": 160, "spa": 80, "spd": 160, "spe": 200},
            status="brn",
        )
        battle = SimpleNamespace(
            battle_tag="status-pivot",
            turn=31,
            opponent_active_pokemon=slither,
            opponent_team={"donphan": donphan, "slitherwing": slither},
        )
        mem = self.bot._get_battle_memory(battle)
        mem["last_action"] = "move"
        mem["last_action_turn"] = 30
        mem["last_move_id"] = "glare"
        mem["last_move_category"] = "status"
        mem["last_opponent_species"] = "Donphan"

        self.bot._update_status_pivot_absorber_memory(battle)

        battle.turn = 32
        battle.opponent_active_pokemon = donphan
        score = self.bot._should_use_status_move(DummyMove("glare"), active, donphan, battle)

        self.assertEqual(score, 0.0)

    def test_spore_blocked_by_grass_target(self):
        active = DummyPokemon(
            species="amoonguss",
            stats={"hp": 300, "atk": 100, "def": 150, "spa": 180, "spd": 160, "spe": 60},
        )
        opponent = DummyPokemon(
            species="sceptile",
            stats={"hp": 260, "atk": 190, "def": 120, "spa": 200, "spd": 140, "spe": 240},
            type_1=PokemonType.GRASS,
        )

        score = self.bot._should_use_status_move(DummyMove("spore"), active, opponent, SimpleNamespace())

        self.assertEqual(score, 0.0)

    def test_known_encore_threat_blocks_setup(self):
        self.bot.ENCORE_SETUP_GUARD = True
        active = DummyPokemon(
            species="ironcrown",
            stats={"hp": 300, "atk": 120, "def": 170, "spa": 220, "spd": 170, "spe": 180},
        )
        opponent = DummyPokemon(
            species="whimsicott",
            stats={"hp": 250, "atk": 100, "def": 140, "spa": 170, "spd": 150, "spe": 260},
        )
        opponent.moves = {"encore": DummyMove("encore")}
        calm_mind = DummyMove("calmmind", boosts={"spa": 1, "spd": 1}, target="self")

        self.assertFalse(self.bot._should_setup_move(calm_mind, active, opponent))

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

    def test_move_score_respects_tera_type_and_long_boost_alias(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 200, "def": 120, "spa": 100, "spd": 120, "spe": 95},
            type_1=PokemonType.FIRE,
            current_hp_fraction=1.0,
            data=data,
        )
        normal_gogoat = DummyPokemon(
            stats={"hp": 320, "atk": 150, "def": 100, "spa": 90, "spd": 100, "spe": 80},
            type_1=PokemonType.GRASS,
            current_hp_fraction=1.0,
            data=data,
        )
        tera_water_boosted = DummyPokemon(
            stats={"hp": 320, "atk": 150, "def": 100, "spa": 90, "spd": 100, "spe": 80},
            boosts={"defense": 4},
            type_1=PokemonType.GRASS,
            tera_type=PokemonType.WATER,
            is_terastallized=True,
            current_hp_fraction=1.0,
            data=data,
        )
        pyro_ball = DummyMove(
            "pyroball",
            move_type=PokemonType.FIRE,
            category=MoveCategory.PHYSICAL,
            base_power=120,
        )

        normal_score = self.bot._calculate_move_score(pyro_ball, active, normal_gogoat, None)
        tera_score = self.bot._calculate_move_score(pyro_ball, active, tera_water_boosted, None)

        self.assertLess(tera_score, normal_score * 0.25)

    def test_variable_base_power_scores_grass_knot(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            species="delphox",
            stats={"hp": 260, "atk": 140, "def": 150, "spa": 260, "spd": 220, "spe": 260},
            type_1=PokemonType.GRASS,
            tera_type=PokemonType.GRASS,
            is_terastallized=True,
            current_hp_fraction=0.5,
            data=data,
        )
        opponent = DummyPokemon(
            species="basculinbluestriped",
            level=84,
            current_hp_fraction=0.3,
            type_1=PokemonType.WATER,
            data=data,
        )
        grass_knot = DummyMove(
            "grassknot",
            move_type=PokemonType.GRASS,
            category=MoveCategory.SPECIAL,
            base_power=0,
        )

        score = self.bot._calculate_move_score(grass_knot, active, opponent, SimpleNamespace())

        self.assertGreater(score, 150.0)

    def test_switch_out_blocked_when_faster_ko_available(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            species="delphox",
            stats={"hp": 260, "atk": 140, "def": 150, "spa": 260, "spd": 220, "spe": 260},
            type_1=PokemonType.GRASS,
            tera_type=PokemonType.GRASS,
            is_terastallized=True,
            current_hp_fraction=0.48,
            data=data,
        )
        opponent = DummyPokemon(
            species="basculinbluestriped",
            level=84,
            current_hp_fraction=0.25,
            type_1=PokemonType.WATER,
            data=data,
        )
        grass_knot = DummyMove(
            "grassknot",
            move_type=PokemonType.GRASS,
            category=MoveCategory.SPECIAL,
            base_power=0,
        )
        battle = SimpleNamespace(
            battle_tag="outspeed-ko",
            available_moves=[grass_knot],
            available_switches=[DummyPokemon(species="goodra", current_hp_fraction=1.0, data=data)],
            active_pokemon=active,
            opponent_active_pokemon=opponent,
            fields={},
            force_switch=False,
        )
        self.bot._estimate_matchup = lambda *_: -2.0
        self.bot._estimate_best_reply_score = lambda *_: 500.0
        self.bot._score_switch = lambda *_: 10.0

        self.assertFalse(self.bot._should_switch_out(battle))

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

    def test_trick_room_discourages_thunder_wave(self):
        data = GenData.from_gen(9)
        active = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 80, "spd": 120, "spe": 45},
            type_1=PokemonType.WATER,
            current_hp_fraction=1.0,
            data=data,
        )
        opponent = DummyPokemon(
            stats={"hp": 300, "atk": 120, "def": 120, "spa": 80, "spd": 120, "spe": 120},
            type_1=PokemonType.WATER,
            current_hp_fraction=1.0,
            data=data,
        )
        battle = SimpleNamespace(fields={Field.TRICK_ROOM: 3}, available_moves=[DummyMove("thunderwave")])
        self.bot._current_battle = battle
        score = self.bot._should_use_status_move(DummyMove("thunderwave"), active, opponent, battle)
        self.assertEqual(score, 0.0)

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
