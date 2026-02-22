import unittest
from types import SimpleNamespace
from collections import defaultdict

from poke_env.battle import PokemonType, MoveCategory, SideCondition
from poke_env.battle.effect import Effect
from poke_env.data.gen_data import GenData

from src.players.oranguru_engine import OranguruEnginePlayer
import constants


class DummyMove:
    def __init__(
        self,
        move_id,
        move_type=None,
        category=None,
        base_power=0,
        boosts=None,
        target=None,
    ):
        self.id = move_id
        self.type = move_type
        self.category = category
        self.base_power = base_power
        self.boosts = boosts or {}
        self.target = target


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
        boosts=None,
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
        self.boosts = boosts or {}


class DummyBattle:
    def __init__(
        self,
        *,
        available_moves,
        opponent_active_pokemon,
        active_pokemon=None,
        available_switches=None,
    ):
        self.available_moves = available_moves
        self.opponent_active_pokemon = opponent_active_pokemon
        self.active_pokemon = active_pokemon
        self.available_switches = available_switches or []
        self.battle_tag = "test-battle"


class DummyResult:
    def __init__(self, choices):
        self.total_visits = sum(v for _, v in choices) or 1
        self.side_one = [
            SimpleNamespace(move_choice=choice, visits=visits)
            for choice, visits in choices
        ]


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

    def test_finish_pressure_scales_status_and_switch(self):
        active = DummyPokemon(status=None, current_hp_fraction=0.8)
        opponent = DummyPokemon(status=None, current_hp_fraction=0.2)
        battle = DummyBattle(
            available_moves=[
                DummyMove("recover", category=MoveCategory.STATUS),
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.FINISH_PRESSURE = True
        self.engine.FINISH_PRESSURE_THRESHOLD = 200.0
        self.engine.FINISH_PRESSURE_MAX_OPP_HP = 0.7
        self.engine.FINISH_PRESSURE_NON_DAMAGE_SCALE = 0.2
        self.engine.FINISH_PRESSURE_SWITCH_SCALE = 0.4
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        self.engine._estimate_best_reply_score = lambda *_: 0.0
        policy = {"recover": 1.0, "earthquake": 1.0, "switch skarmory": 1.0}
        scaled = self.engine._apply_finish_pressure(policy, battle)
        self.assertLess(scaled["recover"], scaled["earthquake"])
        self.assertLess(scaled["switch skarmory"], scaled["earthquake"])

    def test_deterministic_low_confidence_reranks_topk(self):
        active = DummyPokemon(status=None, current_hp_fraction=1.0)
        opponent = DummyPokemon(status=None, current_hp_fraction=1.0)
        battle = DummyBattle(
            available_moves=[
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
                DummyMove("thunderwave", category=MoveCategory.STATUS),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.DETERMINISTIC_RERANK = True
        self.engine.DETERMINISTIC_RERANK_TOPK = 2
        self.engine.DETERMINISTIC_RERANK_CONF = 0.99
        self.engine.DETERMINISTIC_RERANK_MARGIN = 0.99
        self.engine.ACTION_DOMINANCE = False
        self.engine.FINISH_PRESSURE = False
        self.engine.NO_PROGRESS_TURNS = 999
        self.engine._heuristic_action_score = lambda _battle, c: 100.0 if c == "thunderwave" else 1.0
        results = [(DummyResult([("earthquake", 51), ("thunderwave", 49)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "thunderwave")

    def test_hybrid_low_conf_prefers_heuristic_over_mcts_top(self):
        active = DummyPokemon(status=None, current_hp_fraction=1.0)
        opponent = DummyPokemon(status=None, current_hp_fraction=1.0)
        battle = DummyBattle(
            available_moves=[
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
                DummyMove("thunderwave", category=MoveCategory.STATUS),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.HYBRID_RULEBOT_LOWCONF = True
        self.engine.HYBRID_RULEBOT_CONF = 0.99
        self.engine.HYBRID_RULEBOT_MARGIN = 0.99
        self.engine.DETERMINISTIC_RERANK = False
        self.engine.ACTION_DOMINANCE = False
        self.engine.FINISH_PRESSURE = False
        self.engine.NO_PROGRESS_TURNS = 999
        self.engine._heuristic_action_score = lambda _battle, c: 100.0 if c == "thunderwave" else 1.0
        results = [(DummyResult([("earthquake", 55), ("thunderwave", 45)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "thunderwave")

    def test_setup_ko_guard_blocks_setup_when_ko_available(self):
        active = DummyPokemon(status=None, current_hp_fraction=0.8, boosts={})
        opponent = DummyPokemon(status=None, current_hp_fraction=0.4)
        battle = DummyBattle(
            available_moves=[
                DummyMove(
                    "calmmind",
                    category=MoveCategory.STATUS,
                    boosts={"spa": 1},
                    target="self",
                )
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.SETUP_KO_GUARD = True
        self.engine.SETUP_KO_THRESHOLD = 100.0
        self.engine.SETUP_KO_MAX_OPP_HP = 0.8
        self.engine.SETUP_REPLY_GUARD = 999.0
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        self.engine._estimate_best_reply_score = lambda *_: 0.0
        score = self.engine._heuristic_action_score(battle, "calmmind")
        self.assertEqual(score, 0.0)

    def test_hybrid_low_conf_ignores_nonpositive_heuristics(self):
        active = DummyPokemon(status=None, current_hp_fraction=1.0)
        opponent = DummyPokemon(status=None, current_hp_fraction=1.0)
        battle = DummyBattle(
            available_moves=[
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
                DummyMove("thunderwave", category=MoveCategory.STATUS),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.HYBRID_RULEBOT_LOWCONF = True
        self.engine.HYBRID_RULEBOT_CONF = 0.99
        self.engine.HYBRID_RULEBOT_MARGIN = 0.99
        self.engine.DETERMINISTIC_RERANK = False
        self.engine.ACTION_DOMINANCE = False
        self.engine.FINISH_PRESSURE = False
        self.engine.NO_PROGRESS_TURNS = 999
        self.engine._heuristic_action_score = lambda *_: 0.0
        results = [(DummyResult([("earthquake", 55), ("thunderwave", 45)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "earthquake")

    def test_det_damage_tiebreak_prefers_near_tied_damage_over_status(self):
        active = DummyPokemon(status=None, current_hp_fraction=1.0)
        opponent = DummyPokemon(status=None, current_hp_fraction=0.2)
        battle = DummyBattle(
            available_moves=[
                DummyMove("thunderwave", category=MoveCategory.STATUS),
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.DET_DAMAGE_TIEBREAK = True
        self.engine.DET_DAMAGE_TIEBREAK_RATIO = 0.85
        self.engine.DET_DAMAGE_TIEBREAK_KO_THRESHOLD = 100.0
        self.engine.HYBRID_RULEBOT_LOWCONF = False
        self.engine.DETERMINISTIC_RERANK = False
        self.engine.ACTION_DOMINANCE = False
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        results = [(DummyResult([("thunderwave", 55), ("earthquake", 50)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "earthquake")

    def test_map_side_conditions_clamps_negative_and_large_values(self):
        src = {
            SideCondition.SPIKES: 9,
            SideCondition.TOXIC_SPIKES: -2,
            SideCondition.REFLECT: 99,
            SideCondition.STEALTH_ROCK: 4,
        }
        dest = defaultdict(lambda: 0)
        self.engine._map_side_conditions(src, dest)
        self.assertEqual(dest[constants.SPIKES], 3)
        self.assertEqual(dest[constants.TOXIC_SPIKES], 0)
        self.assertEqual(dest[constants.REFLECT], 8)
        self.assertEqual(dest[constants.STEALTH_ROCK], 1)

    def test_sanitize_fp_side_conditions_clamps_toxic_count(self):
        battler = SimpleNamespace(side_conditions=defaultdict(lambda: 0))
        battler.side_conditions[constants.TOXIC_COUNT] = 1000
        battler.side_conditions[constants.TAILWIND] = -3
        self.engine._sanitize_fp_side_conditions(battler)
        self.assertEqual(battler.side_conditions[constants.TOXIC_COUNT], 15)
        self.assertEqual(battler.side_conditions[constants.TAILWIND], 0)

    def test_det_damage_tiebreak_does_not_trigger_if_not_near_tied(self):
        active = DummyPokemon(status=None, current_hp_fraction=1.0)
        opponent = DummyPokemon(status=None, current_hp_fraction=0.2)
        battle = DummyBattle(
            available_moves=[
                DummyMove("thunderwave", category=MoveCategory.STATUS),
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.DET_DAMAGE_TIEBREAK = True
        self.engine.DET_DAMAGE_TIEBREAK_RATIO = 0.95
        self.engine.DET_DAMAGE_TIEBREAK_KO_THRESHOLD = 100.0
        self.engine.HYBRID_RULEBOT_LOWCONF = False
        self.engine.DETERMINISTIC_RERANK = False
        self.engine.ACTION_DOMINANCE = False
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        results = [(DummyResult([("thunderwave", 80), ("earthquake", 50)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "thunderwave")

    def test_force_finish_blow_prefers_damaging_when_ko_window(self):
        active = DummyPokemon(status=None, current_hp_fraction=0.9)
        opponent = DummyPokemon(status=None, current_hp_fraction=0.3)
        battle = DummyBattle(
            available_moves=[
                DummyMove("thunderwave", category=MoveCategory.STATUS),
                DummyMove("earthquake", category=MoveCategory.PHYSICAL, base_power=100),
            ],
            opponent_active_pokemon=opponent,
            active_pokemon=active,
        )
        self.engine.MCTS_DETERMINISTIC = True
        self.engine.MCTS_DETERMINISTIC_EVAL_ONLY = False
        self.engine.FORCE_FINISH_BLOW = True
        self.engine.FORCE_FINISH_MAX_OPP_HP = 0.5
        self.engine.FORCE_FINISH_KO_THRESHOLD = 100.0
        self.engine.FORCE_FINISH_REPLY_GUARD = 999.0
        self.engine.DET_DAMAGE_TIEBREAK = False
        self.engine.HYBRID_RULEBOT_LOWCONF = False
        self.engine.DETERMINISTIC_RERANK = False
        self.engine.ACTION_DOMINANCE = False
        self.engine._estimate_best_damage_score = lambda *_: 500.0
        self.engine._estimate_best_reply_score = lambda *_: 0.0
        results = [(DummyResult([("thunderwave", 70), ("earthquake", 30)]), 1.0)]
        choice = self.engine._select_move_from_results(results, battle)
        self.assertEqual(choice, "earthquake")


if __name__ == "__main__":
    unittest.main()
