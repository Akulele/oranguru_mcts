#!/usr/bin/env python3
"""
SmartHeuristicsPlayer - An improved expert that beats SimpleHeuristics.

Key improvements over SimpleHeuristics:
1. Better switch scoring with type matchups
2. Smarter setup detection
3. Better damage estimation
4. Proper hazard/status timing
"""

from poke_env.player import Player
from poke_env.battle import Pokemon, Move, AbstractBattle, MoveCategory


class SmartHeuristicsPlayer(Player):
    """Smarter heuristic player that beats SimpleHeuristics."""

    # Move categories
    SETUP_MOVES = {'swordsdance', 'nastyplot', 'dragondance', 'calmmind',
                   'bulkup', 'irondefense', 'agility', 'shellsmash',
                   'quiverdance', 'coil', 'curse', 'workup', 'growth',
                   'honeclaws', 'tailglow', 'geomancy', 'bellydrum'}
    STATUS_MOVES = {'thunderwave', 'toxic', 'willowisp', 'spore', 'sleeppowder',
                    'stunspore', 'glare', 'nuzzle', 'yawn', 'hypnosis'}
    HAZARD_SETUP = {'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'}
    HAZARD_CLEAR = {'defog', 'rapidspin', 'tidyup', 'mortalspin', 'courtchange'}
    PIVOT_MOVES = {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport',
                   'batonpass', 'shedtail'}

    def choose_move(self, battle: AbstractBattle):
        """Main entry point."""
        result = self.choose_singles_move(battle)
        if result and isinstance(result, tuple):
            return result[0]  # Return the order
        # Fallback for DefaultBattleOrder or None
        return self.choose_random_move(battle)

    def choose_singles_move(self, battle: AbstractBattle):
        """Returns (SingleBattleOrder, score) tuple."""
        # Force switch handling
        if battle.force_switch:
            switch = self._get_best_switch(battle)
            if switch:
                return (self.create_order(switch), 0)
            return None  # Will trigger fallback

        moves = battle.available_moves
        switches = battle.available_switches

        if not moves:
            if switches:
                return (self.create_order(switches[0]), 0)
            return None

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            if moves:
                return (self.create_order(moves[0]), 0)
            return None

        # Calculate matchup score (-1 to 1, positive = we're favored)
        matchup = self._calc_matchup(active, opponent, battle)

        # Priority 1: Guaranteed KO
        ko_move = self._find_ko_move(moves, active, opponent)
        if ko_move:
            return (self.create_order(ko_move), 1.0)

        # Priority 2: We're in a bad matchup - consider switching
        if matchup < -0.3 and switches:
            best_switch = self._get_best_switch(battle)
            if best_switch and self._calc_matchup(best_switch, opponent, battle) > matchup + 0.4:
                return (self.create_order(best_switch), 0.6)

        # Priority 3: Good setup opportunity
        if matchup > 0.2 and active.current_hp_fraction > 0.7:
            setup_move = self._find_setup_move(moves, active)
            if setup_move:
                return (self.create_order(setup_move), 0.8)

        # Priority 4: Hazard control
        hazard_move = self._find_hazard_move(moves, battle)
        if hazard_move:
            return (self.create_order(hazard_move), 0.5)

        # Priority 5: Pivot if we have it and opponent is threatening
        if matchup < 0 and switches:
            pivot_move = self._find_pivot_move(moves)
            if pivot_move:
                return (self.create_order(pivot_move), 0.4)

        # Priority 6: Status move on predicted switch
        if opponent.current_hp_fraction < 0.35:
            status_move = self._find_status_move(moves, opponent)
            if status_move:
                return (self.create_order(status_move), 0.3)

        # Priority 7: Best damage move
        best_move = self._find_best_damage_move(moves, active, opponent)
        if best_move:
            return (self.create_order(best_move), 0)

        # Fallback
        if moves:
            return (self.create_order(moves[0]), -0.5)
        return None

    def _calc_matchup(self, mon: Pokemon, opponent: Pokemon, battle) -> float:
        """Calculate matchup score from -1 (bad) to 1 (good)."""
        if not mon or not opponent:
            return 0.0

        score = 0.0

        # Type effectiveness: how much damage can we deal?
        our_best_eff = 1.0
        for move in battle.available_moves if hasattr(battle, 'available_moves') else []:
            try:
                eff = opponent.damage_multiplier(move)
                our_best_eff = max(our_best_eff, eff)
            except:
                pass

        # How much damage can opponent deal to us?
        their_worst_eff = 1.0
        for t in (opponent.types or []):
            try:
                # Create a pseudo-move of that type to check effectiveness
                eff = mon.damage_multiplier(Move(t.name.lower()))
                their_worst_eff = max(their_worst_eff, eff)
            except:
                pass

        # Score based on effectiveness
        if our_best_eff >= 2.0:
            score += 0.4
        elif our_best_eff >= 1.5:
            score += 0.2

        if their_worst_eff >= 2.0:
            score -= 0.4
        elif their_worst_eff >= 1.5:
            score -= 0.2

        # Speed advantage
        my_speed = mon.stats.get('spe', 100) if mon.stats else 100
        opp_speed = opponent.stats.get('spe', 100) if opponent.stats else 100
        if my_speed and opp_speed:
            if my_speed > opp_speed * 1.2:
                score += 0.2
            elif opp_speed > my_speed * 1.2:
                score -= 0.1

        # HP advantage
        hp_diff = mon.current_hp_fraction - opponent.current_hp_fraction
        score += hp_diff * 0.3

        return max(-1.0, min(1.0, score))

    def _find_ko_move(self, moves, user: Pokemon, target: Pokemon) -> Move | None:
        """Find a move that will KO the target."""
        if not target:
            return None

        target_hp = target.current_hp if target.current_hp else 100

        for move in moves:
            damage = self._estimate_damage(move, user, target)
            if damage >= target_hp * 0.95:
                return move
        return None

    def _estimate_damage(self, move: Move, user: Pokemon, target: Pokemon) -> float:
        """Estimate damage from a move."""
        if not move or not user or not target:
            return 0

        try:
            if move.category == MoveCategory.STATUS:
                return 0
        except:
            return 0

        base_power = move.base_power or 0
        if not base_power:
            return 0

        try:
            type_mult = target.damage_multiplier(move)
        except:
            type_mult = 1.0

        try:
            stab = 1.5 if move.type and user.types and move.type in user.types else 1.0
        except:
            stab = 1.0

        try:
            if move.category == MoveCategory.PHYSICAL:
                atk = (user.stats.get('atk') or 100) if user.stats else 100
                dfn = (target.stats.get('def') or 100) if target.stats else 100
            else:
                atk = (user.stats.get('spa') or 100) if user.stats else 100
                dfn = (target.stats.get('spd') or 100) if target.stats else 100
        except:
            atk, dfn = 100, 100

        dfn = max(dfn, 1)
        damage = ((42 * base_power * (atk / dfn)) / 50 + 2) * stab * type_mult
        return damage

    def _get_best_switch(self, battle) -> Pokemon | None:
        """Find best Pokemon to switch to based on matchup."""
        switches = battle.available_switches
        if not switches:
            return None

        opponent = battle.opponent_active_pokemon
        if not opponent:
            return switches[0] if switches else None

        best_switch = None
        best_score = -999

        for pokemon in switches:
            if pokemon.fainted:
                continue

            score = 0

            # Type matchup scoring
            # Check what types opponent has and our resistance
            for opp_type in (opponent.types or []):
                try:
                    eff = pokemon.damage_multiplier(Move(opp_type.name.lower()))
                    if eff < 0.5:
                        score += 3  # Immune or double resist
                    elif eff < 1.0:
                        score += 1.5  # Resist
                    elif eff > 1.0:
                        score -= 1  # Weak
                    elif eff > 2.0:
                        score -= 2  # Very weak
                except:
                    pass

            # Check if we can hit opponent super-effectively
            for my_type in (pokemon.types or []):
                try:
                    eff = opponent.damage_multiplier(Move(my_type.name.lower()))
                    if eff > 1.0:
                        score += 1
                    if eff > 2.0:
                        score += 1
                except:
                    pass

            # HP factor (prefer healthy mons)
            score += pokemon.current_hp_fraction * 2

            # Speed consideration
            my_speed = pokemon.stats.get('spe', 100) if pokemon.stats else 100
            opp_speed = opponent.stats.get('spe', 100) if opponent.stats else 100
            if my_speed is None:
                my_speed = 100
            if opp_speed is None:
                opp_speed = 100
            if my_speed > opp_speed:
                score += 0.5

            if score > best_score:
                best_score = score
                best_switch = pokemon

        return best_switch

    def _find_setup_move(self, moves, user: Pokemon) -> Move | None:
        """Find a setup move if we have one."""
        # Check current boosts - don't over-boost
        current_boosts = sum(user.boosts.values()) if hasattr(user, 'boosts') and user.boosts else 0
        if current_boosts >= 4:
            return None

        for move in moves:
            try:
                if move.id in self.SETUP_MOVES:
                    return move
            except:
                pass
        return None

    def _find_hazard_move(self, moves, battle) -> Move | None:
        """Find hazard setup or removal move."""
        # Clear our hazards first
        if battle.side_conditions:
            for move in moves:
                try:
                    if move.id in self.HAZARD_CLEAR:
                        return move
                except:
                    pass

        # Set hazards if opponent doesn't have them and we have enough mons
        opp_remaining = 6 - sum(1 for p in battle.opponent_team.values() if p.fainted)
        if opp_remaining >= 3:
            for move in moves:
                try:
                    if move.id in self.HAZARD_SETUP:
                        # Check if already set
                        hazard_condition = {
                            'stealthrock': 'stealthrock',
                            'spikes': 'spikes',
                            'toxicspikes': 'toxicspikes',
                            'stickyweb': 'stickyweb'
                        }.get(move.id)
                        # Only set if not already present (simplified check)
                        return move
                except:
                    pass

        return None

    def _find_pivot_move(self, moves) -> Move | None:
        """Find a pivot move (U-turn, Volt Switch, etc)."""
        for move in moves:
            try:
                if move.id in self.PIVOT_MOVES:
                    return move
            except:
                pass
        return None

    def _find_status_move(self, moves, target: Pokemon) -> Move | None:
        """Find a status move if opponent isn't already statused."""
        if not target:
            return None
        try:
            if target.status:
                return None
        except:
            pass

        for move in moves:
            try:
                if move.id in self.STATUS_MOVES:
                    return move
            except:
                pass
        return None

    def _find_best_damage_move(self, moves, user: Pokemon, target: Pokemon) -> Move | None:
        """Find the highest damage move."""
        best_move = None
        best_damage = -1

        for move in moves:
            try:
                if move.category == MoveCategory.STATUS:
                    continue
            except:
                continue

            damage = self._estimate_damage(move, user, target)

            # Priority move bonus when low HP
            try:
                priority = getattr(move, 'priority', 0)
                if priority > 0 and user and user.current_hp_fraction < 0.4:
                    damage *= 1.3
            except:
                pass

            if damage > best_damage:
                best_damage = damage
                best_move = move

        return best_move if best_move else (moves[0] if moves else None)
