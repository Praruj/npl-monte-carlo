"""Match models. Both expose `play(a, b, rng)` over arrays of team indices (one entry per
simulated season) and return the full score line, so net run rate can be tracked.

Returned dict (all arrays of length n):
    bat_first, chase  team indices
    r1, r2            runs
    o1, o2            overs that count for NRR (full quota if bowled out)
    winner            team index
"""
from __future__ import annotations

import numpy as np

from .config import OUTCOME_RUNS, OUTCOMES
from .elo import expected

RUNS = np.array(OUTCOME_RUNS)
W = OUTCOMES.index("W")


def _toss(a, b, rng):
    first = np.where(rng.random(len(a)) < 0.5, a, b)
    return first, a + b - first


class EloModel:
    """Winner from Elo; score line resampled from real NPL games.

    The first-innings total is drawn from NPL 20-over first innings. If the side batting
    first wins, the margin is drawn from NPL run margins; if the chase wins, the balls to
    spare are drawn from NPL successful chases. Margins do not depend on the rating gap,
    so NRR here is noisier than real life (see the ball-by-ball model for that).
    """

    name = "Elo"

    def __init__(self, ratings: np.ndarray, h: float, results):
        self.ratings = np.asarray(ratings, float)
        self.h = h
        full = results[results.overs_limit == 20]
        self.r1 = full.r1.values.astype(int)
        self.margin = full.margin_runs.dropna().values.astype(int)
        self.balls_left = full.chase_balls_left.dropna().values.astype(int)
        won = full[full.winner == full.team2]
        self.overshoot = np.clip((won.r2 - won.r1 - 1).values.astype(int), 0, 5)

    def p_chase_wins(self, bat_first, chase):
        return expected(self.ratings[chase], self.ratings[bat_first], self.h)

    def win_prob(self, a: int, b: int) -> float:
        """P(a beats b) with a fair toss."""
        return 0.5 * (self.p_chase_wins(b, a) + 1 - self.p_chase_wins(a, b))

    def play(self, a, b, rng):
        a, b = np.asarray(a), np.asarray(b)
        n = len(a)
        first, chase = _toss(a, b, rng)
        chase_won = rng.random(n) < self.p_chase_wins(first, chase)
        r1 = rng.choice(self.r1, n)
        r2_loss = np.maximum(r1 - rng.choice(self.margin, n), 30)
        r2_win = r1 + 1 + rng.choice(self.overshoot, n)
        o2_win = (120 - rng.choice(self.balls_left, n)) / 6
        return dict(bat_first=first, chase=chase, r1=r1, o1=np.full(n, 20.0),
                    r2=np.where(chase_won, r2_win, r2_loss), o2=np.where(chase_won, o2_win, 20.0),
                    winner=np.where(chase_won, chase, first))


class BallByBallModel:
    """Simulates each legal ball: P(outcome) = NPL base[innings, phase] x batter mult x bowler
    mult, renormalised. Wides/no-balls add a run and a re-bowl at the NPL rate."""

    name = "Ball-by-ball"

    def __init__(self, rates, team_lineups: dict, teams: list):
        self.rates = rates
        idx = rates.player_index
        self.bat = np.array([[idx.get(p, 0) for p in team_lineups[t]["bat"]] for t in teams])
        from .players import bowling_sequence
        self.bowl = np.array([[idx.get(p, 0) for p in bowling_sequence(team_lineups[t]["bowl"])]
                              for t in teams])

    def innings(self, bat, bowl, inn, rng, target=None, overs=20):
        """bat (n, 11) and bowl (n, 20) hold player rows. Returns runs, wickets, NRR overs."""
        rates = self.rates
        n = bat.shape[0]
        ar = np.arange(n)
        runs = np.zeros(n, int)
        wk = np.zeros(n, int)
        balls = np.zeros(n, int)
        striker = np.zeros(n, int)
        non = np.ones(n, int)
        nxt = np.full(n, 2)
        done = np.zeros(n, bool)
        for ov in range(overs):
            ph = 0 if ov < 6 else (1 if ov < 15 else 2)
            base = rates.base[inn, ph]
            bm = rates.bowl_mult[bowl[:, ov]]
            p_extra = rates.extras[inn, ph] / (1 + rates.extras[inn, ph])
            for _ in range(6):
                act = ~done
                n_extra = rng.geometric(1 - p_extra, n) - 1
                runs += n_extra * act
                if target is not None:
                    done |= runs >= target
                    act = ~done
                p = base * rates.bat_mult[bat[ar, np.minimum(striker, 10)]] * bm
                cum = np.cumsum(p, 1)
                o = np.minimum((rng.random(n)[:, None] * cum[:, -1:] > cum).sum(1), len(RUNS) - 1)
                r = RUNS[o] * act
                runs += r
                balls += act
                out = (o == W) & act
                wk += out
                striker = np.where(out, nxt, striker)
                nxt = nxt + out
                sw = (r % 2 == 1)
                striker, non = np.where(sw, non, striker), np.where(sw, striker, non)
                done |= wk >= 10
                if target is not None:
                    done |= runs >= target
            striker, non = np.where(done, striker, non), np.where(done, non, striker)
            if done.all():
                break
        nrr_overs = np.where(wk >= 10, float(overs), balls / 6)
        return runs, wk, nrr_overs

    def play_xi(self, bat1, bowl1, bat2, bowl2, rng, overs=20):
        """Arrays of player rows for the side batting first (1) and second (2)."""
        r1, w1, o1 = self.innings(bat1, bowl2, 0, rng, overs=overs)
        r2, w2, o2 = self.innings(bat2, bowl1, 1, rng, target=r1 + 1, overs=overs)
        tie = r1 == r2
        chase_won = (r2 > r1) | (tie & (rng.random(len(r1)) < 0.5))   # super over = coin flip
        return dict(r1=r1, o1=o1, r2=r2, o2=o2, chase_won=chase_won, tie=tie)

    def play(self, a, b, rng):
        a, b = np.asarray(a), np.asarray(b)
        first, chase = _toss(a, b, rng)
        res = self.play_xi(self.bat[first], self.bowl[first], self.bat[chase], self.bowl[chase], rng)
        return dict(bat_first=first, chase=chase, r1=res["r1"], o1=res["o1"], r2=res["r2"],
                    o2=res["o2"], winner=np.where(res["chase_won"], chase, first))

    def win_prob(self, a: int, b: int, n: int = 4000, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        res = self.play(np.full(n, a), np.full(n, b), rng)
        return float((res["winner"] == a).mean())
