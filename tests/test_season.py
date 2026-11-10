import numpy as np
import pandas as pd

from npl.config import TEAMS, TEAM_IDX
from npl.season import playoffs, simulate_season


class LowerIndexWins:
    """Deterministic toy model: the team with the lower index always wins by 20 runs."""

    def play(self, a, b, rng):
        a, b = np.asarray(a), np.asarray(b)
        n = len(a)
        return dict(bat_first=a, chase=b, r1=np.full(n, 160), o1=np.full(n, 20.0),
                    r2=np.where(a < b, 140, 161), o2=np.where(a < b, 20.0, 18.0),
                    winner=np.minimum(a, b))


class CoinFlip:
    def play(self, a, b, rng):
        a, b = np.asarray(a), np.asarray(b)
        n = len(a)
        toss = rng.random(n) < 0.5                    # fair toss keeps teams symmetric
        a, b = np.where(toss, a, b), np.where(toss, b, a)
        first_wins = rng.random(n) < 0.5
        return dict(bat_first=a, chase=b, r1=np.full(n, 150), o1=np.full(n, 20.0),
                    r2=np.where(first_wins, 130, 151), o2=np.where(first_wins, 20.0, 19.0),
                    winner=np.where(first_wins, a, b))


def round_robin():
    rows, k = [], 1
    for i in range(len(TEAMS)):
        for j in range(i + 1, len(TEAMS)):
            rows.append(dict(match_no=k, date="2026-01-01", bat_first=TEAMS[i], bat_second=TEAMS[j],
                             r1=np.nan, w1=np.nan, o1=np.nan, r2=np.nan, w2=np.nan, o2=np.nan,
                             winner=np.nan, overs_limit=20, status="scheduled"))
            k += 1
    return pd.DataFrame(rows)


def test_full_round_robin_has_28_games():
    assert len(round_robin()) == 28


def test_deterministic_model_gives_deterministic_table_and_playoffs():
    sim = simulate_season(round_robin(), LowerIndexWins(), n=50, seed=1)
    assert (sim.order == np.arange(8)).all()
    assert (sim.points[:, 0] == 14).all() and (sim.points[:, 7] == 0).all()
    # Q1: 0 beats 1; Eliminator: 2 beats 3; Q2: 1 beats 2; Final: 0 beats 1.
    assert (sim.champion == 0).all()
    assert (np.sort(sim.finalists, 1) == [0, 1]).all()
    s = sim.summary()
    assert s.loc[TEAMS[0], "p_title"] == 1 and s.loc[TEAMS[4], "p_playoffs"] == 0


def test_playoff_bracket_routes_losers_correctly():
    class SeedWins:   # the side listed first always wins
        def play(self, a, b, rng):
            return dict(winner=np.asarray(a))

    seeds = np.array([[5, 3, 7, 1]])
    finalists, champ = playoffs(seeds, SeedWins(), np.random.default_rng(0))
    # Q1 5 beats 3; Elim 7 beats 1; Q2 3 (Q1 loser) beats 7; Final 5 beats 3.
    assert finalists.tolist() == [[5, 3]] and champ.tolist() == [5]


def test_probabilities_are_consistent():
    s = simulate_season(round_robin(), CoinFlip(), n=4000, seed=3).summary()
    assert np.isclose(s.p_top.sum(), 1)
    assert np.isclose(s.p_playoffs.sum(), 4)
    assert np.isclose(s.p_final.sum(), 2)
    assert np.isclose(s.p_title.sum(), 1)
    assert (s.p_title <= s.p_final).all() and (s.p_final <= s.p_playoffs).all()
    # symmetric teams: each near 1/8 to win
    assert (abs(s.p_title - 1 / 8) < 0.03).all()


def test_same_seed_same_answer():
    a = simulate_season(round_robin(), CoinFlip(), n=500, seed=11).summary()
    b = simulate_season(round_robin(), CoinFlip(), n=500, seed=11).summary()
    pd.testing.assert_frame_equal(a, b)


def test_played_games_are_fixed_and_no_results_score_one_point():
    f = round_robin().astype({"winner": object})
    f.loc[0, ["r1", "w1", "o1", "r2", "w2", "o2", "winner", "status"]] = \
        [100, 10, 18.0, 101, 1, 10.0, TEAMS[1], "result"]
    f.loc[1, "status"] = "no_result"
    sim = simulate_season(f, LowerIndexWins(), n=20, seed=0)
    # Team 0 loses to team 1, gets 1 point from the washout v team 2, wins the other 5.
    assert (sim.points[:, TEAM_IDX[TEAMS[0]]] == 11).all()
    assert (sim.points[:, 1] == 14).all()          # 2 for the upset + 12 from other games
    assert (sim.order[:, 0] == 1).all()


def test_no_result_rate_is_applied():
    sim = simulate_season(round_robin(), LowerIndexWins(), n=2000, seed=5, p_no_result=1.0)
    assert (sim.points == 7).all()                  # every game washed out
    sim = simulate_season(round_robin(), LowerIndexWins(), n=2000, seed=5, p_no_result=0.1)
    assert 0.5 < (sim.points[:, 7] > 0).mean() < 0.6  # 1 - 0.9^7 = 0.52


def test_net_run_rate_breaks_points_ties():
    class Margin:
        """All games coin flips on result, but team 0 always wins big and loses small."""
        def play(self, a, b, rng):
            a, b = np.asarray(a), np.asarray(b)
            n = len(a)
            fw = rng.random(n) < 0.5
            big = (a == 0) & fw
            return dict(bat_first=a, chase=b, r1=np.where(big, 220, 150), o1=np.full(n, 20.0),
                        r2=np.where(fw, np.where(big, 100, 140), 151), o2=np.full(n, 20.0),
                        winner=np.where(fw, a, b))
    sim = simulate_season(round_robin(), Margin(), n=3000, seed=2)
    pos = np.argsort(sim.order, 1)                  # league position of each team
    ahead, level = 0, 0
    for i in range(len(pos)):
        rivals = [t for t in range(1, 8) if sim.points[i, t] == sim.points[i, 0]]
        level += len(rivals)
        ahead += sum(pos[i, 0] < pos[i, t] for t in rivals)
        for t in rivals:   # whoever is ahead on equal points has the better NRR
            assert (pos[i, 0] < pos[i, t]) == (sim.nrr[i, 0] > sim.nrr[i, t])
    assert level > 100 and ahead / level > 0.6
