import numpy as np
import pandas as pd

from npl.elo import expected, run_elo
from npl.models import BallByBallModel, EloModel
from npl.players import OVER_PATTERN, Rates, bowling_sequence


def toy_rates(bat_boost=1.0):
    base = np.tile(np.array([0.36, 0.36, 0.07, 0.005, 0.11, 0.045, 0.05]), (2, 3, 1))
    ones = np.ones((3, 7))
    strong = np.ones(7)
    strong[[4, 5]] *= bat_boost        # more fours and sixes
    bat = np.vstack([ones, strong])     # row 3 = strong batter
    return Rates(base=base, extras=np.full((2, 3), 0.05), player_index={}, bat_mult=bat,
                 bowl_mult=np.ones((4, 7)), bat_balls={})


def model(rates):
    m = BallByBallModel.__new__(BallByBallModel)
    m.rates = rates
    return m


def test_innings_respects_limits():
    m = model(toy_rates())
    rng = np.random.default_rng(0)
    n = 3000
    bat = np.zeros((n, 11), int)
    bowl = np.zeros((n, 20), int)
    runs, wk, ov = m.innings(bat, bowl, 0, rng)
    assert (wk <= 10).all() and (ov <= 20).all() and (ov[wk == 10] == 20).all()
    assert 120 < runs.mean() < 200
    # a chase stops as soon as the target is reached (a six can overshoot by at most 5)
    target = np.full(n, 100)
    r2, w2, o2 = m.innings(bat, bowl, 1, rng, target=target)
    won = r2 >= 100
    assert won.mean() > 0.6
    assert (r2[won] <= 105).all()


def test_better_batters_score_more():
    m = model(toy_rates(bat_boost=1.6))
    rng = np.random.default_rng(1)
    n = 3000
    bowl = np.zeros((n, 20), int)
    avg, _, _ = m.innings(np.zeros((n, 11), int), bowl, 0, rng)
    strong, _, _ = m.innings(np.full((n, 11), 3), bowl, 0, rng)
    assert strong.mean() > avg.mean() + 15


def test_play_xi_winner_matches_scores():
    m = model(toy_rates())
    rng = np.random.default_rng(2)
    n = 2000
    z11, z20 = np.zeros((n, 11), int), np.zeros((n, 20), int)
    res = m.play_xi(z11, z20, z11, z20, rng)
    clear = ~res["tie"]
    assert ((res["r2"] > res["r1"])[clear] == res["chase_won"][clear]).all()
    assert abs(res["chase_won"].mean() - 0.5) < 0.08


def test_bowling_pattern_four_overs_each_never_consecutive():
    assert sorted(np.bincount(OVER_PATTERN)) == [4] * 5
    assert all(a != b for a, b in zip(OVER_PATTERN, OVER_PATTERN[1:]))
    seq = bowling_sequence(["a", "b", "c"])
    assert len(seq) == 20 and None not in seq


def _results():
    return pd.DataFrame(dict(
        date=["2024-01-0%d" % i for i in range(1, 7)], season=[2024] * 3 + [2025] * 3,
        team1=["Lumbini Lions"] * 6, team2=["Pokhara Avengers"] * 6,
        winner=["Lumbini Lions"] * 6, tie=[False] * 6, r1=[160] * 6, r2=[140] * 6, w2=[8] * 6,
        b2=[120] * 6, overs_limit=[20] * 6, margin_runs=[20.0] * 6,
        chase_balls_left=[np.nan] * 6, chase_wkts_left=[np.nan] * 6))


def test_elo_is_zero_sum_and_regresses_between_seasons():
    er = run_elo(_results(), k=20, regress=0.5, h=0.0)
    total = sum(er.ratings.values())
    assert np.isclose(total, 1500 * 8)
    gap_end_2024 = er.history[er.history.season == 2024].groupby("team").rating.last()
    start_2025 = er.season_start[2025]
    assert np.isclose(start_2025["Lumbini Lions"] - 1500, (gap_end_2024["Lumbini Lions"] - 1500) * 0.5)


def test_elo_model_scorelines_are_consistent():
    res = _results()
    res.loc[3:, ["winner", "margin_runs", "chase_balls_left", "chase_wkts_left"]] = \
        ["Pokhara Avengers", np.nan, 12.0, 4.0]
    ratings = np.full(8, 1500.0)
    ratings[0] = 1700
    m = EloModel(ratings, 0.0, res)
    rng = np.random.default_rng(0)
    n = 5000
    out = m.play(np.zeros(n, int), np.ones(n, int), rng)
    chase_won = out["winner"] == out["chase"]
    assert (out["r2"][chase_won] > out["r1"][chase_won]).all()
    assert (out["r2"][~chase_won] < out["r1"][~chase_won]).all()
    assert abs((out["winner"] == 0).mean() - expected(1700, 1500, 0)) < 0.03
