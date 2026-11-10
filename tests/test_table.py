import json

import numpy as np
import pandas as pd
import pytest

from npl.config import RAW, canon
from npl.data import load_fixtures
from npl.table import nrr_overs, overs_to_balls, rank, standings


def test_overs_to_balls():
    assert overs_to_balls("16.5") == 101
    assert overs_to_balls(20) == 120
    assert overs_to_balls(9.4) == 58


def test_all_out_counts_full_quota():
    assert nrr_overs(17.2, 10, 20) == 20
    assert nrr_overs(17.2, 9, 20) == pytest.approx(17 + 2 / 6)
    assert nrr_overs(5, 3, 5) == 5


def _fx(rows):
    cols = ["match_no", "bat_first", "bat_second", "r1", "w1", "o1", "r2", "w2", "o2", "winner",
            "overs_limit", "status"]
    return pd.DataFrame(rows, columns=cols)


def test_standings_points_and_nrr_by_hand():
    f = _fx([
        [1, "A", "B", 150, 5, 20, 120, 10, 18.0, "A", 20, "result"],
        [2, "B", "C", 100, 10, 15.0, 101, 2, 10.0, "C", 20, "result"],
        [3, "A", "C", None, None, None, None, None, None, None, 20, "no_result"],
    ])
    t = standings(f, teams=["A", "B", "C"])
    assert t.loc["A", "Pts"] == 3 and t.loc["C", "Pts"] == 3 and t.loc["B", "Pts"] == 0
    # A: 150/20 - 120/20 (B bowled out -> 20 overs)
    assert t.loc["A", "NRR"] == pytest.approx(150 / 20 - 120 / 20)
    # B: scored 120+100 in 20+20 overs, conceded 150+101 in 20+10
    assert t.loc["B", "NRR"] == pytest.approx(220 / 40 - 251 / 30)
    assert t.loc["C", "NRR"] == pytest.approx(101 / 10 - 100 / 20)


def test_2026_table_matches_published():
    """Points through match 20 and NRR through match 19 match the Wikipedia table of
    9 Nov 2026 (its NRR column had not yet been updated for match 20)."""
    f = load_fixtures()
    pts = standings(f)
    expected_pts = {"Kathmandu Gorkhas": 8, "Lumbini Lions": 7, "Sudur Paschim Royals": 6,
                    "Chitwan Rhinos": 5, "Janakpur Bolts": 4, "Biratnagar Kings": 4,
                    "Karnali Yaks": 4, "Pokhara Avengers": 2}
    for team, p in expected_pts.items():
        assert pts.loc[team, "Pts"] == p
    nrr19 = standings(f[f.match_no <= 19])
    published = {"Kathmandu Gorkhas": 1.313, "Lumbini Lions": 1.500, "Sudur Paschim Royals": -0.275,
                 "Chitwan Rhinos": 0.207, "Janakpur Bolts": -0.083, "Biratnagar Kings": -0.820,
                 "Karnali Yaks": -0.846, "Pokhara Avengers": -0.707}
    for team, v in published.items():
        assert nrr19.loc[team, "NRR"] == pytest.approx(v, abs=0.0015)


@pytest.mark.skipif(not (RAW / "npl").exists(), reason="Cricsheet data not downloaded")
def test_fixture_csv_agrees_with_cricsheet():
    f = load_fixtures().set_index("match_no")
    checked = 0
    for path in (RAW / "npl").glob("155*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        no = int(d["info"]["event"]["match_number"])
        row = f.loc[no]
        inns = d["innings"][:2]
        assert canon(inns[0]["team"]) == row.bat_first
        for inn, r, w, o in ((inns[0], row.r1, row.w1, row.o1), (inns[1], row.r2, row.w2, row.o2)):
            dl = [x for ov in inn["overs"] for x in ov["deliveries"]]
            assert sum(x["runs"]["total"] for x in dl) == r
            assert sum(len(x.get("wickets", [])) for x in dl) == w
            legal = sum(1 for x in dl if not {"wides", "noballs"} & set(x.get("extras", {})))
            assert legal == overs_to_balls(o)
        assert canon(d["info"]["outcome"]["winner"]) == row.winner
        checked += 1
    assert checked >= 12


def test_rank_points_then_nrr():
    pts = np.array([[4, 4, 6, 2]])
    nrr = np.array([[0.5, 1.0, -2.0, 3.0]])
    order = rank(pts, nrr, np.random.default_rng(0))
    assert order.tolist() == [[2, 1, 0, 3]]
