"""Download and parse Cricsheet data; load the hand-entered 2026 fixture list."""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import CRICSHEET, MANUAL, OUTCOMES, PROCESSED, RAW, T20I_SINCE, canon, phase

CACHE_VERSION = 2


def ensure_raw(force: bool = False) -> None:
    """Fetch the Cricsheet zips (JSON only) into data/raw/<name>/ if missing."""
    for name, url in CRICSHEET.items():
        out = RAW / name
        if out.exists() and any(out.glob("*.json")) and not force:
            continue
        print(f"downloading {url}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                if member.endswith(".json") or member.endswith(".txt"):
                    (out / Path(member).name).write_bytes(zf.read(member))


def _outcome(delivery: dict) -> int:
    if delivery.get("wickets"):
        return OUTCOMES.index("W")
    r = delivery["runs"]["total"]
    if r >= 6:
        return OUTCOMES.index("6")
    if r == 5:
        return OUTCOMES.index("4")
    return OUTCOMES.index(str(r))


def parse_match(d: dict, mid: str, comp: str):
    """Return (match dict, list of ball dicts, lineup dict) for one Cricsheet match."""
    info = d["info"]
    people = info["registry"]["people"]
    date = info["dates"][0]
    teams = [canon(t) for t in info["teams"]]

    balls, innings_summary = [], []
    order: dict[str, list[str]] = {t: [] for t in teams}
    bowl_balls: dict[str, dict[str, int]] = {t: {} for t in teams}
    main_innings = [inn for inn in d.get("innings", []) if not inn.get("super_over")][:2]
    for k, inn in enumerate(main_innings):
        bat_team = canon(inn["team"])
        field_team = [t for t in teams if t != bat_team][0]
        runs = wkts = legal = 0
        for ov in inn["overs"]:
            for dl in ov["deliveries"]:
                ex = dl.get("extras", {})
                is_legal = "wides" not in ex and "noballs" not in ex
                bat_id, bowl_id = people[dl["batter"]], people[dl["bowler"]]
                for p in (dl["batter"], dl["non_striker"]):
                    pid = people[p]
                    if pid not in order[bat_team]:
                        order[bat_team].append(pid)
                if is_legal:
                    bowl_balls[field_team][bowl_id] = bowl_balls[field_team].get(bowl_id, 0) + 1
                nw = len(dl.get("wickets", []))
                runs += dl["runs"]["total"]
                wkts += nw
                legal += is_legal
                balls.append(
                    dict(
                        comp=comp, match_id=mid, date=date, inning=k, over=ov["over"],
                        phase=phase(ov["over"]), batter=bat_id, bowler=bowl_id,
                        legal=is_legal, outcome=_outcome(dl), runs=dl["runs"]["total"],
                    )
                )
        innings_summary.append(dict(team=bat_team, runs=runs, wkts=wkts, balls=legal,
                                    target_overs=inn.get("target", {}).get("overs")))

    outcome = info.get("outcome", {})
    winner = canon(outcome["winner"]) if "winner" in outcome else (
        canon(outcome["eliminator"]) if "eliminator" in outcome else None)
    match = dict(
        comp=comp, match_id=mid, date=date, season=int(date[:4]),
        match_number=info.get("event", {}).get("match_number"),
        stage=info.get("event", {}).get("stage"),
        team1=innings_summary[0]["team"] if innings_summary else teams[0],
        team2=innings_summary[1]["team"] if len(innings_summary) > 1 else teams[1],
        r1=innings_summary[0]["runs"] if innings_summary else np.nan,
        w1=innings_summary[0]["wkts"] if innings_summary else np.nan,
        b1=innings_summary[0]["balls"] if innings_summary else np.nan,
        r2=innings_summary[1]["runs"] if len(innings_summary) > 1 else np.nan,
        w2=innings_summary[1]["wkts"] if len(innings_summary) > 1 else np.nan,
        b2=innings_summary[1]["balls"] if len(innings_summary) > 1 else np.nan,
        overs_limit=(innings_summary[1]["target_overs"] if len(innings_summary) > 1
                     and innings_summary[1]["target_overs"] else info.get("overs", 20)),
        winner=winner,
        tie="eliminator" in outcome or outcome.get("result") == "tie",
        no_result=outcome.get("result") == "no result",
        toss_winner=canon(info["toss"]["winner"]) if "toss" in info else None,
        toss_decision=info.get("toss", {}).get("decision"),
    )
    lineup = {
        t: dict(xi=[people[p] for p in info["players"][canon_key]], bat_order=order[t],
                bowl_balls=bowl_balls[t])
        for canon_key in info["players"] for t in [canon(canon_key)]
    }
    return match, balls, lineup


def load_all(refresh: bool = False):
    """Parsed (matches, balls, lineups, names) for NPL + T20I, cached in data/processed."""
    cache = PROCESSED / "parsed.pkl"
    if cache.exists() and not refresh:
        obj = pickle.loads(cache.read_bytes())
        if obj.get("version") == CACHE_VERSION:
            return obj["matches"], obj["balls"], obj["lineups"], obj["names"]
    ensure_raw()
    matches, balls, lineups, names = [], [], {}, {}
    for comp, folder in [("NPL", RAW / "npl"), ("T20I", RAW / "t20i")]:
        for f in sorted(folder.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if comp == "T20I" and (d["info"]["dates"][0] < T20I_SINCE
                                   or d["info"].get("overs", 20) != 20):
                continue
            m, b, lu = parse_match(d, f.stem, comp)
            matches.append(m)
            balls.extend(b)
            lineups[m["match_id"]] = lu
            for nm, pid in d["info"]["registry"]["people"].items():
                names.setdefault(pid, nm)
    matches = pd.DataFrame(matches)
    balls = pd.DataFrame(balls)
    balls["date"] = pd.to_datetime(balls["date"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(dict(version=CACHE_VERSION, matches=matches, balls=balls,
                                        lineups=lineups, names=names)))
    return matches, balls, lineups, names


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    """2026 fixture list: played rows carry scores, scheduled rows are blank."""
    f = pd.read_csv(path or MANUAL / "npl_2026_fixtures.csv")
    for c in ("bat_first", "bat_second", "winner"):
        f[c] = f[c].map(lambda t: canon(t) if isinstance(t, str) else t)
    return f


def npl_results_for_elo(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """All completed NPL games in date order: Cricsheet for 2024-25, the fixture CSV for 2026.

    Columns: date, season, team1 (batted first), team2, winner, tie, margin_runs,
    chase_balls_left, chase_wkts_left.
    """
    rows = []
    hist = matches[(matches.comp == "NPL") & (matches.season < 2026)]
    for m in hist.itertuples():
        if m.no_result or pd.isna(m.r2):
            continue
        rows.append(dict(date=m.date, season=m.season, team1=m.team1, team2=m.team2,
                         winner=None if m.tie else m.winner, tie=bool(m.tie),
                         r1=m.r1, r2=m.r2, w2=m.w2, b2=m.b2, overs_limit=m.overs_limit))
    from .table import overs_to_balls
    for f in fixtures[fixtures.status == "result"].itertuples():
        rows.append(dict(date=f.date, season=2026, team1=f.bat_first, team2=f.bat_second,
                         winner=f.winner, tie=False, r1=f.r1, r2=f.r2, w2=f.w2,
                         b2=overs_to_balls(f.o2), overs_limit=f.overs_limit))
    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    chase_won = df.winner == df.team2
    df["margin_runs"] = np.where(~chase_won & ~df.tie, df.r1 - df.r2, np.nan)
    df["chase_balls_left"] = np.where(chase_won, df.overs_limit * 6 - df.b2, np.nan)
    df["chase_wkts_left"] = np.where(chase_won, 10 - df.w2, np.nan)
    return df
