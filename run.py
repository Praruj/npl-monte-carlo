"""One command, end to end:  python run.py [--sims 10000] [--seed 7]

Downloads Cricsheet data if missing, fits Elo and player rates, backtests both on
2025-26, simulates the rest of NPL 2026, writes results/ and the README results block.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from npl import backtest, charts, elo
from npl.config import MANUAL, RESULTS, ROOT, SHORT, TEAM_IDX, TEAMS
from npl.data import load_all, load_fixtures, npl_results_for_elo
from npl.models import BallByBallModel, EloModel
from npl.players import estimate, current_lineups
from npl.season import no_result_rate, simulate_season
from npl.table import standings

def pct(x: float) -> str:
    if 0 < x < 0.005:
        return "<1%"
    if 0.995 <= x < 1:
        return ">99%"
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backtest-sims", type=int, default=2000)
    ap.add_argument("--refresh", action="store_true", help="re-parse the raw Cricsheet files")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    matches, balls, lineups, names = load_all(refresh=args.refresh)
    fixtures = load_fixtures()
    played = fixtures[fixtures.status != "scheduled"]
    as_of = played.date.max()
    print(f"data: {len(matches[matches.comp == 'NPL'])} NPL + {len(matches[matches.comp == 'T20I'])} "
          f"T20I matches from Cricsheet; 2026 table through match {played.match_no.max()} ({as_of})")

    # ---- Elo -----------------------------------------------------------------------
    results = npl_results_for_elo(matches, fixtures)
    best, grid = elo.tune(results)
    h = elo.chase_bonus(results)
    er = elo.run_elo(results, k=best["k"], regress=best["regress"], h=h)
    grid.to_csv(RESULTS / "elo_grid.csv", index=False)
    print(f"elo: K={best['k']:.0f}, regress={best['regress']}, chase bonus H={h:.1f}")

    # ---- Player rates + current XIs ------------------------------------------------
    rates = estimate(balls)
    adds = json.loads((MANUAL / "lineup_additions.json").read_text(encoding="utf-8"))
    adds = {k: v for k, v in adds.items() if not k.startswith("_")}
    xis = current_lineups(matches, lineups, 2026, TEAMS, additions=adds, rates=rates)

    elo_model = EloModel(np.array([er.ratings[t] for t in TEAMS]), h, results)
    ball_model = BallByBallModel(rates, xis, TEAMS)

    # ---- Backtest ------------------------------------------------------------------
    bt = backtest.ball_model_predictions(matches, balls, lineups, n=args.backtest_sims, seed=args.seed)
    bt = bt.merge(er.predictions[["date", "team1", "team2", "p_chase"]], on=["date", "team1", "team2"], how="left")
    bt["coin"] = 0.5
    bt["ensemble"] = (bt.p_chase + bt.p_ball) / 2
    scores = backtest.score(bt, {"Coin flip": "coin", "Elo": "p_chase", "Ball-by-ball": "p_ball",
                                 "Ensemble": "ensemble"})
    bt.to_csv(RESULTS / "backtest_predictions.csv", index=False)
    scores.to_csv(RESULTS / "backtest_scores.csv", index=False)
    print(scores.round(4).to_string(index=False))

    # ---- Season simulation ---------------------------------------------------------
    league_played = len(played) + 28 * 2       # 2024 + 2025 league stages had no washouts
    n_nr = int((fixtures.status == "no_result").sum())
    p_nr = no_result_rate(n_nr, league_played)
    sims = {m.name: simulate_season(fixtures, m, n=args.sims, seed=args.seed, p_no_result=p_nr)
            for m in (elo_model, ball_model)}
    summaries = {k: s.summary() for k, s in sims.items()}
    ens = sum(summaries.values()) / len(summaries)
    ens_pos = sum(s.position_matrix() for s in sims.values()) / len(sims)
    pre = {m.name: simulate_season(fixtures, m, n=args.sims, seed=args.seed, p_no_result=p_nr,
                                   from_scratch=True).summary() for m in (ball_model,)}

    table_now = standings(fixtures)
    out = ens.join(table_now[["P", "Pts", "NRR"]]).sort_values(["p_title", "p_playoffs"], ascending=False)
    out.to_csv(RESULTS / "season_odds_ensemble.csv")
    for k, s in summaries.items():
        s.to_csv(RESULTS / f"season_odds_{k.lower().replace('-', '_')}.csv")
    ens_pos.to_csv(RESULTS / "position_matrix.csv")

    remaining = fixtures[fixtures.status == "scheduled"]
    fx_rows = []
    for f in remaining.itertuples():
        a, b = TEAM_IDX[f.bat_first], TEAM_IDX[f.bat_second]
        pe, pb = elo_model.win_prob(a, b), ball_model.win_prob(a, b, seed=f.match_no)
        fx_rows.append(dict(match=f.match_no, date=f.date, home=f.bat_first, away=f.bat_second,
                            p_elo=pe, p_ball=pb, p_ens=(pe + pb) / 2))
    fx_df = pd.DataFrame(fx_rows)
    fx_df.to_csv(RESULTS / "remaining_fixtures.csv", index=False)

    # Model sanity: simulated vs real Kirtipur first-innings totals.
    rng = np.random.default_rng(args.seed)
    a = rng.integers(0, len(TEAMS), 20000)
    b = (a + rng.integers(1, len(TEAMS), 20000)) % len(TEAMS)
    sim = ball_model.play(a, b, rng)
    real = results[results.overs_limit == 20]
    calib = dict(sim_r1_mean=float(sim["r1"].mean()), sim_r1_sd=float(sim["r1"].std()),
                 real_r1_mean=float(real.r1.mean()), real_r1_sd=float(real.r1.std()),
                 sim_chase_win=float((sim["winner"] == sim["chase"]).mean()),
                 real_chase_win=float((real.winner == real.team2).mean()))

    # ---- Charts --------------------------------------------------------------------
    for theme, suffix in (("dark", "-dark"), ("light", "")):
        charts.use_theme(theme)
        charts.odds(ens, SHORT, RESULTS / f"odds{suffix}.png", as_of, args.sims)
        charts.positions(ens_pos, SHORT, RESULTS / f"positions{suffix}.png", as_of)
        charts.elo_history(er.history, TEAMS, SHORT, RESULTS / f"elo_history{suffix}.png")
        charts.model_compare(summaries, SHORT, RESULTS / f"model_compare{suffix}.png")

    summary = dict(as_of=as_of, sims_per_model=args.sims, seed=args.seed, p_no_result=p_nr,
                   elo=dict(**best, h=h, ratings=er.ratings), calibration=calib,
                   backtest=scores.to_dict("records"))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    write_readme(out, summaries, pre, fx_df, scores, calib, best, h, p_nr, as_of, args, er, xis, names)
    print(out.round(3).to_string())


def write_readme(out, summaries, pre, fx_df, scores, calib, best, h, p_nr, as_of, args, er, xis, names):
    L = []
    L.append(f"_Generated by `python run.py --sims {args.sims} --seed {args.seed}`. "
             f"Table includes every game up to {as_of}._\n")
    L.append("| Team | P | Pts | NRR | Top of table | Top 4 | Final | Title | Title (Elo) | Title (ball-by-ball) | Title (pre-season, ball-by-ball) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    pre_b = pre["Ball-by-ball"]
    for t, r in out.iterrows():
        L.append(f"| {t} | {r.P:.0f} | {r.Pts:.0f} | {r.NRR:+.3f} | {pct(r.p_top)} | {pct(r.p_playoffs)} | "
                 f"{pct(r.p_final)} | **{pct(r.p_title)}** | {pct(summaries['Elo'].p_title[t])} | "
                 f"{pct(summaries['Ball-by-ball'].p_title[t])} | {pct(pre_b.p_title[t])} |")
    L.append("\n![Playoff and title odds](results/odds.png)\n")
    L.append("![Finishing positions](results/positions.png)\n")
    L.append("**Remaining league games** (probability the first-named team wins):\n")
    L.append("| # | Date | Game | Elo | Ball-by-ball | Ensemble |")
    L.append("|---:|---|---|---:|---:|---:|")
    for r in fx_df.itertuples():
        L.append(f"| {r.match} | {r.date} | {r.home} v {r.away} | {pct(r.p_elo)} | {pct(r.p_ball)} | {pct(r.p_ens)} |")
    L.append("\n**Backtest** (NPL 2025 and 2026 games with Cricsheet line-ups, predicted before each game; "
             "lower log loss / Brier is better):\n")
    L.append("| Model | Games | Log loss | Brier | Accuracy |")
    L.append("|---|---:|---:|---:|---:|")
    for r in scores.itertuples():
        L.append(f"| {r.model} | {r.n} | {r.log_loss:.3f} | {r.brier:.3f} | {r.accuracy:.0%} |")
    L.append(f"\n**Calibration check:** simulated first-innings totals at Kirtipur average "
             f"{calib['sim_r1_mean']:.0f} (sd {calib['sim_r1_sd']:.0f}) against {calib['real_r1_mean']:.0f} "
             f"(sd {calib['real_r1_sd']:.0f}) in real NPL games; the chasing side wins "
             f"{calib['sim_chase_win']:.0%} of simulated games and {calib['real_chase_win']:.0%} of real ones.\n")
    L.append(f"**Elo settings:** K = {best['k']:.0f}, {best['regress']:.0%} regression to 1500 between "
             f"seasons, chase bonus {h:+.0f} points. No-result chance per remaining league game: {p_nr:.1%}.\n")
    L.append("![Elo history](results/elo_history.png)\n")
    L.append("![Model comparison](results/model_compare.png)\n")
    L.append("<details><summary>Batting orders and bowlers used for the remaining games</summary>\n")
    for t, x in xis.items():
        bat = ", ".join(names.get(p, p) for p in x["bat"])
        bowl = ", ".join(names.get(p, p) for p in x["bowl"])
        L.append(f"- **{t}**: {bat}. Bowlers: {bowl}.")
    L.append("\n</details>")
    block = "\n".join(L)

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                 lambda m: m.group(1) + "\n" + block + "\n" + m.group(2), text, flags=re.S)
    readme.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
