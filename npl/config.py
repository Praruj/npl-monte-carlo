"""Paths, team names and fixed settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANUAL = ROOT / "data" / "manual"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CRICSHEET = {
    "npl": "https://cricsheet.org/downloads/npl_json.zip",
    "t20i": "https://cricsheet.org/downloads/t20s_male_json.zip",
}

# Canonical names follow Cricsheet's 2026 spelling.
TEAMS = [
    "Biratnagar Kings",
    "Chitwan Rhinos",
    "Janakpur Bolts",
    "Karnali Yaks",
    "Kathmandu Gorkhas",
    "Lumbini Lions",
    "Pokhara Avengers",
    "Sudur Paschim Royals",
]
SHORT = {
    "Biratnagar Kings": "BRK",
    "Chitwan Rhinos": "CHR",
    "Janakpur Bolts": "JB",
    "Karnali Yaks": "KY",
    "Kathmandu Gorkhas": "KG",
    "Lumbini Lions": "LL",
    "Pokhara Avengers": "PA",
    "Sudur Paschim Royals": "SPR",
}
ALIASES = {
    "Kathmandu Gurkhas": "Kathmandu Gorkhas",
    "Sudurpaschim Royals": "Sudur Paschim Royals",
}
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}

# T20I data older than this is ignored (squads and the game have moved on).
T20I_SINCE = "2021-01-01"

# Outcome categories of a legal delivery.
OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
OUTCOME_RUNS = [0, 1, 2, 3, 4, 6, 0]

WIN_POINTS = 2
NR_POINTS = 1
PLAYOFF_TEAMS = 4


def canon(team: str) -> str:
    return ALIASES.get(team, team)


def phase(over: int) -> int:
    """0 = powerplay (overs 1-6), 1 = middle (7-15), 2 = death (16-20)."""
    return 0 if over < 6 else (1 if over < 15 else 2)
