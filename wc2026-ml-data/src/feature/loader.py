from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPO_ZIP = "https://github.com/openfootball/internationals/archive/refs/heads/master.zip"

ALIASES = {
    "USA": "United States",
    "USMNT": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Turkey": "Türkiye",
    "Czechia": "Czech Republic",
    "Democratic Republic of the Congo": "DR Congo",
    "D. R. Congo": "DR Congo",
    "Congo DR": "DR Congo",
}

HOSTS = {"USA", "Canada", "Mexico"}

WC_TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Türkiye",
    "Germany", "Curaçao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
]

MONTH = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
DATE_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})\s*$")
MATCH_RE = re.compile(r"^\s+(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)(?:\s+@\s+(.+?))?(?:\s+\[.*\])?\s*$")
TITLE_RE = re.compile(r"^=\s*(.+?)\s*(?:#.*)?$")
YEAR_RE = re.compile(r"(18|19|20)\d{2}")


def canonical(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return ALIASES.get(name, name)


def year_from_path(path: str) -> int | None:
    m = YEAR_RE.search(path)
    return int(m.group(0)) if m else None


def clean_tournament(title: str, path: str) -> str:
    title = re.sub(r"\s*#.*$", "", title).strip()
    if title:
        return title
    parent = Path(path).parent.name.replace("_", " ").title()
    return parent


def parse_file(path: str, text: str) -> list[dict]:
    y = year_from_path(path)
    if y is None:
        return []
    tournament = Path(path).parent.name.replace("_", " ").title()
    current_date: datetime | None = None
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        mt = TITLE_RE.match(line)
        if mt:
            tournament = clean_tournament(mt.group(1), path)
            continue
        md = DATE_RE.match(line.strip())
        if md:
            _, mon, day = md.groups()
            current_date = datetime(y, MONTH[mon], int(day))
            continue
        if current_date is None:
            continue
        mm = MATCH_RE.match(line)
        if not mm:
            continue
        home, hs, a_s, away, venue = mm.groups()
        home = canonical(home)
        away = canonical(away)
        if not home or not away or home.startswith("("):
            continue
        out.append(dict(
            date=current_date,
            home_team=home,
            away_team=away,
            home_score=int(hs),
            away_score=int(a_s),
            tournament=tournament,
            venue=venue,
            source_file=path,
        ))
    return out


def download_recent_results(before_date: str | None = None) -> pd.DataFrame:
    import requests
    r = requests.get(REPO_ZIP, timeout=60)
    r.raise_for_status()

    matches = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".txt"):
                continue
            base = Path(info.filename).name
            if not YEAR_RE.search(base):
                continue
            text = zf.read(info).decode("utf-8", errors="replace")
            matches.extend(parse_file(info.filename, text))

    teamset = {canonical(t) for t in WC_TEAMS}
    cutoff = pd.to_datetime(before_date) if before_date else None
    rows = []
    for m in matches:
        if cutoff is not None and pd.Timestamp(m["date"].date()) > cutoff:
            continue
        for team in (m["home_team"], m["away_team"]):
            if team not in teamset:
                continue
            is_home = team == m["home_team"]
            team_score = m["home_score"] if is_home else m["away_score"]
            opp_score = m["away_score"] if is_home else m["home_score"]
            opp = m["away_team"] if is_home else m["home_team"]
            result = "win" if team_score > opp_score else "loss" if team_score < opp_score else "draw"
            match_type = "friendly" if "friendly" in m["tournament"].lower() else "competitive"
            rows.append({
                "date": m["date"].date().isoformat(),
                "country": team,
                "home_away": "home" if is_home else "away",
                "wc_team_score": team_score,
                "opposition_score": opp_score,
                "result": result,
                "match_type": match_type,
                "opposition_country": opp,
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "tournament": m["tournament"],
                "source_file": m["source_file"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date_sort"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values(["country", "date_sort"], ascending=[True, False])
    df = df.groupby("country", group_keys=False).head(20)
    df = df.drop(columns=["date_sort"]).reset_index(drop=True)
    return df


def load_elo_ratings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "elo_ratings.csv")
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def load_fifa_ratings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fifa_ratings.csv")
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def load_schedule() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "FIFA2026_schedule_Fixtures.csv")
    df["date_dt"] = pd.to_datetime(df["date_dt"], format="mixed")
    return df


def get_latest_ratings(elo_df: pd.DataFrame, fifa_df: pd.DataFrame, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(as_of)
    elo_latest = elo_df[elo_df["date"] <= cutoff].sort_values("date").groupby("country").last().reset_index()
    fifa_latest = fifa_df[fifa_df["date"] <= cutoff].sort_values("date").groupby("country").last().reset_index()
    return elo_latest, fifa_latest


def build_feature_view_data(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_df: pd.DataFrame,
    n_games: int = 20,
    include_friendly: bool = True,
    n_test: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = results_df.copy()
    if not include_friendly:
        df = df[df["match_type"] == "competitive"].copy()

    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values(["country", "date"], ascending=[True, True])

    df["country_canon"] = df["country"].map(lambda x: ALIASES.get(x, x))
    df["opposition_canon"] = df["opposition_country"].map(lambda x: ALIASES.get(x, x))

    elo_map = elo_df.set_index("country")["elo_rating"].to_dict()
    fifa_rank_map = fifa_df.set_index("country")["ranking"].to_dict()
    fifa_points_map = fifa_df.set_index("country")["points"].to_dict()

    df["team_elo"] = df["country_canon"].map(elo_map).fillna(1500)
    df["opp_elo"] = df["opposition_canon"].map(elo_map).fillna(1500)
    df["elo_diff"] = df["team_elo"] - df["opp_elo"]
    df["team_rank"] = df["country_canon"].map(fifa_rank_map).fillna(100)
    df["opp_rank"] = df["opposition_canon"].map(fifa_rank_map).fillna(100)
    df["rank_diff"] = df["opp_rank"] - df["team_rank"]
    df["team_points"] = df["country_canon"].map(fifa_points_map).fillna(1000)
    df["opp_points"] = df["opposition_canon"].map(fifa_points_map).fillna(1000)
    df["points_diff"] = df["team_points"] - df["opp_points"]

    df["home"] = (df["home_away"] == "home").astype(int)
    df["is_friendly"] = (df["match_type"] == "friendly").astype(int)
    df["is_host"] = df["country_canon"].isin(HOSTS).astype(int)

    label_map = {"loss": 0, "draw": 1, "win": 2}
    df["label"] = df["result"].map(label_map)

    feature_cols = ["home", "is_friendly", "team_elo", "opp_elo", "elo_diff",
                    "team_rank", "opp_rank", "rank_diff", "team_points", "opp_points",
                    "points_diff"]
    if include_friendly:
        feature_cols = ["home", "is_friendly", "team_elo", "opp_elo", "elo_diff",
                        "team_rank", "opp_rank", "rank_diff", "team_points", "opp_points",
                        "points_diff"]

    df = df.sort_values(["country", "date"], ascending=[True, True])
    test_mask = pd.Series(False, index=df.index)
    for _, grp in df.groupby("country"):
        if len(grp) >= n_test:
            test_mask[grp.index[-n_test:]] = True
    train_mask = ~test_mask

    df["date"] = pd.to_datetime(df["date"], format="mixed")

    return df, df[feature_cols], df["label"], test_mask


def load_all_data(before_date: str = "2026-06-10") -> pd.DataFrame:
    print("Downloading recent match results...")
    results = download_recent_results(before_date)
    print(f"  {len(results)} result rows")
    print("Loading Elo ratings...")
    elo = load_elo_ratings()
    print(f"  {len(elo)} rows")
    print("Loading FIFA ratings...")
    fifa = load_fifa_ratings()
    print(f"  {len(fifa)} rows")
    return results, elo, fifa
