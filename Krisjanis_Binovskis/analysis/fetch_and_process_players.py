"""
Fetch NBA player stats for a season, process them into simplified
attributes for the game, and save to CSV files in data/.
"""

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


def norm(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series to [0, 1]."""
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def main(season: str = "2023-24") -> None:
    """
    Fetch per-game player stats for the given season and write:
    - data/players_raw.csv      (raw API result)
    - data/players_processed.csv (simplified stats used by the game)
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Fetching NBA data for season {season}...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    )
    df = stats.get_data_frames()[0]

    # Save raw API result
    raw_path = data_dir / "players_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

    # Columns we care about (adapt if the API changes names)
    cols = [
        "PLAYER_NAME",
        "GP",
        "PTS",
        "REB",
        "AST",
        "TOV",
        "FG_PCT",
        "FG3_PCT",
    ]
    df = df[cols].copy()

    # Filter out players with few games (remove garbage/small samples)
    df = df[df["GP"] >= 15].reset_index(drop=True)

    # Ensure numeric types
    for col in ["PTS", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # Game attributes (all normalized 0–1)
    df["scoring"] = norm(df["PTS"] + 5 * df["FG_PCT"])
    df["playmaking"] = norm(df["AST"] - 0.5 * df["TOV"])
    df["discipline"] = norm(-df["TOV"])

    df["impact_raw"] = (
        0.5 * df["scoring"] +
        0.3 * df["playmaking"] +
        0.2 * df["discipline"]
    )
    df["luck_factor"] = norm(df["impact_raw"])

    # Tiers: bust / role_player / star
    q_low = df["impact_raw"].quantile(0.2)
    q_high = df["impact_raw"].quantile(0.8)

    def tier_row(row) -> str:
        if row["impact_raw"] <= q_low:
            return "bust"
        if row["impact_raw"] >= q_high:
            return "star"
        return "role_player"

    df["tier"] = df.apply(tier_row, axis=1)

    # Height/weight are not available here → use reasonable defaults
    avg_height_m = 2.0   # ~6'7"
    avg_weight_kg = 100.0
    df["height_m"] = avg_height_m
    df["weight_kg"] = avg_weight_kg

    out = df[
        [
            "PLAYER_NAME",
            "height_m",
            "weight_kg",
            "scoring",
            "playmaking",
            "discipline",
            "luck_factor",
            "tier",
        ]
    ].rename(columns={"PLAYER_NAME": "player_name"})

    out_path = data_dir / "players_processed.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved processed players to {out_path}")
    print("Tier counts:")
    print(out["tier"].value_counts())


if __name__ == "__main__":
    main()
