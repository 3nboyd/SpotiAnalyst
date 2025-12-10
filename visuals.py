"""Quick Matplotlib plots for SpotiAnalyst.

Place your CSVs next to this file and run:
    python visuals.py

Charts are written into img/ and referenced by index.html as static PNGs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG_DIR / name, dpi=140)
    plt.close()


def truncate_label(text: str, max_len: int = 40) -> str:
    """Shorten long labels for readable bar charts."""
    text = text or ""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def scale_values(values):
    """Return scaled values and a simple unit label if numbers are large."""
    max_val = max(values) if values else 0
    if max_val >= 1_000_000_000:
        return [v / 1_000_000_000 for v in values], "Billions"
    if max_val >= 1_000_000:
        return [v / 1_000_000 for v in values], "Millions"
    return values, None


# ---------------- Data loading ----------------
def load_global(path: str = "moststreamedsongs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={"artist(s)_name": "artist_name", "released_year": "release_year", "bpm": "tempo"}
    )
    df = df.dropna(subset=["track_name", "artist_name", "streams"])

    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df["release_year"] = pd.to_numeric(df.get("release_year"), errors="coerce")
    for col in ["danceability", "energy", "valence", "tempo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["streams"])


def load_user(path: str = "spotify_history.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names we expect from Spotify exports
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "trackName": "track_name",
            "artistName": "artist_name",
            "msPlayed": "ms_played",
            "endTime": "end_time",
            "timestamp": "end_time",
            "ts": "end_time",
        }
    )
    if "end_time" not in df and "ts" in df:
        df["end_time"] = df["ts"]
    if "end_time" not in df and "timestamp" in df:
        df["end_time"] = df["timestamp"]
    df["ms_played"] = pd.to_numeric(df.get("ms_played"), errors="coerce").fillna(0)
    df["end_time"] = pd.to_datetime(df.get("end_time"), errors="coerce")
    df = df.dropna(subset=["track_name", "artist_name"])
    df["hours_played"] = df["ms_played"] / 3_600_000
    return df


# ---------------- Global charts ----------------
def plot_global_top_songs(df: pd.DataFrame, n: int = 20) -> None:
    top = df.nlargest(n, "streams")[::-1]
    plt.figure(figsize=(9, 10))
    labels = (top["track_name"] + " — " + top["artist_name"]).apply(lambda s: truncate_label(s, 45))
    scaled_streams, unit = scale_values(top["streams"].tolist())
    plt.barh(labels, scaled_streams, color="#1DB954")
    x_label = "Total Streams" + (f" ({unit})" if unit else "")
    plt.xlabel(x_label)
    plt.title(f"Top {n} Most Streamed Songs")
    save_fig("global_top_songs.png")


def plot_release_year(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["release_year"])
    grouped = valid.groupby("release_year")["streams"].sum()
    plt.figure(figsize=(10, 5))
    scaled, unit = scale_values(grouped.values.tolist())
    plt.bar(grouped.index, scaled, color="#1ED760")
    plt.xlabel("Release Year")
    y_label = "Total Streams" + (f" ({unit})" if unit else "")
    plt.ylabel(y_label)
    plt.title("Streams by Release Year")
    save_fig("global_year_streams.png")


def plot_concentration(df: pd.DataFrame) -> None:
    ordered = df.sort_values("streams", ascending=False)
    share = ordered["streams"].cumsum() / ordered["streams"].sum()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(share) + 1), share, marker="o", color="#1DB954")
    plt.xlabel("Song Rank")
    plt.ylabel("Cumulative Share of Streams")
    plt.ylim(0, 1.05)
    plt.title("Stream Concentration")
    save_fig("global_stream_concentration.png")


def plot_energy_valence(df: pd.DataFrame) -> None:
    if "energy" not in df or "valence" not in df:
        return
    sizes = df["streams"] / df["streams"].max() * 800
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["energy"], df["valence"], s=sizes, c=df["energy"], cmap="Greens", alpha=0.7)
    plt.xlabel("Energy")
    plt.ylabel("Valence")
    plt.title("Energy vs Valence")
    plt.colorbar(scatter, label="Energy")
    save_fig("global_energy_valence.png")


def plot_bpm(df: pd.DataFrame) -> None:
    if "tempo" not in df:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df["tempo"].dropna(), bins=30, color="#1DB954")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.title("Tempo Distribution")
    save_fig("global_bpm.png")


# ---------------- User charts ----------------
def plot_user_top_artists(df: pd.DataFrame, n: int = 15) -> None:
    top = df.groupby("artist_name")["hours_played"].sum().sort_values(ascending=False).head(n)[::-1]
    plt.figure(figsize=(8, 8))
    plt.barh(top.index, top.values, color="#1DB954")
    plt.xlabel("Hours Listened")
    plt.title("Top Artists")
    save_fig("user_top_artists.png")


def plot_user_top_tracks(df: pd.DataFrame, n: int = 15) -> None:
    track_labels = df["track_name"] + " — " + df["artist_name"]
    top = df.assign(label=track_labels).groupby("label")["hours_played"].sum().sort_values(ascending=False).head(n)[::-1]
    labels = [truncate_label(lbl, 45) for lbl in top.index]
    plt.figure(figsize=(9, 10))
    plt.barh(labels, top.values, color="#1ED760")
    plt.xlabel("Hours Listened")
    plt.title("Top Tracks")
    save_fig("user_top_tracks.png")


def plot_user_hour(df: pd.DataFrame) -> None:
    if "end_time" not in df:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No end_time column", ha="center")
        save_fig("user_hour.png")
        return
    hours = df.dropna(subset=["end_time"]).copy()
    hours["end_time"] = pd.to_datetime(hours["end_time"], errors="coerce")
    hours = hours.dropna(subset=["end_time"])
    if hours.empty:
        totals = pd.Series([0] * 24, index=range(24))
    else:
        hours["hour"] = hours["end_time"].dt.hour
        totals = hours.groupby("hour")["hours_played"].sum().reindex(range(24), fill_value=0)
    plt.figure(figsize=(10, 4))
    plt.bar(totals.index, totals.values, color="#1DB954")
    plt.xlabel("Hour of Day")
    plt.ylabel("Hours Listened")
    plt.title("Listening by Hour")
    save_fig("user_hour.png")


# ---------------- Comparison ----------------
def overlap(user_df: pd.DataFrame, global_df: pd.DataFrame) -> pd.DataFrame:
    user = user_df.copy()
    global_copy = global_df.copy()
    user["key"] = (user["track_name"] + "|" + user["artist_name"]).str.lower().str.strip()
    global_copy["key"] = (global_copy["track_name"] + "|" + global_copy["artist_name"]).str.lower().str.strip()
    merged = user.merge(global_copy, on="key", suffixes=("_user", "_global"))
    # Ensure we have a clear hours column for the user side.
    if "hours_played_user" not in merged.columns and "hours_played" in merged.columns:
        merged = merged.rename(columns={"hours_played": "hours_played_user"})
    return merged


def plot_overlap_bar(merged: pd.DataFrame, user_df: pd.DataFrame) -> None:
    hours_on_hits = merged["hours_played_user"].sum() if "hours_played_user" in merged else 0
    total_hours = user_df["hours_played"].sum()
    other_hours = max(total_hours - hours_on_hits, 0)

    plt.figure(figsize=(6, 5))
    plt.bar(["Global Hits", "Other Songs"], [hours_on_hits, other_hours], color=["#1DB954", "#444"])
    plt.ylabel("Hours Listened")
    plt.title("Time Spent on Global Hits")
    save_fig("user_global_overlap.png")


def plot_overlap_top_tracks(merged: pd.DataFrame, top_n: int = 10) -> None:
    """Top overlapping tracks by hours listened."""
    if merged.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No overlapping songs found", ha="center")
        save_fig("user_overlap_top_tracks.png")
        return
    labels = merged["track_name_user"] + " — " + merged["artist_name_user"]
    totals = (
        merged.assign(label=labels)
        .groupby("label")["hours_played_user"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)[::-1]
    )
    short_labels = [truncate_label(lbl, 45) for lbl in totals.index]
    plt.figure(figsize=(9, 8))
    plt.barh(short_labels, totals.values, color="#1ED760")
    plt.xlabel("Hours Listened")
    plt.title("Top Overlapping Songs")
    save_fig("user_overlap_top_tracks.png")


def plot_feature_profile(merged: pd.DataFrame, global_df: pd.DataFrame) -> None:
    features = [c for c in ["danceability", "energy", "valence", "tempo"] if c in global_df.columns]
    if len(features) < 3 or merged.empty:
        return

    global_means = global_df[features].mean()
    user_means = merged[[f + "_global" for f in features]].mean()

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    g_vals = np.concatenate([global_means.values, [global_means.values[0]]])
    u_vals = np.concatenate([user_means.values, [user_means.values[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, g_vals, label="Global", color="#1DB954")
    ax.fill(angles, g_vals, alpha=0.25, color="#1DB954")
    ax.plot(angles, u_vals, label="User Overlap", color="#1ED760")
    ax.fill(angles, u_vals, alpha=0.25, color="#1ED760")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title("Audio Feature Profile")
    ax.legend(loc="upper right")
    save_fig("user_vs_global_radar.png")


# ---------------- Entry point ----------------
if __name__ == "__main__":
    global_df = load_global()
    user_df = load_user()
    merged = overlap(user_df, global_df)

    plot_global_top_songs(global_df)
    plot_release_year(global_df)
    plot_concentration(global_df)
    plot_energy_valence(global_df)
    plot_bpm(global_df)

    plot_user_top_artists(user_df)
    plot_user_top_tracks(user_df)
    plot_user_hour(user_df)

    plot_overlap_bar(merged, user_df)
    plot_overlap_top_tracks(merged)
    if not merged.empty:
        plot_feature_profile(merged, global_df)
