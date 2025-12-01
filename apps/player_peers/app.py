import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# Import the data loader robustly so the app can be run with `streamlit run`.
try:
    # preferred absolute import when running from repo root
    from apps.player_peers.data_loader import load_engineered_features
except Exception:
    try:
        # when running as a package
        from .data_loader import load_engineered_features
    except Exception:
        # last-resort: import by path
        import importlib.util, sys
        loader_path = Path(__file__).resolve().parent / "data_loader.py"
        spec = importlib.util.spec_from_file_location("player_peers.data_loader", str(loader_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        load_engineered_features = mod.load_engineered_features


DEMO_PEERS = [
    "Stephen Curry",
    "LeBron James",
    "Kevin Durant",
    "Nikola Jokic",
]


def ensure_name_in_df(df, name_col="fullName"):
    if name_col not in df.columns:
        raise KeyError(f"Expected column '{name_col}' in data")


def main():
    st.set_page_config(page_title="Player Peers — NBA ML", layout="wide")
    st.title("Player Peers — Quick Comparison")

    try:
        df = load_engineered_features()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    ensure_name_in_df(df)

    # normalize fullName column
    df["fullName"] = df["fullName"].astype(str)

    st.markdown("Select players to compare. Use the search box to find players in the dataset.")

    # build player list from data (unique fullName values)
    all_players = sorted(df["fullName"].dropna().unique().tolist())

    demo_mode = st.checkbox("Demo mode — auto-select the 4-star players", value=True)
    if demo_mode:
        default_selection = [p for p in DEMO_PEERS if p in all_players]
    else:
        default_selection = []

    players = st.multiselect("Players", options=all_players, default=default_selection)

    if not players:
        st.info("Pick at least one player from the list.")
        return

    # filter for chosen players
    sel = df[df["fullName"].isin(players)].copy()
    if sel.empty:
        st.warning("No rows matched the selected players — check names in data `fullName` column.")
        return

    sel = sel.sort_values("gameDate")

    # date range: convert to native Python datetimes for Streamlit slider
    sel["gameDate"] = pd.to_datetime(sel["gameDate"])
    min_ts = sel["gameDate"].min()
    max_ts = sel["gameDate"].max()
    # ensure native python datetime objects (Streamlit slider doesn't accept pandas.Timestamp)
    try:
        min_dt = min_ts.to_pydatetime()
        max_dt = max_ts.to_pydatetime()
    except Exception:
        # fallback for numpy datetime64
        min_dt = pd.to_datetime(min_ts).to_pydatetime()
        max_dt = pd.to_datetime(max_ts).to_pydatetime()

    # Use plain date objects for the slider to avoid pandas.Timestamp being passed to Streamlit
    min_date = min_dt.date()
    max_date = max_dt.date()

    # Preset checkbox: prefer a September 1 -> December 1 window using the data's year
    use_sep_dec_preset = st.checkbox("Use Sep 1 → Dec 1 preset for slider", value=False)
    if use_sep_dec_preset:
        data_year = max_date.year
        preset_start = datetime.date(data_year, 9, 1)
        preset_end = datetime.date(data_year, 12, 1)
        # clip preset to available data range
        preset_start = max(min_date, preset_start)
        preset_end = min(max_date, preset_end)
        default_value = (preset_start, preset_end)
    else:
        default_value = (min_date, max_date)

    start_date, end_date = st.slider(
        "Date range",
        value=default_value,
        min_value=min_date,
        max_value=max_date,
    )
    # Filter by plain dates to avoid timezone-aware vs naive Timestamp comparisons
    try:
        sel = sel[(sel["gameDate"].dt.date >= start_date) & (sel["gameDate"].dt.date <= end_date)]
    except Exception:
        # Fallback: construct timezone-aware Timestamps matching the series dtype
        tz = sel["gameDate"].dt.tz
        if tz is not None:
            start_ts = pd.Timestamp(start_date).tz_localize(tz)
            end_ts = pd.Timestamp(end_date).tz_localize(tz)
        else:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
        sel = sel[(sel["gameDate"] >= start_ts) & (sel["gameDate"] <= end_ts)]

    # main charts
    st.subheader("Points Over Time")
    fig = px.line(sel, x="gameDate", y="points", color="fullName", markers=True, title="Points per game")

    # add rolling average (5-game) per player
    for p in players:
        tmp = sel[sel["fullName"] == p].set_index("gameDate").sort_index()
        if len(tmp) >= 3:
            roll = tmp["points"].rolling(5, min_periods=1).mean()
            fig.add_scatter(x=roll.index, y=roll.values, mode="lines", name=f"{p} 5g avg", line=dict(dash="dash"))

    st.plotly_chart(fig, use_container_width=True)

    # summary metrics
    st.subheader("Summary Metrics")
    metrics = []
    for p in players:
        t = sel[sel["fullName"] == p]
        metrics.append(
            {
                "player": p,
                "games": len(t),
                "avg_points": float(t["points"].mean()) if len(t) else 0.0,
                "avg_minutes": float(t["numMinutes"].mean()) if "numMinutes" in t else np.nan,
                "recent_3g": float(t["rolling_3g_points"].iloc[-1]) if ("rolling_3g_points" in t and len(t)) else np.nan,
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("player")
    st.dataframe(metrics_df)

    # bar chart comparing average points/assists/rebounds
    st.subheader("Average Box Score Comparison")
    agg = sel.groupby("fullName").agg(
        avg_points=("points", "mean"),
        avg_assists=("assists", "mean"),
        avg_rebounds=("reboundsTotal", "mean"),
    )
    agg = agg.loc[players]
    fig2 = px.bar(agg.reset_index().melt(id_vars=["fullName"], var_name="metric", value_name="value"), x="fullName", y="value", color="metric", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # show raw table if requested
    with st.expander("Show raw rows for selected players"):
        st.write(sel[["gameDate", "fullName", "playerteamName", "opponentteamName", "points", "assists", "reboundsTotal"]].head(200))


if __name__ == "__main__":
    main()
