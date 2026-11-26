import os
import re
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st


# ==========================================
# MATCHIQ - FREE FOOTBALL DATA PLATFORM
# ==========================================

st.set_page_config(
    page_title="MATCHIQ",
    page_icon="⚽",
    layout="wide"
)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "oladeeayo/open-matchday/master"
)

DATA_URL = f"{BASE_URL}/data"

DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

st.title("⚽ MATCHIQ")
st.caption("Football Intelligence & Prediction Platform")



# ==========================================
# LOCAL HISTORICAL DATA
# ==========================================

def load_local_data(filename):
    path = os.path.join(DATA_FOLDER, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


historical = load_local_data(
    "MATCHIQ_Historical_Database.csv"
)

historical_predictions = load_local_data(
    "MATCHIQ_Final_V3_Predictions.csv"
)



# ==========================================
# FREE DATA CONNECTION
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_file(url):
    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.text


@st.cache_data(ttl=3600, show_spinner=False)
def get_competitions():

    url = f"{BASE_URL}/LEAGUE_INDEX.md"

    content = download_file(url)

    competitions = []

    for line in content.splitlines():

        if not line.startswith("|"):
            continue

        parts = [
            item.strip()
            for item in line.split("|")
        ]

        if len(parts) < 4:
            continue

        folder = parts[1].strip("` ")
        league = parts[2].strip()
        country = parts[3].strip()

        if (
            folder
            and league
            and country
            and folder.lower() != "folder"
            and not folder.startswith("---")
        ):

            competitions.append({
                "folder": folder,
                "league": league,
                "country": country
            })

    return pd.DataFrame(competitions)


@st.cache_data(ttl=3600, show_spinner=False)
def get_competition_results(folder):

    url = (
        f"{DATA_URL}/{folder}/results_2627.csv"
    )

    content = download_file(url)

    return pd.read_csv(
        StringIO(content)
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_competition_fixtures(folder):

    url = (
        f"{DATA_URL}/{folder}/fixtures_2627.csv"
    )

    content = download_file(url)

    return pd.read_csv(
        StringIO(content)
    )



# ==========================================
# TEAM RECENT FORM
# ==========================================

def get_team_recent_matches(results, team_name, limit=5):

    if results.empty:
        return pd.DataFrame()

    required_columns = [
        "date",
        "home",
        "away",
        "ft_home",
        "ft_away"
    ]

    if not all(
        column in results.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    matches = results[
        (results["home"] == team_name)
        | (results["away"] == team_name)
    ].copy()

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce"
    )

    matches = matches.dropna(
        subset=["date", "ft_home", "ft_away"]
    )

    matches = matches.sort_values(
        "date",
        ascending=False
    ).head(limit)

    records = []

    for _, match in matches.iterrows():

        is_home = match["home"] == team_name

        if is_home:

            goals_for = match["ft_home"]
            goals_against = match["ft_away"]
            opponent = match["away"]
            venue = "Home"

        else:

            goals_for = match["ft_away"]
            goals_against = match["ft_home"]
            opponent = match["home"]
            venue = "Away"

        if goals_for > goals_against:
            outcome = "W"
        elif goals_for == goals_against:
            outcome = "D"
        else:
            outcome = "L"

        records.append({
            "Date": match["date"].strftime("%Y-%m-%d"),
            "Opponent": opponent,
            "Venue": venue,
            "Goals For": int(goals_for),
            "Goals Against": int(goals_against),
            "Result": outcome
        })

    return pd.DataFrame(records)



# ==========================================
# TEAM FORM SUMMARY
# ==========================================

def calculate_team_form(results, team_name):

    matches = get_team_recent_matches(
        results,
        team_name,
        limit=5
    )

    if matches.empty:
        return None

    wins = (matches["Result"] == "W").sum()
    draws = (matches["Result"] == "D").sum()
    losses = (matches["Result"] == "L").sum()

    points = wins * 3 + draws

    return {
        "played": len(matches),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "points": int(points),
        "goals_for": int(matches["Goals For"].sum()),
        "goals_against": int(matches["Goals Against"].sum()),
        "matches": matches
    }



# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("MATCHIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Competitions",
        "Upcoming Fixtures",
        "Team Form",
        "Historical Records"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Data source: open-matchday"
)

st.sidebar.caption(
    "Free CSV football data"
)



# ==========================================
# LOAD COMPETITIONS
# ==========================================

try:

    competitions = get_competitions()

except Exception as error:

    competitions = pd.DataFrame()

    st.error(
        f"Could not connect to the free football dataset: {error}"
    )



# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.header("Platform Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Historical Matches",
        f"{len(historical):,}"
    )

    col2.metric(
        "Historical Predictions",
        f"{len(historical_predictions):,}"
    )

    col3.metric(
        "Available Competitions",
        f"{len(competitions):,}"
    )

    st.divider()

    st.subheader("Data Connection")

    if not competitions.empty:

        st.success(
            "Free football data directory connected."
        )

        st.write(
            f"MATCHIQ can access {len(competitions)} "
            "listed competitions."
        )

    else:

        st.warning(
            "Competition data is currently unavailable."
        )

    st.subheader("Historical Model Performance")

    if not historical_predictions.empty:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "V3 Accuracy",
            "45.34%"
        )

        col2.metric(
            "Brier Score",
            "0.6504"
        )

        col3.metric(
            "Log Loss",
            "1.0851"
        )

        st.caption(
            "Historical backtest figures, not verified "
            "current-season prediction accuracy."
        )

    else:

        st.info(
            "Historical prediction records have not been loaded."
        )

    st.divider()

    st.subheader("About MATCHIQ")

    st.write(
        "MATCHIQ is designed to retrieve football data, "
        "analyse team performance and develop statistical "
        "match predictions."
    )

    st.write(
        "The current data connection uses publicly accessible "
        "CSV files rather than a paid API subscription."
    )

    st.warning(
        "This dataset is updated daily, not live. "
        "Missing statistics are treated as unavailable, "
        "not as zero."
    )



# ==========================================
# COMPETITIONS
# ==========================================

elif page == "Competitions":

    st.header("Available Football Competitions")

    if competitions.empty:

        st.warning(
            "No competitions could be loaded."
        )

    else:

        search = st.text_input(
            "Search competition or country"
        )

        display = competitions.copy()

        if search:

            mask = (
                display["league"].str.contains(
                    search,
                    case=False,
                    na=False
                )
                |
                display["country"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            )

            display = display[mask]

        st.metric(
            "Competitions Found",
            len(display)
        )

        st.dataframe(
            display[["league", "country"]],
            width="stretch",
            hide_index=True
        )



# ==========================================
# UPCOMING FIXTURES
# ==========================================

elif page == "Upcoming Fixtures":

    st.header("Upcoming Fixtures")

    if competitions.empty:

        st.warning(
            "Competition directory is unavailable."
        )

    else:

        selected_league = st.selectbox(
            "Select Competition",
            competitions["league"].tolist()
        )

        selected = competitions[
            competitions["league"] == selected_league
        ].iloc[0]

        folder = selected["folder"]

        try:

            with st.spinner(
                "Retrieving fixtures..."
            ):

                fixtures = get_competition_fixtures(
                    folder
                )

            if fixtures.empty:

                st.info(
                    "No fixtures are currently listed."
                )

            else:

                fixtures["date"] = pd.to_datetime(
                    fixtures["date"],
                    errors="coerce"
                )

                today = pd.Timestamp.now(
                    tz="Africa/Lagos"
                ).date()

                upcoming_fixtures = fixtures[
                    fixtures["date"].dt.date >= today
                ].copy()

                upcoming_fixtures = upcoming_fixtures.sort_values(
                    "date"
                )

                st.metric(
                    "Upcoming Fixtures",
                    len(upcoming_fixtures)
                )

                st.dataframe(
                    upcoming_fixtures,
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "Download Fixtures",
                    data=upcoming_fixtures.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="MATCHIQ_Upcoming_Fixtures.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Could not retrieve fixtures: {error}"
            )



# ==========================================
# TEAM FORM
# ==========================================

elif page == "Team Form":

    st.header("Team Performance & Recent Form")

    if competitions.empty:

        st.warning(
            "Competition directory is unavailable."
        )

    else:

        selected_league = st.selectbox(
            "Select Competition",
            competitions["league"].tolist()
        )

        selected = competitions[
            competitions["league"] == selected_league
        ].iloc[0]

        folder = selected["folder"]

        try:

            with st.spinner(
                "Loading completed matches..."
            ):

                results = get_competition_results(
                    folder
                )

            if results.empty:

                st.warning(
                    "No completed matches are available."
                )

            else:

                teams = sorted(
                    set(results["home"].dropna().tolist())
                    |
                    set(results["away"].dropna().tolist())
                )

                selected_team = st.selectbox(
                    "Select Team",
                    teams
                )

                form = calculate_team_form(
                    results,
                    selected_team
                )

                if form is None:

                    st.warning(
                        "Insufficient completed matches "
                        "to calculate recent form."
                    )

                else:

                    st.subheader(selected_team)

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Played",
                        form["played"]
                    )

                    col2.metric(
                        "Wins",
                        form["wins"]
                    )

                    col3.metric(
                        "Draws",
                        form["draws"]
                    )

                    col4.metric(
                        "Losses",
                        form["losses"]
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Points",
                        form["points"],
                        f"Out of {form['played'] * 3}"
                    )

                    col2.metric(
                        "Goals Scored",
                        form["goals_for"]
                    )

                    col3.metric(
                        "Goals Conceded",
                        form["goals_against"]
                    )

                    st.subheader(
                        "Last Five Completed Matches"
                    )

                    st.dataframe(
                        form["matches"],
                        width="stretch",
                        hide_index=True
                    )

                    st.caption(
                        "Form is calculated from the latest "
                        "completed matches available in the dataset."
                    )

        except Exception as error:

            st.error(
                f"Could not retrieve team data: {error}"
            )



# ==========================================
# HISTORICAL RECORDS
# ==========================================

elif page == "Historical Records":

    st.header("MATCHIQ Historical Records")

    st.subheader("Historical Match Database")

    if historical.empty:

        st.info(
            "Historical match database is unavailable."
        )

    else:

        st.metric(
            "Historical Matches",
            len(historical)
        )

        st.dataframe(
            historical,
            width="stretch"
        )

        st.download_button(
            "Download Historical Database",
            data=historical.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="MATCHIQ_Historical_Database.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Historical Model Predictions")

    if historical_predictions.empty:

        st.info(
            "Historical prediction database is unavailable."
        )

    else:

        st.metric(
            "Historical Predictions",
            len(historical_predictions)
        )

        st.dataframe(
            historical_predictions,
            width="stretch"
        )

        st.download_button(
            "Download Historical Predictions",
            data=historical_predictions.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="MATCHIQ_Final_V3_Predictions.csv",
            mime="text/csv"
        )



# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MATCHIQ | Football data and statistical analysis."
)

st.caption(
    "External data source: open-matchday "
    "| Updated daily | Not an in-play data feed"
)
