import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

# ==========================================
# MATCHIQ - FOOTBALL INTELLIGENCE PLATFORM
# ==========================================

st.set_page_config(
    page_title="MATCHIQ",
    page_icon="⚽",
    layout="wide"
)

API_BASE = "https://v3.football.api-sports.io"
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

st.title("⚽ MATCHIQ")
st.caption("Football Intelligence & Prediction Platform")


# ==========================================
# DATA LOADING
# ==========================================

def load_data(filename):
    path = os.path.join(DATA_FOLDER, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


predictions = load_data("MATCHIQ_Final_V3_Predictions.csv")
historical = load_data("MATCHIQ_Historical_Database.csv")
upcoming = load_data("MATCHIQ_Upcoming_Fixtures.csv")


# ==========================================
# API CONFIGURATION
# ==========================================

def get_api_key():
    try:
        return st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        return ""


def api_get(endpoint, params=None):
    key = get_api_key()

    if not key:
        raise ValueError(
            "API key not found. Check Streamlit Secrets."
        )

    headers = {
        "x-apisports-key": key
    }

    response = requests.get(
        f"{API_BASE}/{endpoint}",
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    errors = result.get("errors")

    if errors:
        raise ValueError(str(errors))

    return result


# ==========================================
# LIVE FIXTURE RETRIEVAL
# ==========================================

def get_live_fixtures():
    now = datetime.now(ZoneInfo("Africa/Lagos"))
    today = now.strftime("%Y-%m-%d")

    result = api_get(
        "fixtures",
        {
            "date": today,
            "timezone": "Africa/Lagos"
        }
    )

    records = []

    for item in result.get("response", []):
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        status = fixture.get("status", {})

        records.append({
            "fixture_id": fixture.get("id"),
            "date": fixture.get("date"),
            "league_id": league.get("id"),
            "league": league.get("name"),
            "country": league.get("country"),
            "season": league.get("season"),
            "home_team_id": teams.get("home", {}).get("id"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team_id": teams.get("away", {}).get("id"),
            "away_team": teams.get("away", {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "status": status.get("short"),
            "data_status": "CURRENT FIXTURE RETRIEVED",
            "prediction_status": "NOT GENERATED"
        })

    return pd.DataFrame(records)
# RECENT TEAM MATCH RETRIEVAL
# ==========================================

def get_recent_team_matches(team_id):
    now = datetime.now(ZoneInfo("Africa/Lagos"))
    yesterday = now.date() - pd.Timedelta(days=1)
    start_date = now.date() - pd.Timedelta(days=60)

    result = api_get(
        "fixtures",
        {
            "team": int(team_id),
            "from": start_date.strftime("%Y-%m-%d"),
            "to": yesterday.strftime("%Y-%m-%d"),
            "timezone": "Africa/Lagos"
        }
    )

    matches = result.get("response", [])

    completed = [
        match for match in matches
        if match.get("fixture", {}).get("status", {}).get("short")
        in ["FT", "AET", "PEN"]
    ]

    completed.sort(
        key=lambda match: match.get("fixture", {}).get("date", ""),
        reverse=True
    )

    return completed[:5]



# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("MATCHIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Historical Predictions",
       "Upcoming Fixtures",
"Live API Connection",
"Recent Team Data Test"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("MATCHIQ Football Analytics")


# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.header("Platform Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Historical Matches", f"{len(historical):,}")
    col2.metric("Historical Predictions", f"{len(predictions):,}")
    col3.metric("Uploaded Upcoming Fixtures", f"{len(upcoming):,}")

    st.divider()

    st.subheader("Historical Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("V3 Accuracy", "45.34%")
    col2.metric("Brier Score", "0.6504")
    col3.metric("Log Loss", "1.0851")

    st.info(
        "Chronological holdout accuracy was 51.01% "
        "across 447 matches. This is historical evaluation, "
        "not a guarantee of future performance."
    )

    st.subheader("About MATCHIQ")

    st.write(
        "MATCHIQ combines football data retrieval, "
        "historical analysis and statistical prediction models."
    )

    st.write(
        "The current historical model uses an upgraded "
        "Poisson prediction engine."
    )


# ==========================================
# HISTORICAL PREDICTIONS
# ==========================================

elif page == "Historical Predictions":

    st.header("Historical Predictions")

    if predictions.empty:
        st.warning("Historical prediction data is unavailable.")

    else:
        search = st.text_input("Search team or competition")

        filtered = predictions.copy()

        if search:
            text_columns = filtered.select_dtypes(
                include=["object"]
            ).columns

            if len(text_columns):
                mask = filtered[text_columns].astype(str).apply(
                    lambda col: col.str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

        st.metric("Matching Records", len(filtered))

        st.dataframe(
            filtered,
            width="stretch"
        )

        st.download_button(
            "Download Historical Predictions",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="MATCHIQ_Historical_Predictions.csv",
            mime="text/csv"
        )


# ==========================================
# UPCOMING FIXTURES
# ==========================================

elif page == "Upcoming Fixtures":

    st.header("Upcoming Fixtures")

    st.caption(
        "Uploaded historical fixture snapshot. "
        "Use Live API Connection to retrieve today's fixtures."
    )

    if upcoming.empty:
        st.warning("No uploaded upcoming fixture records found.")

    else:
        search = st.text_input("Search teams or competitions")

        filtered = upcoming.copy()

        if search:
            text_columns = filtered.select_dtypes(
                include=["object"]
            ).columns

            if len(text_columns):
                mask = filtered[text_columns].astype(str).apply(
                    lambda col: col.str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ).any(axis=1)

                filtered = filtered[mask]

        st.metric("Available Records", len(filtered))

        st.dataframe(
            filtered,
            width="stretch"
        )

        st.download_button(
            "Download Upcoming Fixtures",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="MATCHIQ_Upcoming_Fixtures.csv",
            mime="text/csv"
        )


# ==========================================
# LIVE API CONNECTION
# ==========================================

elif page == "Live API Connection":

    st.header("Live Football Data Connection")

    st.write(
        "Connect MATCHIQ to API-Football and retrieve "
        "the latest fixtures for the current date."
    )

    if get_api_key():
        st.success("API key detected in Streamlit Secrets.")
    else:
        st.error(
            "API key not detected. Check the API_FOOTBALL_KEY "
            "entry in Streamlit Secrets."
        )

    if st.button(
        "Test API Connection and Retrieve Today's Fixtures",
        type="primary"
    ):

        if not get_api_key():
            st.error("API key is missing.")

        else:
            with st.spinner("Connecting to API-Football..."):

                try:
                    status_result = api_get("status")

                    st.success(
                        "API authentication successful."
                    )

                    st.subheader("API Account Information")

                    st.json(status_result.get("response", {}))

                    live_df = get_live_fixtures()

                    st.session_state["live_fixtures"] = live_df

                    st.success(
                        f"Successfully retrieved {len(live_df)} "
                        "fixtures for today."
                    )

                except requests.exceptions.HTTPError as error:
                    st.error(
                        f"API HTTP error: {error}. "
                        "Check your subscription and API access."
                    )

                except requests.exceptions.RequestException as error:
                    st.error(f"Connection error: {error}")

                except Exception as error:
                    st.error(f"API retrieval error: {error}")

    if "live_fixtures" in st.session_state:

        live_df = st.session_state["live_fixtures"]

        st.divider()
        st.subheader("Retrieved Live Fixtures")

        st.metric("Fixtures Retrieved", len(live_df))

        if live_df.empty:
            st.info("The API returned no fixtures for today.")

        else:
            st.dataframe(
                live_df,
                width="stretch"
            )

            st.download_button(
                "Download Live Fixtures",
                data=live_df.to_csv(index=False).encode("utf-8"),
                file_name="MATCHIQ_Live_Fixtures.csv",
                mime="text/csv"
            )

        st.warning(
            "Retrieving fixtures does not automatically generate "
            "predictions. Recent team data and model validation "
            "must be completed first."
        )

# ==========================================
# RECENT TEAM DATA TEST
# ==========================================

elif page == "Recent Team Data Test":

    st.header("Recent Team Data Retrieval Test")

    if "live_fixtures" not in st.session_state:
        st.warning(
            "First retrieve today's fixtures "
            "from the Live API Connection page."
        )

    else:
        live_df = st.session_state["live_fixtures"]

        available = live_df.dropna(
            subset=["home_team_id", "away_team_id"]
        )

        if available.empty:
            st.warning("No teams available for testing.")

        else:
            fixture = available.iloc[0]

            st.write(
                f"Test fixture: {fixture['home_team']} "
                f"vs {fixture['away_team']}"
            )

            if st.button("Test Recent Team Data"):

                for side in ["home", "away"]:

                    team_id = fixture[f"{side}_team_id"]
                    team_name = fixture[f"{side}_team"]

                    try:
                        matches = get_recent_team_matches(team_id)

                        st.success(
                            f"{team_name}: "
                            f"{len(matches)} matches retrieved."
                        )

                        if matches:
                            st.dataframe(
                                pd.json_normalize(matches),
                                width="stretch"
                            )

                    except Exception as error:
                        st.error(
                            f"{team_name}: {error}"
                        )
# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MATCHIQ | Statistical football intelligence. "
    "Predictions are not guaranteed outcomes."
)
