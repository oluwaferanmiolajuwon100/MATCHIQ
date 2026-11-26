import math
import json
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

    url = f"{BASE_URL}/build_summary.json"

    content = download_file(url)
    summary = json.loads(content)

    competitions = []

    for item in summary.get("leagues", []):

        folder_path = item.get("folder", "")
        folder = folder_path.split("/")[-1]
        league = item.get("league", "")

        if folder and league:
            competitions.append({
                "folder": folder,
                "league": league,
                "country": folder.split("_")[0]
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
        "Match Predictions",
        "Backtesting",
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
# MATCH PREDICTIONS
# ==========================================

elif page == "Match Predictions":

    st.header("MATCHIQ Match Predictions")

    if competitions.empty:

        st.warning("Competition data is unavailable.")

    else:

        selected_league = st.selectbox(
            "Select Competition",
            competitions["league"].tolist(),
            key="prediction_league"
        )

        selected = competitions[
            competitions["league"] == selected_league
        ].iloc[0]

        folder = selected["folder"]

        try:

            with st.spinner("Loading match data..."):

                results = get_competition_results(folder)
                fixtures = get_competition_fixtures(folder)

            results["date"] = pd.to_datetime(
                results["date"],
                errors="coerce"
            )

            fixtures["date"] = pd.to_datetime(
                fixtures["date"],
                errors="coerce"
            )

            results = results.dropna(
                subset=["date", "home", "away", "ft_home", "ft_away"]
            )

            today = pd.Timestamp.now(
                tz="Africa/Lagos"
            ).date()

            fixtures = fixtures[
                fixtures["date"].dt.date >= today
            ].copy()

            if fixtures.empty:

                st.info("No upcoming fixtures are available.")

            else:

                fixtures = fixtures.sort_values("date")

                fixture_index = st.selectbox(
                    "Select Match",
                    options=range(len(fixtures)),
                    format_func=lambda i: (
                        f"{fixtures.iloc[i]['date'].strftime('%d %b %Y')} | "
                        f"{fixtures.iloc[i]['home']} vs "
                        f"{fixtures.iloc[i]['away']}"
                    )
                )

                match = fixtures.iloc[fixture_index]

                home_team = match["home"]
                away_team = match["away"]

                if st.button(
                    "Generate Match Prediction",
                    type="primary"
                ):

                    home_form = calculate_team_form(
                        results,
                        home_team
                    )

                    away_form = calculate_team_form(
                        results,
                        away_team
                    )

                    if home_form is None or away_form is None:

                        st.warning(
                            "Insufficient recent match data "
                            "for one or both teams. No prediction generated."
                        )

                    elif home_form["played"] < 3 or away_form["played"] < 3:

                        st.warning(
                            "At least three recent completed matches "
                            "are required for each team."
                        )

                    else:

                        avg_home_goals = results["ft_home"].mean()
                        avg_away_goals = results["ft_away"].mean()

                        avg_team_goals = (
                            results["ft_home"].sum()
                            + results["ft_away"].sum()
                        ) / (2 * len(results))

                        if avg_team_goals <= 0:

                            st.warning(
                                "Competition scoring data is insufficient."
                            )

                        else:

                            home_attack = (
                                home_form["goals_for"]
                                / home_form["played"]
                            ) / avg_team_goals

                            home_defence = (
                                home_form["goals_against"]
                                / home_form["played"]
                            ) / avg_team_goals

                            away_attack = (
                                away_form["goals_for"]
                                / away_form["played"]
                            ) / avg_team_goals

                            away_defence = (
                                away_form["goals_against"]
                                / away_form["played"]
                            ) / avg_team_goals

                            expected_home = (
                                avg_home_goals
                                * home_attack
                                * away_defence
                            )

                            expected_away = (
                                avg_away_goals
                                * away_attack
                                * home_defence
                            )

                            expected_home = max(
                                0.05,
                                min(expected_home, 5)
                            )

                            expected_away = max(
                                0.05,
                                min(expected_away, 5)
                            )

                            def poisson_probability(goals, expected):

                                return (
                                    math.exp(-expected)
                                    * expected ** goals
                                    / math.factorial(goals)
                                )

                            score_probabilities = {}

                            for h in range(9):
                                for a in range(9):

                                    probability = (
                                        poisson_probability(
                                            h,
                                            expected_home
                                        )
                                        *
                                        poisson_probability(
                                            a,
                                            expected_away
                                        )
                                    )

                                    score_probabilities[(h, a)] = probability

                            total_probability = sum(
                                score_probabilities.values()
                            )

                            score_probabilities = {
                                score: probability / total_probability
                                for score, probability
                                in score_probabilities.items()
                            }

                            home_win = sum(
                                p for (h, a), p
                                in score_probabilities.items()
                                if h > a
                            )

                            draw = sum(
                                p for (h, a), p
                                in score_probabilities.items()
                                if h == a
                            )

                            away_win = sum(
                                p for (h, a), p
                                in score_probabilities.items()
                                if h < a
                            )

                            over_25 = sum(
                                p for (h, a), p
                                in score_probabilities.items()
                                if h + a >= 3
                            )

                            under_25 = 1 - over_25

                            btts = sum(
                                p for (h, a), p
                                in score_probabilities.items()
                                if h > 0 and a > 0
                            )

                            most_likely_score = max(
                                score_probabilities,
                                key=score_probabilities.get
                            )

                            st.divider()

                            st.subheader(
                                f"{home_team} vs {away_team}"
                            )

                            col1, col2 = st.columns(2)

                            col1.metric(
                                "Expected Home Goals",
                                f"{expected_home:.2f}"
                            )

                            col2.metric(
                                "Expected Away Goals",
                                f"{expected_away:.2f}"
                            )

                            st.subheader("1X2 Probabilities")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Home Win",
                                f"{home_win * 100:.1f}%"
                            )

                            col2.metric(
                                "Draw",
                                f"{draw * 100:.1f}%"
                            )

                            col3.metric(
                                "Away Win",
                                f"{away_win * 100:.1f}%"
                            )

                            st.subheader("Goals Markets")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Over 2.5 Goals",
                                f"{over_25 * 100:.1f}%"
                            )

                            col2.metric(
                                "Under 2.5 Goals",
                                f"{under_25 * 100:.1f}%"
                            )

                            col3.metric(
                                "Both Teams to Score",
                                f"{btts * 100:.1f}%"
                            )

                            st.subheader("Most Likely Scoreline")

                            st.metric(
                                "Predicted Score",
                                f"{most_likely_score[0]} - "
                                f"{most_likely_score[1]}"
                            )

                            st.caption(
                                "Model: basic Poisson baseline using "
                                "recent five-match team form and "
                                "competition scoring averages. "
                                "Probabilities are estimates, not guarantees."
                            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )

# ==========================================
# BACKTESTING
# ==========================================

elif page == "Backtesting":

    st.header("MATCHIQ Historical Backtesting")

    st.write(
        "Testing predictions against completed matches "
        "using only data available before each match."
    )

    selected_league = st.selectbox(
        "Select Competition",
        competitions["league"].tolist(),
        key="backtest_league"
    )

    selected = competitions[
        competitions["league"] == selected_league
    ].iloc[0]

    try:
        results = get_competition_results(selected["folder"])

        results["date"] = pd.to_datetime(
            results["date"], errors="coerce"
        )
        results["ft_home"] = pd.to_numeric(
            results["ft_home"], errors="coerce"
        )
        results["ft_away"] = pd.to_numeric(
            results["ft_away"], errors="coerce"
        )

        results = results.dropna(
            subset=["date", "home", "away", "ft_home", "ft_away"]
        ).sort_values("date").reset_index(drop=True)

        if len(results) < 30:
            st.warning("Not enough completed matches for backtesting.")

        else:
            predictions = []

            with st.spinner("Testing historical matches..."):

                for i in range(len(results)):

                    match = results.iloc[i]

                    previous = results[
                        results["date"] < match["date"]
                    ]

                    if len(previous) < 20:
                        continue

                    home_form = calculate_team_form(
                        previous, match["home"]
                    )
                    away_form = calculate_team_form(
                        previous, match["away"]
                    )

                    if home_form is None or away_form is None:
                        continue

                    if home_form["played"] < 3 or away_form["played"] < 3:
                        continue

                    avg_home = previous["ft_home"].mean()
                    avg_away = previous["ft_away"].mean()

                    avg_team = (
                        previous["ft_home"].sum()
                        + previous["ft_away"].sum()
                    ) / (2 * len(previous))

                    if avg_team <= 0:
                        continue

                    home_attack = (
                        home_form["goals_for"] / home_form["played"]
                    ) / avg_team
                    home_defence = (
                        home_form["goals_against"] / home_form["played"]
                    ) / avg_team
                    away_attack = (
                        away_form["goals_for"] / away_form["played"]
                    ) / avg_team
                    away_defence = (
                        away_form["goals_against"] / away_form["played"]
                    ) / avg_team

                    expected_home = max(
                        0.05,
                        min(avg_home * home_attack * away_defence, 5)
                    )
                    expected_away = max(
                        0.05,
                        min(avg_away * away_attack * home_defence, 5)
                    )

                    def poisson_probability(goals, expected):
                        return (
                            math.exp(-expected)
                            * expected ** goals
                            / math.factorial(goals)
                        )

                    score_probs = {}

                    for h in range(9):
                        for a in range(9):
                            score_probs[(h, a)] = (
                                poisson_probability(h, expected_home)
                                * poisson_probability(a, expected_away)
                            )

                    total = sum(score_probs.values())

                    if total <= 0:
                        continue

                    score_probs = {
                        score: probability / total
                        for score, probability in score_probs.items()
                    }

                    home_win = sum(
                        p for (h, a), p in score_probs.items() if h > a
                    )
                    draw = sum(
                        p for (h, a), p in score_probs.items() if h == a
                    )
                    away_win = sum(
                        p for (h, a), p in score_probs.items() if h < a
                    )

                    actual_home = int(match["ft_home"])
                    actual_away = int(match["ft_away"])

                    actual = (
                        "H" if actual_home > actual_away
                        else "A" if actual_home < actual_away
                        else "D"
                    )

                    probabilities = {
                        "H": home_win,
                        "D": draw,
                        "A": away_win
                    }

                    predicted = max(
                        probabilities, key=probabilities.get
                    )

                    predictions.append({
                        "actual": actual,
                        "predicted": predicted,
                        "home_probability": home_win,
                        "draw_probability": draw,
                        "away_probability": away_win
                    })

            if not predictions:
                st.warning("No matches had enough earlier data to test.")

            else:
                test = pd.DataFrame(predictions)

                correct = (
                    test["actual"] == test["predicted"]
                ).sum()

                accuracy = correct / len(test)

                brier_scores = []

                for _, row in test.iterrows():

                    actual_values = {
                        "H": 1 if row["actual"] == "H" else 0,
                        "D": 1 if row["actual"] == "D" else 0,
                        "A": 1 if row["actual"] == "A" else 0
                    }

                    brier = (
                        (row["home_probability"] - actual_values["H"]) ** 2
                        + (row["draw_probability"] - actual_values["D"]) ** 2
                        + (row["away_probability"] - actual_values["A"]) ** 2
                    )

                    brier_scores.append(brier)

                st.divider()

                col1, col2, col3 = st.columns(3)

                col1.metric("Matches Tested", len(test))
                col2.metric("1X2 Accuracy", f"{accuracy * 100:.2f}%")
                col3.metric(
                    "Brier Score",
                    f"{sum(brier_scores) / len(brier_scores):.4f}"
                )

                st.subheader("Prediction Breakdown")
                st.write(f"Correct predictions: {correct}")
                st.write(f"Incorrect predictions: {len(test) - correct}")

                st.dataframe(
                    test.tail(20),
                    use_container_width=True
                )

                st.caption(
                    "Historical test only. Results are not a guarantee "
                    "of future prediction accuracy or betting profit."
                )

    except Exception as error:
        st.error(f"Backtesting error: {error}")
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
