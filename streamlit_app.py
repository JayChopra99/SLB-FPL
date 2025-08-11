import streamlit as st
import pandas as pd
import requests
from pandas import json_normalize

# ---- Page Config ----
st.set_page_config(
    page_title="SLB-FPL Dashboard",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/d/d6/SLB_Logo_2022.svg"
)

# ---- Header Layout ----
col1, col2 = st.columns([0.2, 1])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/d/d6/SLB_Logo_2022.svg",
        use_container_width=True
    )
with col2:
    st.title("SLB-FPL Dashboard")

st.markdown(
    """
    Tracking the SLB Mini-League (FPL Elite) for [Fantasy Premier League](https://fantasy.premierleague.com/leagues/334417/standings/c).
    """
)

st.write("")  # spacing

# ---- SelectBox ----
gw_option = st.selectbox(
    "Select GameWeek?",
    list(range(1, 39)),
    index=0,
    format_func=lambda x: f"GameWeek {x}",
    help="Select the gameweek to view data for."
)

st.write(f"You selected: GameWeek {gw_option}")



# ---- Data Function ----
@st.cache_data
def get_ml_data(league_id=334417, page_id=1, phase=1):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page_id}&phase={phase}"
    r = requests.get(url).json()

    new_entries_list = r.get('new_entries', {}).get('results', [])
    players_df = pd.json_normalize(new_entries_list)
    players_df = players_df[['entry_name', 'player_first_name', 'player_last_name']]
    players_df.columns = ['Team Name', 'First Name', 'Last Name']

    return players_df

ml_df = get_ml_data(phase=gw_option)

# ---- Display Data ----
st.dataframe(ml_df)

# ---- Fines Data Function ----
@st.cache_data
def fines_data(league_id=334417):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
    r = requests.get(url).json()

    new_entries_list = r.get('new_entries', {}).get('results', [])
    fines_df = pd.json_normalize(new_entries_list)
    fines_df = fines_df[['player_first_name', 'player_last_name']]
    fines_df.columns = ['First Name', 'Last Name']
    fines_df['Total Fine'] = 0
    # Example fines table as dictionary
    fines_data = {
        1: {'Weekly': 0, 'Monthly': 0, 'Annual': 0},
        2: {'Weekly': 1.00, 'Monthly': 1.50, 'Annual': 10.00},
        3: {'Weekly': 2.00, 'Monthly': 3.00, 'Annual': 20.00},
        4: {'Weekly': 3.00, 'Monthly': 4.50, 'Annual': 35.00},
        5: {'Weekly': 4.00, 'Monthly': 6.00, 'Annual': 55.00},
    }

    # Convert fines_data to DataFrame for easier merging
    fines_df2 = pd.DataFrame.from_dict(fines_data, orient='index').reset_index()
    fines_df2.rename(columns={'index': 'Rank'}, inplace=True)

    return fines_df

fines_df = fines_data()  # <-- call the function here

# ---- Display Fines Data ----
st.dataframe(fines_df)
