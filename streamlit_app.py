import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests
import plotly.express as px

# ---- Page Config ----
st.set_page_config(
    page_title="SLB-FPL Dashboard",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/d/d6/SLB_Logo_2022.svg",
    layout="wide"
)

# ---- Data Functions ----
@st.cache_data(ttl=60)
def get_ml_data(league_id=334417, page_id=1, phase=1):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page_id}&phase={phase}"
    r = requests.get(url).json()
    new_entries_list = r.get('new_entries', {}).get('results', [])
    players_df = pd.json_normalize(new_entries_list)
    players_df = players_df[['entry_name', 'player_first_name', 'player_last_name']]
    players_df.columns = ['Team Name', 'First Name', 'Last Name']
    return players_df

@st.cache_data(ttl=60)
def fines_data(league_id=334417):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
    r = requests.get(url).json()
    new_entries_list = r.get('new_entries', {}).get('results', [])
    fines_df = pd.json_normalize(new_entries_list)
    fines_df = fines_df[['player_first_name', 'player_last_name']]
    fines_df.columns = ['First Name', 'Last Name']
    fines_df['Player Name'] = fines_df['First Name'] + " " + fines_df['Last Name']
    fines_df = fines_df.drop(columns=['First Name', 'Last Name'])
    fines_df['Total Fine'] = 5  # placeholder
    return fines_df

@st.cache_data(ttl=60)
def users_data(league_id=334417):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
    r = requests.get(url).json()
    new_entries_list = r.get('new_entries', {}).get('results', [])
    users_df = pd.json_normalize(new_entries_list)
    users_df = users_df[['player_first_name', 'player_last_name']]
    users_df.columns = ['First Name', 'Last Name']
    users_df['Player Name'] = users_df['First Name'] + " " + users_df['Last Name']
    users_df = users_df.drop(columns=['First Name', 'Last Name'])
    users_df['Total Fine'] = 5  # placeholder
    return users_df

@st.cache_data(ttl=60)
def gw_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url).json()
    events_list = r.get('events', [])
    gw_df = pd.DataFrame(events_list)[['id', 'name', 'deadline_time', 'finished', 'average_entry_score']]
    gw_df.columns = ['GW ID', 'GW Name', 'Deadline', 'Finished', 'Average Points']
    return gw_df



# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["Home", "GW Data", "Fines"],
        icons= ["house-door-fill","database","currency-dollar"],
        menu_icon= [""],
        default_index=0
    )
    

# ---- Main Page Layout ----
if selected == "Home":
    header_col1, header_col2 = st.columns([0.1, 0.85])
    with header_col1:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/d/d6/SLB_Logo_2022.svg",
            use_container_width=True
        )
    with header_col2:
        st.title("SLB-FPL Dashboard")
        st.markdown(
            """
            Tracking the SLB Mini-League (FPL Elite) for 
            [Fantasy Premier League](https://fantasy.premierleague.com/leagues/334417/standings/c).
            """
        )
    st.markdown("---")

 
  


elif selected == "GW Data":
    gw_option = st.selectbox(
        "Select GameWeek:",
        list(range(1, 39)),
        index=0,
        format_func=lambda x: f"GameWeek {x}",
        help="Select the gameweek to view data for."
    )
    ml_df = get_ml_data(phase=gw_option)
    fines_df = fines_data()

    st.subheader(f"League Table - GameWeek {gw_option}")
    st.dataframe(ml_df, use_container_width=True)
    st.markdown("---")
    gw_df = gw_data()

    st.subheader(f"GameWeek")
    st.dataframe(gw_df, use_container_width=True)
    st.markdown("---")

     # ---- Get and display GW vs Player chart ----
    users_df = users_data()
    fig = px.scatter(
        users_df,
        x="Total Fine",
        y="Player Name",
        size="Total Fine",
        color="Total Fine",
        title="GameWeek Performance by Player",
        labels={"GW Points": "Points"},
        hover_name="Player Name"
    )
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Fines":
    fines_df = fines_data()

    fines_col1, fines_col2 = st.columns([0.5, 0.5])
    with fines_col1:
        st.subheader("Fines Overview")
        st.dataframe(fines_df, use_container_width=True)

    with fines_col2:
        st.subheader("Fines Distribution")
        fig = px.pie(
            fines_df,
            values='Total Fine',
            names='Player Name',
            hover_data=['Total Fine'],
            labels={'Total Fine': '£'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300, margin=dict(t=0))
        st.plotly_chart(fig, use_container_width=True)






