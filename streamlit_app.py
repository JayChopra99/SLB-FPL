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
@st.cache_data(ttl=300)
def get_league_managers(league_id=334417, page_id=1):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page_id}"
    r = requests.get(url).json()
    standings_list = r.get('standings', {}).get('results', [])
    df = pd.json_normalize(standings_list)
    if not df.empty:
        df = df[['entry', 'entry_name', 'player_name']]
        df.columns = ['Team ID', 'Team Name', 'Manager Name']
        return df
    else:
        return pd.DataFrame(columns=['Team ID', 'Team Name', 'Manager Name'])

@st.cache_data(ttl=300)
def get_all_gw_points(league_id=334417, total_gws=38):
    managers_df = get_league_managers(league_id)
    all_data = []

    for _, row in managers_df.iterrows():
        team_id = row['Team ID']
        team_name = row['Team Name']
        manager_name = row['Manager Name']

        # Fetch individual team data
        url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
        r = requests.get(url).json()
        gw_list = r.get('current', [])  # 'current' has a list of dicts for each GW

        for gw in gw_list:
            all_data.append({
                'Team Name': team_name,
                'Manager Name': manager_name,
                'GW': gw['event'],
                'GW Points': gw['points']
            })

    if all_data:
        df = pd.DataFrame(all_data)
        # Pivot so GWs are columns
        pivot_df = df.pivot(index='Manager Name', columns='GW', values='GW Points').fillna(0)
        pivot_df.reset_index(inplace=True)
        return pivot_df
    else:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fines_data(league_id=334417, page_id=1):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page_id}"
    r = requests.get(url).json()
    standings_list = r.get('standings', {}).get('results', [])
    df = pd.json_normalize(standings_list)
    if not df.empty:
        df = df[['entry_name','player_name']]
        df.columns = ['Team Name','Manager Name']
        df['Total Fine'] = 5  # placeholder
        return df
    else:
        return pd.DataFrame(columns=['Team Name', 'Manager Name', 'Total Fine'])

# ---- Sidebar menu ----
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["Home", "GW Data", "Fines"],
        icons=["house-door-fill","database","currency-dollar"],
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
    total_gws = 38
    gw_points_df = get_all_gw_points(total_gws=total_gws)

    if not gw_points_df.empty:
        # Ensure all 38 GWs exist as columns
        for gw in range(1, total_gws + 1):
            if gw not in gw_points_df.columns:
                gw_points_df[gw] = 0

        # Convert GW columns to numeric
        gw_cols = list(range(1, total_gws + 1))
        gw_points_df[gw_cols] = gw_points_df[gw_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Sort columns by GW number
        gw_points_df = gw_points_df[['Manager Name'] + gw_cols]

        # Replace 0 points with blank for display
        display_points_df = gw_points_df.replace(0, "")

        # Rank per GW (only rank non-blank points)
        rank_df = gw_points_df.copy()
        for gw in gw_cols:
            rank_df[gw] = (
                rank_df[gw]
                .replace(0, pd.NA)
                .rank(ascending=False, method='min')
                .astype("Int64")   # keep as integer with NA support
            )

        display_rank_df = rank_df  # already integers with NA

        # --- Highlighting functions ---
        def highlight_points(val, col):
            try:
                val = float(val)
            except:
                return ''
            max_val = gw_points_df[col].max()
            min_val = gw_points_df[col].min()
            if val == max_val:
                return 'background-color: lightgreen; font-weight: bold'
            elif val == min_val or val == 0:
                return 'background-color: salmon'
            else:
                return ''

        def highlight_rank(val, col):
            if pd.isna(val):
                return ''
            min_val = rank_df[col].min()
            max_val = rank_df[col].max()
            if val == min_val:
                return 'background-color: lightgreen; font-weight: bold'
            elif val == max_val:
                return 'background-color: salmon'
            else:
                return ''

        # Apply styles, leave 'Manager Name' column unstyled
        styled_points_df = display_points_df.style.apply(
            lambda row: [''] + [highlight_points(row[gw], gw) for gw in gw_cols], axis=1
        )

        styled_rank_df = display_rank_df.style.apply(
            lambda row: [''] + [highlight_rank(row[gw], gw) for gw in gw_cols], axis=1
        )

        # Display tables with styling
        st.subheader("GW Points per Manager")
        st.dataframe(styled_points_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("GW Rank per Manager")
        st.dataframe(styled_rank_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("No GW points data available yet.")

elif selected == "Fines":
    fines_df = fines_data()

    fines_col1, fines_col2 = st.columns([0.5, 0.5])
    with fines_col1:
        st.subheader("Fines Overview")
        st.dataframe(fines_df, use_container_width=True, hide_index=True)

    with fines_col2:
        st.subheader("Fines Distribution")
        fig = px.pie(
            fines_df,
            values="Total Fine",
            names="Manager Name",
            title="Fines Distribution",
            hover_data=["Team Name"]
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
