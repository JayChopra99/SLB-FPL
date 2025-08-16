import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests
import plotly.express as px
import numpy as np

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
@st.cache_data(ttl=60)    
def gw_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url).json()
    
    # Gameweek data
    events_list = r.get('events', [])
    gw_df = pd.DataFrame(events_list)[['id', 'name', 'deadline_time', 'finished', 'average_entry_score']]
    gw_df.columns = ['GW ID', 'GW Name', 'Deadline', 'Finished', 'Average Points']
    
    # Phase data
    phases_list = r.get('phases', [])
    phases_df = pd.DataFrame(phases_list)[['id', 'name', 'start_event', 'stop_event']]
    phases_df.columns = ['Phase ID', 'Phase Name', 'Start GW', 'End GW']
    
    # Map each GW to its phase
    phase_lookup = {}
    for _, row in phases_df.iterrows():
        for gw in range(row['Start GW'], row['End GW'] + 1):
            phase_lookup[gw] = row['Phase Name']
    gw_df['Phase Name'] = gw_df['GW ID'].map(phase_lookup)
    
    # Convert deadline to datetime & extract month
    gw_df['Deadline'] = pd.to_datetime(gw_df['Deadline'])
    gw_df['Month'] = gw_df['Deadline'].dt.to_period('M')  # e.g., 2025-08
    
    return gw_df

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

    # --- Line chart for rank progression ---
    total_gws = 38
    gw_points_df = get_all_gw_points(total_gws=total_gws)

    if not gw_points_df.empty:
        # Ensure all 38 GWs exist
        for gw in range(1, total_gws + 1):
            if gw not in gw_points_df.columns:
                gw_points_df[gw] = 0

        gw_cols = list(range(1, total_gws + 1))
        gw_points_df[gw_cols] = gw_points_df[gw_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Compute rank per GW
        rank_df = gw_points_df.copy()
        for gw in gw_cols:
            rank_df[gw] = (
                rank_df[gw]
                .replace(0, pd.NA)
                .rank(ascending=False, method='min')
                .astype("Int64")
            )

        # Add GW 0 as starting point with rank 0
        rank_df.insert(1, 0, 0)  # insert GW0 after Manager Name

        # Reshape to long format
        rank_long = rank_df.melt(
            id_vars=["Manager Name"],
            value_vars=[0] + gw_cols,  # include GW0
            var_name="Gameweek",
            value_name="Rank"
        )

        # Ensure Gameweek is integer
        rank_long["Gameweek"] = rank_long["Gameweek"].astype(int)

        # Plot line chart
        fig = px.line(
            rank_long,
            x="Gameweek",
            y="Rank",
            color="Manager Name",
            markers=True,
            title="Rank Progression Over Gameweeks"
        )

        # Reverse y-axis so rank 1 is on top
        fig.update_yaxes(autorange=False, range=[0, rank_long["Rank"].max()])

        # Start x-axis from 0
        fig.update_xaxes(tickmode="linear", dtick=1, range=[0, rank_long["Gameweek"].max()])

        st.plotly_chart(fig, use_container_width=True)

elif selected == "GW Data":
    total_gws = 38
    gw_points_df = get_all_gw_points(total_gws=total_gws)

    if not gw_points_df.empty:
        gw_cols = list(range(1, total_gws + 1))

        # Ensure all GW columns exist
        for gw in gw_cols:
            if gw not in gw_points_df.columns:
                gw_points_df[gw] = pd.NA

        # Convert GW columns to numeric
        gw_points_df[gw_cols] = gw_points_df[gw_cols].apply(pd.to_numeric, errors='coerce')

        # Sort columns
        gw_points_df = gw_points_df[['Manager Name'] + gw_cols]

        # --- Rank calculation ---
        rank_df = gw_points_df.copy()
        for gw in gw_cols:
            rank_df[gw] = (
                rank_df[gw]
                .rank(ascending=False, method='min')
                .astype('Int64')
            )

        # --- Precompute min/max per GW for faster styling ---
        points_max = gw_points_df[gw_cols].max()
        points_min = gw_points_df[gw_cols].min()
        rank_max = rank_df[gw_cols].max()
        rank_min = rank_df[gw_cols].min()

        # --- Highlighting functions ---
        def highlight_points(val, col):
            if pd.isna(val):
                return ''
            if val == points_max[col]:
                return 'background-color: lightgreen; font-weight: bold'
            elif val == points_min[col]:
                return 'background-color: salmon'
            else:
                return ''

        def highlight_rank(val, col):
            if pd.isna(val):
                return ''
            if val == rank_min[col]:
                return 'background-color: lightgreen; font-weight: bold'
            elif val == rank_max[col]:
                return 'background-color: salmon'
            else:
                return ''

        # --- Create Tabs ---
        weekly_tab, monthly_tab = st.tabs(["Weekly", "Monthly"])

        # --- Weekly Tab ---
        with weekly_tab:
            st.subheader("GW Points per Manager")
            styled_points_df = gw_points_df.style.apply(
                lambda row: [''] + [highlight_points(row[gw], gw) for gw in gw_cols], axis=1
            )
            st.dataframe(styled_points_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.subheader("GW Rank per Manager")
            styled_rank_df = rank_df.style.apply(
                lambda row: [''] + [highlight_rank(row[gw], gw) for gw in gw_cols], axis=1
            )
            st.dataframe(styled_rank_df, use_container_width=True, hide_index=True)

        # --- Monthly Tab ---
        with monthly_tab:
            gw_df = gw_data()  # get GW metadata including Month

            # Convert GW points DF from wide to long
            points_long = gw_points_df.melt(
                id_vars=['Manager Name'],
                value_vars=gw_cols,
                var_name='GW',
                value_name='GW Points'
            )
            points_long['GW'] = points_long['GW'].astype(int)

            # Merge with GW metadata to get Month
            points_long = points_long.merge(
                gw_df[['GW ID','Month']],
                left_on='GW',
                right_on='GW ID',
                how='left'
            )

            # Convert Period to month abbreviation (e.g., Aug, Sep)
            points_long['Month'] = points_long['Month'].dt.strftime('%b')

            # Group by Manager and Month, sum points and convert to integer
            monthly_points = points_long.groupby(['Manager Name','Month'])['GW Points'].sum().reset_index()
            monthly_points['GW Points'] = monthly_points['GW Points'].astype('Int64')

            # Pivot to have months as columns, keep missing months as NaN
            monthly_points_pivot = monthly_points.pivot(
                index='Manager Name', 
                columns='Month', 
                values='GW Points'
            )

            # Sort months in calendar order, missing months remain NaN
            calendar_order = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
            monthly_points_pivot = monthly_points_pivot.reindex(columns=calendar_order)
            monthly_points_pivot = monthly_points_pivot.replace(0, np.nan)

            # --- Precompute min/max per Month, ignoring NaNs ---
            month_max = monthly_points_pivot.max(skipna=True)
            month_min = monthly_points_pivot.min(skipna=True)

            # Highlighting function for monthly points
            def highlight_monthly(val, col_max, col_min):
                if pd.isna(val):
                    return ''
                elif val == col_max:
                    return 'background-color: lightgreen; font-weight: bold'
                elif val == col_min:
                    return 'background-color: salmon'
                else:
                    return ''

            # Apply styling
            styled_monthly_df = monthly_points_pivot.style.apply(
                lambda row: [highlight_monthly(row[col], month_max[col], month_min[col]) for col in monthly_points_pivot.columns],
                axis=1
            )

            st.subheader("Monthly Points per Manager")
            st.dataframe(styled_monthly_df, use_container_width=True)
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
