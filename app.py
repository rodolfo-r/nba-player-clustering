"""
NBA Player Archetype Explorer — Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import run_full_analysis, find_similar_players, FEATURES, CLUSTER_NAMES

# Page config
st.set_page_config(
    page_title="NBA Player Archetypes",
    layout="wide"
)

# Cache the analysis so it only runs once per day instead of every page load
@st.cache_data(ttl=86400)  # 86400 seconds = 24 hours
def load_data():
    return run_full_analysis()

# Header
st.title("NBA Player Archetypes 2025-26")
st.markdown(
    "Using **K-means clustering** to discover the real playing styles in today's NBA — "
    "beyond the outdated 5 traditional positions. Built with live data from Basketball-Reference."
)

# Load data with a spinner
with st.spinner("Loading and clustering player data... (this takes ~10 seconds on first load)"):
    result = load_data()

df = result['df']
X_scaled = result['X_scaled']

# Sidebar with stats
st.sidebar.header("Dataset")
st.sidebar.metric("Players analyzed", len(df))
st.sidebar.metric("Style features used", len(FEATURES))
st.sidebar.metric("Archetypes discovered", df['archetype'].nunique())

st.sidebar.markdown("---")
st.sidebar.subheader("Archetype Counts")
for archetype, count in df['archetype'].value_counts().items():
    st.sidebar.write(f"**{archetype}:** {count}")

# ============================================================
# THE SCATTER PLOT (top of page)
# ============================================================
st.header("The 6 NBA Archetypes")
st.markdown(
    "Each dot is a player. Players close together have similar playing styles."
)

# Build the scatter plot
fig, ax = plt.subplots(figsize=(14, 8))
palette = sns.color_palette("tab10", n_colors=6)

for i, archetype in enumerate(CLUSTER_NAMES.values()):
    cluster_data = df[df['archetype'] == archetype]
    ax.scatter(cluster_data['pca_x'], cluster_data['pca_y'],
               label=archetype, color=palette[i],
               s=80, alpha=0.7, edgecolor='white', linewidth=0.8)

# Label top scorers
top_players = df.sort_values('PTS', ascending=False).head(20)
for _, row in top_players.iterrows():
    ax.annotate(row['Player'], (row['pca_x'], row['pca_y']),
                fontsize=8, alpha=0.9,
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('PC1 — Inside vs. Outside Players', fontsize=11)
ax.set_ylabel('PC2 — Star Usage Level', fontsize=11)
ax.legend(title='Archetype', loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# TABS for the two main features
# ============================================================
tab1, tab2 = st.tabs(["Find a Player", "Compare Players"])

# ============================================================
# TAB 1 — Search for a single player
# ============================================================
with tab1:
    st.subheader("Search for any player")
    
    player_options = sorted(df['Player'].unique().tolist())
    selected_player = st.selectbox(
        "Pick a player to see their archetype and stylistic comps:",
        options=player_options,
        index=player_options.index('Luka Dončić') if 'Luka Dončić' in player_options else 0
    )
    
    if selected_player:
        player, similar = find_similar_players(df, X_scaled, selected_player, n=5)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### {player['Player']}")
            st.markdown(f"**Team:** {player['Team']}")
            st.markdown(f"**Position:** {player['Pos']}")
            st.markdown(f"**Archetype:** :blue[{player['archetype']}]")
            st.markdown("---")
            st.markdown("**Key Stats**")
            st.metric("Points", f"{player['PTS']:.1f}")
            st.metric("Assists", f"{player['AST']:.1f}")
            st.metric("Rebounds", f"{player['TRB']:.1f}")
            st.metric("3PT Rate", f"{player['3PAr']:.0%}")
            st.metric("3PT %", f"{player['3P%']:.1%}")
        
        with col2:
            st.markdown("### 5 Most Similar Players")
            st.markdown("These players have the most similar playing styles:")
            
            display_df = similar[['Player', 'Team', 'Pos', 'archetype', 'PTS', 'AST', 'TRB']].copy()
            display_df.columns = ['Player', 'Team', 'Pos', 'Archetype', 'PTS', 'AST', 'TRB']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.info(
                "The algorithm doesn't care about traditional positions. "
                "It looks at how a player actually plays: shot selection, rebounding rate, "
                "playmaking, and efficiency."
            )

# ============================================================
# TAB 2 — Compare up to 3 players
# ============================================================
with tab2:
    st.subheader("Compare up to 3 players side-by-side")
    
    selected_players = st.multiselect(
        "Pick 2-3 players to compare:",
        options=player_options,
        default=['Luka Dončić', 'Nikola Jokić', 'Giannis Antetokounmpo'] 
                if all(p in player_options for p in ['Luka Dončić', 'Nikola Jokić', 'Giannis Antetokounmpo'])
                else [],
        max_selections=3
    )
    
    if len(selected_players) >= 2:
        st.markdown("### Stat Comparison")
        comparison = df[df['Player'].isin(selected_players)][
            ['Player', 'Team', 'Pos', 'archetype', 'PTS', 'AST', 'TRB',
             '3PAr', 'FTr', '3P%', 'eFG%', 'FT%']
        ].copy()
        comparison.columns = ['Player', 'Team', 'Pos', 'Archetype', 'PTS', 'AST', 'TRB',
                              '3PT Rate', 'FT Rate', '3P%', 'eFG%', 'FT%']
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        archetypes = df[df['Player'].isin(selected_players)]['archetype'].unique()
        st.markdown("### Verdict")
        if len(archetypes) == 1:
            st.success(
                f"**All in the same archetype: {archetypes[0]}.** "
                f"Despite any position differences, these players play similarly."
            )
        else:
            archetype_list = ', '.join(archetypes)
            st.warning(
                f"**Different archetypes:** {archetype_list}. "
                f"These players have distinct playing styles."
            )
        
        st.markdown("### Style Profile Comparison")
        radar_features_short = ['3PAr', 'FTr', 'AST_per36', 'TRB_per36',
                                'STL_per36', 'BLK_per36', 'PTS_per36']
        
        league_max = df[radar_features_short].max()
        league_min = df[radar_features_short].min()
        
        angles = np.linspace(0, 2 * np.pi, len(radar_features_short), endpoint=False).tolist()
        angles += angles[:1]
        
        fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, player_name in enumerate(selected_players):
            player_row = df[df['Player'] == player_name].iloc[0]
            values = [(player_row[f] - league_min[f]) / (league_max[f] - league_min[f])
                      for f in radar_features_short]
            values += values[:1]
            
            ax2.plot(angles, values, color=colors[i], linewidth=2, label=player_name)
            ax2.fill(angles, values, color=colors[i], alpha=0.2)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_features_short, size=10)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.25, 0.5, 0.75])
        ax2.set_yticklabels(['', '', ''])
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True, alpha=0.4)
        ax2.set_title('Stat Profile (relative to league)', size=13, fontweight='bold', pad=20)
        st.pyplot(fig2)
    
    elif len(selected_players) == 1:
        st.info("Pick at least 2 players to compare.")
    else:
        st.info("Pick 2-3 players from the dropdown above to compare.")

# Footer
st.markdown("---")
st.markdown(
    "Built with [Streamlit](https://streamlit.io) | "
    "Data from [Basketball-Reference](https://www.basketball-reference.com) | "
    "[GitHub Repo](https://github.com/rodolfo-r/nba-player-clustering)"
)