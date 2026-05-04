"""
NBA Player Clustering Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


FEATURES = ['3PAr', 'FTr', '3P%', 'eFG%', 'FT%',
            'AST_per36', 'TRB_per36', 'ORB_per36',
            'STL_per36', 'BLK_per36', 'PTS_per36']

CLUSTER_NAMES = {
    0: 'Two-Way Forwards',
    1: '3-and-D Wings',
    2: 'Rim-Running Centers',
    3: 'Primary Scorers',
    4: 'Defensive Connectors',
    5: 'Skilled Bigs'
}


def fetch_raw_data(season_year=2026):
    """Pull per-game stats from Basketball-Reference for a given season."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_per_game.html"
    tables = pd.read_html(url)
    return tables[0]


def deduplicate_traded_players(df):
    """For traded players, keep only the season-total row (2TM/3TM/4TM)."""
    cleaned_rows = []
    for player, group in df.groupby('Player'):
        if len(group) == 1:
            cleaned_rows.append(group)
        else:
            multi_team = group[group['Team'].isin(['2TM', '3TM', '4TM'])]
            if len(multi_team) > 0:
                cleaned_rows.append(multi_team)
            else:
                cleaned_rows.append(group.nlargest(1, 'G'))
    return pd.concat(cleaned_rows, ignore_index=False)


def fix_non_shooter_percentages(df):
    """When a player has 0 attempts, percentage gets reported as 1.0 — fix to 0."""
    df = df.copy()
    df['3PA'] = pd.to_numeric(df['3PA'], errors='coerce')
    df['3P%'] = pd.to_numeric(df['3P%'], errors='coerce')
    df.loc[df['3PA'] == 0, '3P%'] = 0
    
    df['FTA'] = pd.to_numeric(df['FTA'], errors='coerce')
    df['FT%'] = pd.to_numeric(df['FT%'], errors='coerce')
    df.loc[df['FTA'] == 0, 'FT%'] = 0
    return df


def engineer_features(df):
    """Filter to qualified players and create style features."""
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
    
    df_filtered = df[(df['G'] >= 20) & (df['MP'] >= 15)].copy()
    df_filtered = df_filtered[df_filtered['Player'] != 'League Average']
    
    stat_cols = ['FGA', '3PA', '3P%', '2PA', 'FT%', 'FTA', 'ORB', 'DRB',
                 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%', 'MP']
    for col in stat_cols:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    df_filtered['3PAr'] = df_filtered['3PA'] / df_filtered['FGA']
    df_filtered['FTr'] = df_filtered['FTA'] / df_filtered['FGA']
    df_filtered['AST_per36'] = df_filtered['AST'] * 36 / df_filtered['MP']
    df_filtered['TRB_per36'] = (df_filtered['ORB'] + df_filtered['DRB']) * 36 / df_filtered['MP']
    df_filtered['STL_per36'] = df_filtered['STL'] * 36 / df_filtered['MP']
    df_filtered['BLK_per36'] = df_filtered['BLK'] * 36 / df_filtered['MP']
    df_filtered['ORB_per36'] = df_filtered['ORB'] * 36 / df_filtered['MP']
    df_filtered['PTS_per36'] = df_filtered['PTS'] * 36 / df_filtered['MP']
    
    return df_filtered.dropna(subset=FEATURES).copy()


def cluster_players(df_clean, n_clusters=6, random_state=42):
    """Run K-means clustering and assign archetype names."""
    X = df_clean[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_clean = df_clean.copy()
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    df_clean['archetype'] = df_clean['cluster'].map(CLUSTER_NAMES)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['pca_x'] = X_pca[:, 0]
    df_clean['pca_y'] = X_pca[:, 1]
    
    return df_clean, X_scaled, scaler, kmeans, pca


def find_similar_players(df_clean, X_scaled, player_name, n=5):
    """Find the N most stylistically similar players to a given player."""
    matches = df_clean[df_clean['Player'].str.contains(player_name, case=False, na=False)]
    
    if len(matches) == 0:
        return None, None
    
    player = matches.iloc[0]
    player_idx = matches.index[0]
    player_vector = X_scaled[df_clean.index.get_loc(player_idx)].reshape(1, -1)
    distances = euclidean_distances(player_vector, X_scaled)[0]
    
    df_temp = df_clean.copy()
    df_temp['distance'] = distances
    similar = df_temp[df_temp['Player'] != player['Player']].nsmallest(n, 'distance')
    
    return player, similar


def run_full_analysis(season_year=2026):
    """End-to-end pipeline: fetch, clean, cluster. Returns everything the app needs."""
    df_raw = fetch_raw_data(season_year)
    df_dedup = deduplicate_traded_players(df_raw)
    df_fixed = fix_non_shooter_percentages(df_dedup)
    df_features = engineer_features(df_fixed)
    df_clustered, X_scaled, scaler, kmeans, pca = cluster_players(df_features)
    
    return {
        'df': df_clustered,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'kmeans': kmeans,
        'pca': pca,
        'features': FEATURES,
        'cluster_names': CLUSTER_NAMES
    }