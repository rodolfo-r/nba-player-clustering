# NBA Player Archetype Clustering — 2025-26 Season

## Overview
This project uses unsupervised machine learning (K-means clustering) to discover the 
true playing styles in today's NBA, beyond the outdated 5 traditional positions.

After clustering 350 qualified players based on 11 style features (3-point rate, free 
throw rate, per-36 assists/rebounds/steals/blocks/points, and shooting efficiency), 
the model identified **6 distinct modern archetypes**.

## Key Findings

1. **Traditional positions are obsolete.** The "Primary Scorers" cluster contains 
   point guards (Luka, Brunson), shooting guards (Edwards), small forwards (Durant), 
   power forwards, and even a center (Jokić) — all playing the same role.

2. **There are only 22 traditional centers left.** The "Rim-Running Centers" cluster 
   (Gobert, Allen, Zubac, Ayton) is the most isolated group on the visualization, 
   confirming the death of the traditional non-shooting big.

3. **Jokić plays like a guard.** Despite being listed as a center, the algorithm 
   grouped him with Luka, SGA, and Edwards — not with other bigs. His usage and 
   playmaking profile is closer to a primary scorer than a traditional center.

4. **The 3-and-D wing is the most populous archetype.** With 91 players, this 
   role-player archetype outnumbers every other group.

## The 6 Archetypes

| Archetype | Count | Examples |
|-----------|-------|----------|
| 3-and-D Wings | 91 | Mikal Bridges, Derrick White, Cam Johnson |
| Primary Scorers | 86 | Luka, SGA, Edwards, Durant, Jokić |
| Two-Way Forwards | 71 | OG Anunoby, Jaden McDaniels, P.J. Washington |
| Defensive Connectors | 56 | Draymond Green, Marcus Smart, Herb Jones |
| Skilled Bigs | 24 | Giannis, Wembanyama, KAT, Anthony Davis |
| Rim-Running Centers | 22 | Gobert, Jarrett Allen, Zubac, Ayton |

## Methodology

- **Data source:** Basketball-Reference.com per-game stats
- **Cleaning:** Deduplicated traded players (kept season totals); fixed percentage values for non-shooters
- **Filtering:** Players with 20+ games and 15+ minutes per game
- **Features:** 11 style-focused metrics (rates and per-36 stats, not raw counts)
- **Preprocessing:** StandardScaler to normalize feature scales
- **Algorithm:** K-means with K=6 (selected via elbow method)
- **Visualization:** PCA reduction to 2D, capturing 54.8% of variance

## Tech Stack
- Python 3.9
- pandas, numpy
- scikit-learn (KMeans, PCA, StandardScaler)
- matplotlib, seaborn

## Files
- `clustering.ipynb` — Full analysis notebook
- `nba_archetypes.png` — 2D scatter plot of all clusters
- `nba_radars.png` — Statistical profile of each archetype

## Author
Rodolfo Ramirez
