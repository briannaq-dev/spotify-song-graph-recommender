# Brianna Quinn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

'''
We prepare the given Spotify dataset for Neo4j by generating two CSV files: one for song nodes (songs.csv) and one for similarity relationships (edges.csv).
Our sampling strategy: 
 - Include ALL songs by the target artists: The Strokes and Regina Spektor
 - Add a random sample of 1000 other songs from the dataset
 - Use random_state = 42 so the sample is reproducible
'''

# Read in csv and drop duplicates
df = pd.read_csv('spotify.csv')
df = df.drop_duplicates(subset=["track_id"])

# Lowercase artists
df["artists"] = df["artists"].astype(str).str.lower()

# Create mask for target artists
artist_mask = df["artists"].str.contains(
    "the strokes|regina spektor",
    na=False
)

# Split into target artists and other songs
target_df = df[artist_mask].copy()
other_df = df[~artist_mask].copy()

# Sample size = 1000 non-target songs
n = 1000
if len(other_df) < n:
    sampled_other_df = other_df.copy()
else:
    sampled_other_df = other_df.sample(n, random_state=42)

'''
Similarity is defined using audio features: 
 - danceability
 -  energy
 - loudness
 - speechiness
 - acousticness
 - instrumentalness
 - liveness
 - valence
 - tempo

Rows with missing feature values are removed and all features are standardized using StandardScaler
'''

# Define features
feature_cols = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

# Combine target and sampled other songs and drop rows with missing feature values
graph_df = pd.concat([target_df, sampled_other_df], ignore_index=True)
graph_df = graph_df.dropna(subset=feature_cols).copy()

# Scale features and fit scaler
X = graph_df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
Recommendation Approach:
 - k-nearest neighbors, k=10
 - Each song is a point in the space
 - Euclidean distance is used to identify nearest neighbors
 - edges_df stores similarity edges to each song’s nearest neighbors
 - songs_df contains the song data that will become the node attributes 
 - dfs are exported to csv to be loaded into Neo4j

Graph Data Model:
    Nodes represent songs and are labeled :Song
    Node properties:
    - track_id (primary key)
    - artists
    - album_name
    - track_name
    - popularity
    - duration_ms
    - audio features (listed above)
    - track_genre

    Relationships represent similarity between songs and are labeled :SIMILAR_TO
    Relationship properties: Euclidean distance between vectors
 '''

k = 10
# Initialize NearestNeighbors and fit to scaled features
nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean") # + 1 because closes is the song itself
nn.fit(X_scaled)

# Find neighbors
distances, indices = nn.kneighbors(X_scaled)

# Create edges
edges = []
for i, neighbors in enumerate(indices):
    source_id = graph_df.iloc[i]["track_id"]

    for j in range(1, len(neighbors)):  # skip itself at index 0
        neighbor_idx = neighbors[j]
        target_id = graph_df.iloc[neighbor_idx]["track_id"]
        distance = distances[i][j]

        edges.append({
            "source": source_id,
            "target": target_id,
            "distance": distance
        })
edges_df = pd.DataFrame(edges)

# Define song columns and create songs dataframe
songs_cols = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre"
]
songs_df = graph_df[songs_cols].copy()

# Create csvs
songs_df.to_csv("songs.csv", index=False)
edges_df.to_csv("edges.csv", index=False)

print("CSV files created")