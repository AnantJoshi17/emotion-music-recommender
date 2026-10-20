"""Rebuild data/songs.csv from the TidyTuesday Spotify Songs dataset (collected Jan 2020).

Run once from the project root:  python data/build_songs.py
"""
import pandas as pd

URL = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
       "data/2020/2020-01-21/spotify_songs.csv")

df = pd.read_csv(URL).dropna(subset=["track_name", "track_artist"])

# The same song appears on several playlists/albums -> keep the most popular copy.
df = df.sort_values("track_popularity", ascending=False).drop_duplicates("track_id")
df["key"] = (df.track_name.str.lower().str.replace(r"\s*[\(\[-].*$", "", regex=True).str.strip()
             + "|" + df.track_artist.str.lower())
df = df.drop_duplicates("key")

# Drop remixes, live versions, edits, etc. and keep reasonably popular songs.
df = df[~df.track_name.str.contains(r"remix|mix\)|live|edit\)|version|acoustic\)|instrumental|sped up|slowed",
                                    case=False, regex=True)]
df = df[df.track_popularity >= 50]

# Cover every mood: split valence x energy into a 5x5 grid, keep the 20 most popular songs per cell.
bins = [0, .2, .4, .6, .8, 1.01]
df["v_bin"] = pd.cut(df.valence, bins, right=False)
df["e_bin"] = pd.cut(df.energy, bins, right=False)
songs = (df.groupby(["v_bin", "e_bin"], observed=True, group_keys=False)
           .apply(lambda cell: cell.nlargest(20, "track_popularity")))

songs = songs.rename(columns={"track_name": "song_name", "track_artist": "artist",
                              "track_popularity": "popularity"})
songs = songs[["song_name", "artist", "track_id", "popularity", "valence", "energy"]]
songs = songs.sort_values("popularity", ascending=False)
songs.to_csv("data/songs.csv", index=False)
print(f"{len(songs)} songs written to data/songs.csv")
