import pandas as pd
import plotly.express as px
from pprint import pprint
from ast import literal_eval

needs_joining = False
if needs_joining:
    id_df = pd.read_csv("data/movie_id_database.csv", index_col=0)
    details_df = pd.read_csv("data/movie_details_database.csv", index_col=0)
    details_df.drop([x for x in details_df.columns if "belongs_to_collection" in x], inplace=True,axis=1) #Drop the added "collection columns the details df contains"

    duplicate_keys = ['backdrop_path', 'vote_average', 'adult', 'id', 'release_date', 'vote_count', 'title', 'video', 'popularity', 'original_title', 'original_language', 'overview', 'poster_path']
    completed_df = (id_df.set_index("id")).merge(details_df.set_index("id"), on=duplicate_keys, how="left")

    print(completed_df.head())
    print(completed_df.shape)
    print(completed_df.columns)
    completed_df.to_csv("data/completed_movie_database.csv")
else:
    completed_df = pd.read_csv("data/completed_movie_database.csv")

graph = False
if graph:
    g1 = px.scatter(completed_df, x="popularity", y="vote_average",log_x=True, size="vote_count", hover_data="title")
    g1.show()
    g2 = px.scatter(completed_df, x="vote_count", y="vote_average", hover_data="title")
    g2.show()
    g3 = px.histogram(completed_df, x="vote_average")
    g3.show()

# completed_df["genre_ids"] = completed_df["genre_ids"].apply(literal_eval)
# temp = list(set(completed_df["genre_ids"].sum()))
# temp.sort()
# pprint(temp)
# # temp = list(original_set)
# # temp.sort()
# # pprint(temp)

import requests
import dotenv
import os
import time

#Setup for scraping
dotenv.load_dotenv()
api_key = os.getenv('tmdb_read_key')
url = f"https://api.themoviedb.org/3/genre/movie/list"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}",
}

params = {"language": "en-US"}

response = requests.get(url, headers=headers, params=params)

print(response.text)