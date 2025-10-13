import pandas as pd
import plotly.express as px
from pprint import pprint
from ast import literal_eval
import json
import numpy as np

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

generate_genre_table = False
if generate_genre_table:
    completed_df["genre_ids"] = completed_df["genre_ids"].apply(literal_eval)
    genre_df = completed_df[["id", "genre_ids"]]
    genre_df = genre_df.explode("genre_ids")
    with open("genres.json", "r") as f:
        genre_list = pd.json_normalize(json.load(f), record_path="genres")

    genre_df = pd.merge(left=genre_df, right=genre_list, how="left", left_on="genre_ids", right_on="id", suffixes=("","_gl"))
    percent_na = (genre_df["name"].isna().sum()/len(genre_df["name"]))
    print(f"Percent NA values: {percent_na}")
    genre_df = genre_df[["id", "genre_ids", "name"]]
    genre_df.to_csv("data/movie_genre_database.csv")
else:
    genre_df = pd.read_csv("data/movie_genre_database.csv", index_col=0)



graph = True
if graph:
    #Popularity (log x) vs vote average
    g1 = px.scatter(completed_df, x="popularity", y="vote_average",log_x=True, size="vote_count", hover_data="title")
    g1.show()
    #Vote count vs vote average
    g2 = px.scatter(completed_df, x="vote_count", y="vote_average", hover_data="title")
    g2.show()
    #Vote average histogram
    g3 = px.histogram(completed_df, x="vote_average")
    g3.show()
    #Name vs Count
    g4 = px.bar(genre_df, x="name")
    g4.show()
    #Avg Rating by Genre:
    genre_df=pd.merge(completed_df[["id", "vote_average"]], genre_df, how="right", on="id")
    rating_table = pd.DataFrame(genre_df.groupby("name", as_index=False)["vote_average"].mean(), columns=["name", "vote_average"])
    rating_table = rating_table.iloc[2:]
    g5 = px.bar(rating_table, x="name", y="vote_average")
    g5.show()
    #Violin Plots by genre:
    g6 = px.violin(genre_df, y="vote_average", color="name")
    g6.show()
    #Revenue vs budget:
    g7 = px.scatter(completed_df, x="revenue", y="budget")
    g7.show()
    #profit vs ratings:
    completed_df["profit"] = completed_df["revenue"] - completed_df["budget"]
    g8 = px.scatter(completed_df, x="profit", y="vote_average", hover_name="title", color="status")
    g8.show()
    