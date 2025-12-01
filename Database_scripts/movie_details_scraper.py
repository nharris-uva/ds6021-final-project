import pandas as pd
import numpy as np
import requests
import dotenv
import os
import time
from tqdm import tqdm

#Setup for scraping
dotenv.load_dotenv()
api_key = os.getenv('tmdb_read_key')

#load Movie id database
id_df = pd.read_csv("data/movies_15_to_19_enriched.csv")
scrapped_id_list = id_df["id"].values

def get_movie_info(id):
    time.sleep(0.001)
    movie_id = id
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    params = {"language": "en-US"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        row_df = pd.json_normalize(response.json())
        return row_df
    else:
        return None

def add_details_to_df(id_list):
    detail_list = [get_movie_info(x) for x in tqdm(id_list, desc="Fetching movie details")]
    return pd.concat(detail_list)

new_data = add_details_to_df(scrapped_id_list)
new_data.to_csv("data/redux_movie_details_database.csv")