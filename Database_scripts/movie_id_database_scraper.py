import requests
import json
import pandas as pd
import numpy
import dotenv
import os
from pprint import pprint
import time

dotenv.load_dotenv()

api_key = os.getenv('tmdb_read_key')

def get_page_of_movies(page):
    time.sleep(0.001) #bc I got a DDOS protection notice. Whoops!
    url = "https://api.themoviedb.org/3/movie/popular"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    params = {
        "page": page
    }

    response = requests.get(url, headers=headers, params=params)
    print(f"Now fetching page {page}. Status: {response.status_code}")
    if response.status_code == 200:
        current_page_df = pd.json_normalize(json.loads(response.text), record_path="results")
        current_page_df["page"] = [page]*len(current_page_df)
        return current_page_df 
    else:
        return None

def resume_scraping(path_to_cached_df, max_page):
    cached_movies = pd.read_csv(path_to_cached_df, index_col=0)
    for page in range(int(cached_movies["page"].values[-1]), max_page):
        temp_df = get_page_of_movies(page)
        if isinstance(temp_df, pd.DataFrame):
            cached_movies = pd.concat([cached_movies, temp_df])
        print(cached_movies.shape)
    cached_movies.to_csv(path_to_cached_df)

resume_scraping("data/movie_id_database.csv", 500)