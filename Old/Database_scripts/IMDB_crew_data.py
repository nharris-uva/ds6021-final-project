import pandas as pd
import numpy as np
import os
from pprint import pprint


reimport_crew_data = True
if reimport_crew_data:
    crew_df = pd.read_csv("data/imdb_crew.tsv", delimiter="	").replace(r"\N", pd.NA)
    crew_df.dropna(subset=["tconst", "directors", "writers"],inplace=True)
    crew_df.to_csv("data/reduced_principals.csv")
else:
    crew_df = pd.read_csv("data/reduced_crew.csv")


print(crew_df.head())
exit()

reimport_principals_data = False
if reimport_principals_data:
    principals_df = pd.read_csv("data/imdb_principals.tsv", delimiter="	").replace(r"\N", pd.NA)
    principals_df.dropna(subset=["tconst", "category", "job"],inplace=True)
    principals_df.to_csv("data/reduced_principals.csv")
else:
    principals_df = pd.read_csv("data/reduced_principals.csv")


