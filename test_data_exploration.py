import pandas as pd
import plotly.express as px

df = pd.read_csv("data/movie_id_database.csv")

test = px.scatter(df, x="popularity", y="vote_average",log_x=True, size="vote_count", hover_data="title")
test.show()