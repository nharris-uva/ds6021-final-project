import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
movie_data = pd.read_csv('data/redux_movie_details_database.csv').head(2000)
data_df = movie_data.drop(columns=[
    "Unnamed: 0", "backdrop_path", "poster_path", "overview", "video", "homepage", "imdb_id", "tagline",
    "belongs_to_collection", "belongs_to_collection.id", "belongs_to_collection.name",
    "belongs_to_collection.poster_path", "belongs_to_collection.backdrop_path"
])
metadata_df = movie_data[["id", "backdrop_path", "poster_path", "overview", "video", "homepage", "imdb_id", "tagline"]]

numeric_cols = data_df.select_dtypes(include=['number']).columns
categorical_cols = data_df.select_dtypes(include=['object', 'category']).columns

# Transforming Data
transformed_data_df = data_df.copy()
transformed_data_df["budget"] = transformed_data_df["budget"].apply(np.log)
transformed_data_df["budget"] = transformed_data_df["budget"].replace(0, np.nan)
transformed_data_df["popularity"] = transformed_data_df["popularity"].apply(np.log)
transformed_data_df["revenue"] = transformed_data_df["revenue"].apply(np.log)
transformed_data_df["revenue"] = transformed_data_df["revenue"].replace(0, np.nan)
transformed_data_df["runtime"] = transformed_data_df["runtime"].replace(0, np.nan)
transformed_data_df["vote_average"] = transformed_data_df["vote_average"].replace(0, np.nan)
transformed_data_df["vote_count"] = transformed_data_df["vote_count"].apply(np.log)
transformed_data_df["vote_count"] = transformed_data_df["vote_count"].replace(0, np.nan)
transformed_data_df["release_date"] = pd.to_datetime(transformed_data_df["release_date"], errors='coerce')

# Modern stylesheet (Bootstrap 5)
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

app = Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS])
px.defaults.template = "plotly_white"
DEFAULT_FIG_MARGIN = dict(l=10, r=10, t=50, b=10)

numeric_cols = [col for col in data_df.select_dtypes(include=['number']).columns if col not in ['id', 'movie_id', 'page']]
categorical_cols = [col for col in data_df.select_dtypes(include=['object', 'category']).columns if data_df[col].nunique() <= 50]

def create_numeric_graph(df, col, title_prefix=""):
    fig = px.histogram(df, x=col, marginal="box", opacity=0.9)
    fig.update_layout(
        title_text=f'{title_prefix}{col}',
        height=300,
        margin=DEFAULT_FIG_MARGIN,
        title_font=dict(size=16, family="Inter, system-ui"),
    )
    return fig

def create_categorical_graph(df, col, title_prefix=""):
    value_counts = df[col].value_counts().head(20)
    fig = px.bar(x=value_counts.index, y=value_counts.values, text_auto=True, opacity=0.95)
    fig.update_layout(
        title_text=f'{title_prefix}{col}',
        xaxis_title=col,
        yaxis_title='Count',
        height=300,
        margin=DEFAULT_FIG_MARGIN,
        title_font=dict(size=16, family="Inter, system-ui"),
    )
    return fig

def build_dashboard_rows():
    rows = []
    all_cols = list(numeric_cols) + list(categorical_cols)
    for col in all_cols:
        left_card = html.Div([
            html.Div("Original", className="card-header fw-semibold"),
            html.Div(
                dcc.Graph(
                    figure=create_numeric_graph(data_df, col, "Original: ") if col in numeric_cols else create_categorical_graph(data_df, col, "Original: "),
                    config={'displayModeBar': False}
                ),
                className="card-body"
            )
        ], className="card shadow-sm border-0 h-100")
        middle_card = html.Div([
            html.Div(f"Transformation: {col}", className="card-header fw-semibold"),
            html.Div(
                dcc.Markdown(
                    children=f"Notes for {col} will appear here.",
                    className="card__content",
                ),
                className="card-body card__content",
                style={"height": "250px", "overflowY": "auto", "whiteSpace": "pre-wrap"}
            )
        ], className="card card--glass shadow-sm border-0 h-100")
        right_card = html.Div([
            html.Div("Transformed", className="card-header fw-semibold"),
            html.Div(
                dcc.Graph(
                    figure=create_numeric_graph(transformed_data_df, col, "Transformed: ") if col in numeric_cols else create_categorical_graph(transformed_data_df, col, "Transformed: "),
                    config={'displayModeBar': False}
                ),
                className="card-body"
            )
        ], className="card shadow-sm border-0 h-100")
        row = html.Div([
            html.Div(left_card, className="col-12 col-lg-4 mb-3"),
            html.Div(middle_card, className="col-12 col-lg-4 mb-3"),
            html.Div(right_card, className="col-12 col-lg-4 mb-3"),
        ], className="row g-3 align-items-stretch")
        rows.append(row)
    return rows

app.layout = html.Div([
    html.Link(rel="stylesheet", href=GOOGLE_FONTS),
    html.Div([
        html.Div([
            html.H1("Data Transformation Dashboard", className="h3 fw-bold mb-2", style={'fontFamily': 'Inter, system-ui'}),
            html.P("Compare original vs transformed features and document your steps.", className="text-muted mb-0")
        ], className="col")
    ], className="row mb-4"),
    html.Div([
        html.Div([
            html.Div("Original Data", className="card-body text-center fw-semibold"),
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div("Transformation Notes", className="card-body text-center fw-semibold"),
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div("Transformed Data", className="card-body text-center fw-semibold"),
        ], className="col-12 col-lg-4"),
    ], className="row g-3 mb-2"),
    html.Hr(className="my-2"),
    html.Div(build_dashboard_rows(), className="d-grid gap-3"),
], className="container-fluid py-3", style={'backgroundColor': '#f7f7f9'})

if __name__ == '__main__':
    app.run(debug=True, port=8050)
