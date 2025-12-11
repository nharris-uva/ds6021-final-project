import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Visual style constants (aligned with data_vis_page.py)
COLORS = {
    'graph_bg': '#D9376E',
    'bg_main': '#FFFFFE',
    'header': '#FF8E3C',
    'text_primary': '#222222',
    'text_secondary': '#333333',
    'text_light': '#ffffff',
    'bg_transparent': 'rgba(0,0,0,0)',
    'border_light': '2px solid #eee',
    'card_background_color': '#EFF0F3',
}

BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

px.defaults.template = "plotly_dark"
DEFAULT_FIG_MARGIN = dict(l=10, r=10, t=10, b=20)

# --- Data Loading ---
# Default path used in data_vis_page.py
DATA_PATH = "New Data and Work/final_movie_table.csv"

# Fallback alternative paths to try if needed
FALLBACK_PATHS = [
    "final_movie_table.csv",
    "./final_movie_table.csv",
    "../New Data and Work/final_movie_table.csv",
]

def load_data():
    paths = [DATA_PATH] + FALLBACK_PATHS
    last_err = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load dataset from known paths. Last error: {last_err}")


raw_df = load_data()

# --- Baseline transforms (mirroring notebook logic) ---
df = raw_df.copy()

# Ensure expected columns exist
expected_cols = [
    'genres','vote_average','release_year','lead_actor','original_language','adult',
    'budget','revenue','runtime','vote_count','keyword_count','user_avg_rating','user_rating_count','movie_id'
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    # Do not crash; create placeholders if missing to allow partial functionality
    for c in missing:
        if c in ['adult']:
            df[c] = 0
        elif c in ['movie_id','vote_count','user_rating_count','keyword_count','budget','revenue','runtime']:
            df[c] = 0
        else:
            df[c] = np.nan

# Numeric cleaning
df = df.dropna(subset=['vote_average']).copy()

for col in ['user_avg_rating','user_rating_count']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if 'runtime' in df.columns:
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())

# Log transforms (log1p for zero safety)
for col in ['budget','revenue','vote_count','keyword_count','user_rating_count']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])

# Year grouping (bins)
year_bins = [0,1919,1949,1979,1999,2009,2019]
year_labels = ['Pre-1920','1920–1949','1950–1979','1980–1999','2000–2009','2010–2019']
if 'release_year' in df.columns:
    df['year_group'] = pd.cut(df['release_year'], bins=year_bins, labels=year_labels, right=True)

# Actor popularity grouping (global)
def classify_popularity_global(n):
    if n >= 10:
        return 'High Popularity'
    elif n >= 5:
        return 'Medium Popularity'
    else:
        return 'Low Popularity'

# Build actor summary for global pop grouping
if 'lead_actor' in df.columns:
    actor_global = df.groupby('lead_actor').agg(movie_count=('movie_id','count')).reset_index()
    actor_global['popularity_group'] = actor_global['movie_count'].apply(classify_popularity_global)
    df = df.merge(actor_global[['lead_actor','popularity_group']], on='lead_actor', how='left')

# Language mapping
def language_full(code):
    m = {
        'en':'English','fr':'French','it':'Italian','ja':'Japanese','de':'German',
        'ru':'Russian','hi':'Hindi','es':'Spanish'
    }
    return m.get(code, code)

if 'original_language' in df.columns:
    df['language_full'] = df['original_language'].map(language_full)

# Precomputed genre-exploded dataframe
if 'genres' in df.columns:
    df_genres = df.assign(genre=df['genres'].str.split('|')).explode('genre')
    df_genres = df_genres.dropna(subset=['genre'])
else:
    df_genres = df.copy()
    df_genres['genre'] = np.nan

# Helper: subset choice
ALL = 'All Movies'
BUDGET = 'Budget > 0'


def subset_df(choice):
    if choice == BUDGET and 'budget' in df.columns:
        return df[df['budget'] > 0].copy()
    return df.copy()


def subset_genres_df(choice):
    if choice == BUDGET and 'budget' in df_genres.columns:
        return df_genres[df_genres['budget'] > 0].copy()
    return df_genres.copy()

# --- Plot builders ---

def fig_bar_genre(choice):
    d = subset_genres_df(choice)
    if 'genre' not in d.columns:
        return go.Figure()
    genre_summary = (
        d.groupby('genre')['vote_average']
         .mean()
         .sort_values(ascending=False)
         .reset_index()
    )
    fig = px.bar(
        genre_summary, x='genre', y='vote_average',
        color_discrete_sequence=[COLORS['graph_bg']]
    )
    fig.update_layout(
        title=None, height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14)
    )
    fig.update_xaxes(tickangle=-60, automargin=True)
    return fig


def fig_box_year(choice):
    d = subset_df(choice)
    if 'year_group' not in d.columns:
        return go.Figure()
    fig = px.box(d, x='year_group', y='vote_average', color_discrete_sequence=['#2D9CDB'])
    fig.update_layout(
        title=None, height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14)
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_box_actor_pop(choice, mode):
    d = subset_df(choice)
    if 'lead_actor' not in d.columns:
        return go.Figure()

    if mode == 'global':
        # Use precomputed popularity_group
        pop_col = 'popularity_group'
        if pop_col not in d.columns:
            return go.Figure()
    else:
        # Budget-specific grouping: 1 -> Low, 2-5 -> Medium, >5 -> High
        actor_summary = (
            d.groupby('lead_actor')
             .agg(movie_count=('movie_id','count'))
             .reset_index()
        )
        def assign_group(n):
            if n == 1: return 'Low Popularity'
            elif 2 <= n <= 5: return 'Medium Popularity'
            else: return 'High Popularity'
        actor_summary['pop_group_bz'] = actor_summary['movie_count'].apply(assign_group)
        d = d.merge(actor_summary[['lead_actor','pop_group_bz']], on='lead_actor', how='left')
        pop_col = 'pop_group_bz'

    fig = px.box(d, x=pop_col, y='vote_average', color_discrete_sequence=['#27AE60'])
    fig.update_layout(
        title=None, height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14)
    )
    return fig


def fig_box_language(choice, min_count):
    d = subset_df(choice)
    if 'original_language' not in d.columns:
        return go.Figure()
    counts = d['original_language'].value_counts()
    keep = counts[counts >= int(min_count)].index
    d = d.assign(language_group=d['original_language'].where(d['original_language'].isin(keep), 'Other'))
    fig = px.box(d, x='language_group', y='vote_average', color_discrete_sequence=['#9B51E0'])
    fig.update_layout(
        title=None, height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14)
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_numeric_hists(choice, variables):
    d = subset_df(choice)
    vars_ok = [v for v in variables if v in d.columns]
    if not vars_ok:
        return go.Figure()
    rows = int(np.ceil(len(vars_ok)/3))
    fig = make_subplots = go.Figure()
    # Simpler approach: combine into one figure with separate traces; show legend labels
    for v in vars_ok:
        fig.add_trace(go.Histogram(x=d[v], name=v, opacity=0.6))
    fig.update_layout(
        barmode='overlay', title=None, height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14)
    )
    fig.update_traces(marker_color=COLORS['graph_bg'])
    return fig


def fig_corr(choice):
    d = subset_df(choice)
    num_cols = [
        'vote_average','vote_count','log_vote_count',
        'user_rating_count','log_user_rating_count',
        'budget','log_budget','revenue','log_revenue',
        'keyword_count','log_keyword_count','runtime'
    ]
    cols = [c for c in num_cols if c in d.columns]
    if not cols:
        return go.Figure()
    corr = d[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate='%{text}', textfont={"size": 10}
    ))
    fig.update_layout(
        title=None, height=600, margin=dict(l=100, r=50, t=50, b=100),
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=12)
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def fig_budget(choice):
    d = subset_df(choice)
    if 'budget' not in d.columns:
        return go.Figure()
    fig = px.histogram(d[d['budget']>0], x='budget', nbins=50, opacity=0.9, color_discrete_sequence=[COLORS['graph_bg']])
    fig.update_layout(title=None, height=360, margin=DEFAULT_FIG_MARGIN,
                      plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
                      font=dict(color=COLORS['text_primary'], size=14))
    return fig


def fig_log_budget(choice):
    d = subset_df(choice)
    if 'budget' not in d.columns:
        return go.Figure()
    log10_budget = np.log10(d.loc[d['budget']>0, 'budget'])
    fig = px.histogram(log10_budget, nbins=40, opacity=0.9, color_discrete_sequence=[COLORS['graph_bg']])
    fig.update_layout(title=None, height=360, margin=DEFAULT_FIG_MARGIN,
                      plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
                      font=dict(color=COLORS['text_primary'], size=14))
    return fig

# --- App ---
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS], include_assets_files=False)

# --- Static figures (no inputs) ---
DEFAULT_SUBSET = ALL
LANG_MIN_COUNT_DEFAULT = 500
ACTOR_MODE_DEFAULT = 'global'

genre_fig = fig_bar_genre(DEFAULT_SUBSET)
year_fig = fig_box_year(DEFAULT_SUBSET)
actor_fig = fig_box_actor_pop(DEFAULT_SUBSET, ACTOR_MODE_DEFAULT)
lang_fig = fig_box_language(DEFAULT_SUBSET, LANG_MIN_COUNT_DEFAULT)
budget_hist_fig = fig_budget(DEFAULT_SUBSET)
log_budget_fig = fig_log_budget(DEFAULT_SUBSET)
corr_fig = fig_corr(DEFAULT_SUBSET)

# Scatter matrix and regression default variables
SCATTER_DEFAULT_VARS = [
    'vote_average','log_vote_count','log_user_rating_count','log_revenue','log_budget','runtime'
]
REG_DEFAULT_VAR = 'log_vote_count'

_d = subset_df(DEFAULT_SUBSET)
_scatter_vars = [v for v in SCATTER_DEFAULT_VARS if v in _d.columns]
if _scatter_vars:
    scatter_fig = px.scatter_matrix(_d[_scatter_vars])
    scatter_fig.update_layout(
        height=700, margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=12)
    )
else:
    scatter_fig = go.Figure()

if REG_DEFAULT_VAR in _d.columns:
    x = _d[REG_DEFAULT_VAR].astype(float)
    y = _d['vote_average'].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    reg_fig = go.Figure()
    reg_fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='markers', name='data', marker=dict(color='#7f8c8d'), opacity=0.25))
    if len(x_fit) > 2:
        m, b = np.polyfit(x_fit, y_fit, 1)
        xs = np.linspace(x_fit.min(), x_fit.max(), 200)
        reg_fig.add_trace(go.Scatter(x=xs, y=m*xs + b, mode='lines', name='trend', line=dict(color='red')))
    reg_fig.update_layout(
        height=420, margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'], paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14),
        xaxis_title=REG_DEFAULT_VAR, yaxis_title='vote_average'
    )
else:
    reg_fig = go.Figure()

# --- Markdown content from notebook ---
INTRO_MD = (
    "This document includes exploratory data analysis and predictive modeling aimed at answering our primary question: **What factors lead to movie success, where success is defined quantitatively by the average vote rating?**\n\n"
    "In this dashboard, we look at genre impact, temporal trends, star power, and linguistic influence. We also consider distributions of financial variables and perform correlation analysis to understand relationships among numerical features."
)

GENRE_HDR_MD = "#### Genre Impact: Which movie genres typically receive the highest critical acclaim (average rating) from audiences, and which receive the lowest?"
GENRE_SUMMARY_MD = (
    "The visualization shows that Animation, History, and War films receive the highest average ratings, while genres like Western, Horror, and TV Movie score lower."
)

TEMPORAL_HDR_MD = "#### Temporal Trends: How does the era of release influence a movie's average rating?"
TEMPORAL_SUMMARY_MD = (
    "We grouped release years into broader historical categories to reduce noise from sparse early-year data and make long term patterns easier to compare. Across the boxplot, average ratings remain fairly stable over time, with only small fluctuations between groups. There is no strong upward or downward trend, suggesting that audience ratings have been consistent across decades despite changes in filmmaking style and industry growth."
)

STAR_HDR_MD = "#### Star Power: Does the popularity of the lead actor seem to correlate with average rating?"
STAR_SUMMARY_MD = (
    "We grouped actors into popularity tiers based on the number of movies they appeared in: High Popularity (10+ movies), Medium Popularity (5–9 movies), and Low Popularity (fewer than 5 movies). The boxplot shows that, while differences are modest, high-popularity actors tend to appear in slightly better-rated films on average, with a higher median vote score and fewer extreme low-rating outliers. Low-popularity actors show the widest spread, indicating more variability in film quality for performers with limited filmographies."
)

LANG_HDR_MD = "#### Linguistic Influence: How does the original language of a film affect its average rating?"
LANG_SUMMARY_MD = (
    "To analyze language effects, we restricted the dataset to languages with at least 500 films and created a cleaned classification that mapped language codes to their full names. The resulting visualization shows clear differences in average ratings across linguistic groups. Japanese and French films tend to receive noticeably higher vote averages, with medians above most other languages and tighter clustering among higher ratings. English films dominate the dataset numerically but show a broader distribution with many lower-rated entries, which lowers their overall median. German, Italian, Russian, and Hindi films fall in a middle band, each showing moderate central ratings but differing in the spread of outliers."
)

NUM_INTRO_MD = (
    "### Numerical Variables\n\nUp to this point, we've been looking at the impact of various categorical variables on average rating. Now we examine numerical variables and apply log transformations to reduce extreme skew and improve interpretability in later modeling (budget, revenue, vote_count, keyword_count, user_rating_count)."
)

PAIRPLOT_SUMMARY_MD = (
    "The scatter matrix highlights relationships between key numerical predictors after log transformation. Most scatterplots remain diffuse with no strong linear patterns, but popularity-related metrics show mild positive relationships with each other."
)

REG_SUMMARY_MD = (
    "These regression plots show how vote average changes with each selected predictor. Log vote count has the clearest positive relationship, suggesting that movies that attract more votes tend to receive higher average ratings. Log user rating count shows a similar, though weaker, upward trend. Log budget displays only a slight positive slope, implying that higher-budget films may perform marginally better on average but with substantial variability. Runtime shows a moderate positive trend."
)

NONZERO_BUDGET_MD = (
    "The nonzero budget distribution shows that movie spending is extremely right skewed, with most films made on relatively small budgets and a long tail of very expensive productions. After applying a log transformation, the distribution becomes much closer to a smooth, unimodal shape, centered around log10(budget) values between 6 and 7 (≈1 to 10 million dollars)."
)

ADULT_NOTE_MD = (
    "Note: The dataset shows only one adult-rated film among all movies with nonzero budgets. Because this category has essentially no representation, it cannot support meaningful statistical comparison or visualization. We remove the adult variable from further analysis."
)

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.Div("EDA Dashboard", className="card-header fw-semibold"),
                html.Div([
                    html.P(
                        "Interactive visualizations derived from the EDA notebook.",
                        className="mb-0",
                        style={'fontFamily':'Inter, system-ui','fontSize':'16px','color':COLORS['text_secondary']}
                    )
                ], className="card-body")
            ], className="card", style={"backgroundColor":"rgba(0,0,0,0)","boxShadow":"none"})
        ], className="col-12")
    ], className="row mb-4"),

    # Intro markdown
    html.Div([
        html.Div([
            html.Div([
                html.Div("Overview", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(INTRO_MD)], className="card-body")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    
    # Genre Impact
    html.Div([
        html.Div([
            html.Div([
                html.Div("Average Vote by Genre", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(GENRE_HDR_MD)], className="card-body pt-2 pb-0 px-3"),
                html.Div([dcc.Graph(figure=genre_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(GENRE_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Temporal Trends
    html.Div([
        html.Div([
            html.Div([
                html.Div("Vote Average by Release Year Group", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(TEMPORAL_HDR_MD)], className="card-body pt-2 pb-0 px-3"),
                html.Div([dcc.Graph(figure=year_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(TEMPORAL_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Star Power
    html.Div([
        html.Div([
            html.Div([
                html.Div("Average Vote by Actor Popularity", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(STAR_HDR_MD)], className="card-body pt-2 pb-0 px-3"),
                html.Div([dcc.Graph(figure=actor_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(STAR_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Language Influence
    html.Div([
        html.Div([
            html.Div([
                html.Div("Vote Average by Language Group", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(LANG_HDR_MD)], className="card-body pt-2 pb-0 px-3"),
                html.Div([dcc.Graph(figure=lang_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(LANG_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Budget distributions
    html.Div([
        html.Div([
            html.Div([
                html.Div("Budget Distribution (Nonzero)", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Graph(figure=budget_hist_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(NONZERO_BUDGET_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-lg-6 mb-3"),
        html.Div([
            html.Div([
                html.Div("Log10(Budget) Distribution (Nonzero)", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Graph(figure=log_budget_fig, config={'displayModeBar': False})], className="card-body p-2")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-lg-6 mb-3")
    ], className="row g-3 mb-3"),

    # Scatter Matrix (Pairplot analogue)
    html.Div([
        html.Div([
            html.Div([
                html.Div("Scatter Matrix of Key Variables", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Markdown(NUM_INTRO_MD)], className="card-body pt-2 pb-0 px-3"),
                html.Div([dcc.Graph(figure=scatter_fig, config={'displayModeBar': True})], className="card-body p-2"),
                html.Div([dcc.Markdown(PAIRPLOT_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Regression plot (vote_average vs predictor)
    html.Div([
        html.Div([
            html.Div([
                html.Div("Regression: Vote Average vs Predictor", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Graph(figure=reg_fig, config={'displayModeBar': False})], className="card-body p-2"),
                html.Div([dcc.Markdown(REG_SUMMARY_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-3"),

    # Correlation matrix
    html.Div([
        html.Div([
            html.Div([
                html.Div("Correlation Matrix", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                html.Div([dcc.Graph(figure=corr_fig, config={'displayModeBar': True})], className="card-body p-2"),
                html.Div([dcc.Markdown(ADULT_NOTE_MD)], className="card-body pt-0 pb-3 px-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3 mb-4"),
], id='app-container', className="container-fluid py-4 px-3", style={'background': COLORS['bg_main'], 'minHeight': '100vh', 'fontFamily': 'Inter, system-ui'})

if __name__ == '__main__':
    app.run(debug=True, port=8052, host='127.0.0.1')
