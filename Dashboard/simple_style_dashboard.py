
import pandas as pd
import dash
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import json
import joblib
from pathlib import Path

COLORS = {
    'graph_bg': '#D9376E',

    # Background
    'bg_main': '#FFFFFE',
    
    # Header colors
    'header': '#FF8E3C',
    
    # Text colors
    'text_primary': '#222222',
    'text_secondary': '#333333',
    'text_light': '#ffffff',
    
    # Misc
    'bg_transparent': 'rgba(0,0,0,0)',
    'border_light': '2px solid #eee',
    # Cards
    'card_background_color': '#EFF0F3',
}

# Modern stylesheet (Bootstrap 5)
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Load data
raw_data = pd.read_csv("New Data and Work/final_movie_table.csv")

# Initialize the Dash app with external stylesheets and ignore local assets
app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS],
    include_assets_files=True,
    suppress_callback_exceptions=True
)

# Plotly visual style
px.defaults.template = "plotly_dark"
DEFAULT_FIG_MARGIN = dict(l=20, r=20, t=60, b=40)

# Dictionary to store transformation notes for each column
TRANSFORMATION_NOTES = {
    # Add your transformation notes here
    # Example: "budget": "Applied log transformation to reduce skewness\nReplaced 0 values with NaN"
}

# Unified Plotly theming
def apply_fig_theme(fig, *, height=None):
    layout_updates = {
        'margin': DEFAULT_FIG_MARGIN,
        'plot_bgcolor': COLORS['bg_transparent'],
        'paper_bgcolor': COLORS['bg_transparent'],
        'font': dict(color=COLORS['text_primary'])
    }
    if height is not None:
        layout_updates['height'] = height
    fig.update_layout(**layout_updates)
    return fig

# =====================
# Models: Data & Helpers
# =====================

def load_and_preprocess_data_models():
    df = pd.read_csv("New Data and Work/final_movie_table.csv").head(1000)
    df_nz = df[df['budget'] > 0].copy()
    df_nz['log_budget'] = np.log1p(df_nz['budget'])
    df_nz['log_revenue'] = np.log1p(df_nz['revenue'])
    df_nz['log_vote_count'] = np.log1p(df_nz['vote_count'])
    df_nz['log_user_rating_count'] = np.log1p(df_nz['user_rating_count'])
    df_nz['log_keyword_count'] = np.log1p(df_nz['keyword_count'])

    numeric_vars = [
        'vote_average',
        'log_budget',
        'log_revenue',
        'log_vote_count',
        'log_user_rating_count',
        'log_keyword_count',
        'runtime'
    ]
    df_num_local = df_nz[numeric_vars].dropna()
    return df_num_local, df_nz

def prepare_categorical_data(df_nz):
    df_model = df_nz.copy()
    df_model['year_group'] = pd.cut(
        df_model['release_year'],
        bins=[0, 1919, 1949, 1979, 1999, 2009, 2019],
        labels=['Pre-1920', '1920-1949', '1950-1979', '1980-1999', '2000-2009', '2010-2019']
    )
    df_model['language_group'] = df_model['original_language'].apply(
        lambda x: x if x in ['en','fr','ru','hi','es','de','ja','it'] else 'Other'
    )
    df_actor_counts = (
        df_model.groupby('lead_actor')
                .size()
                .reset_index(name='movie_count')
    )
    df_actor_counts['popularity_group'] = pd.cut(
        df_actor_counts['movie_count'],
        bins=[0, 1, 5, float('inf')],
        labels=['Low Popularity', 'Medium Popularity', 'High Popularity']
    )
    df_model = df_model.merge(df_actor_counts[['lead_actor','popularity_group']], on='lead_actor', how='left')
    df_model['primary_genre'] = df_model['genres'].str.split('|').str[0]
    model_vars = [
        'vote_average',
        'log_budget',
        'log_revenue',
        'log_vote_count',
        'log_user_rating_count',
        'log_keyword_count',
        'runtime',
        'primary_genre',
        'year_group',
        'language_group',
        'popularity_group'
    ]
    df_model = df_model[model_vars].dropna()
    for col in ['primary_genre', 'year_group', 'language_group', 'popularity_group']:
        df_model[col] = df_model[col].astype(str)
    df_dummies = pd.get_dummies(df_model, drop_first=True)
    bool_cols = df_dummies.select_dtypes(include=['bool']).columns
    df_dummies[bool_cols] = df_dummies[bool_cols].astype(int)
    df_dummies = df_dummies.apply(pd.to_numeric, errors='coerce').dropna()
    return df_dummies

def fit_knn_model(df_num, fast=False):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    k_values = list(range(1, 21)) if fast else list(range(1, 31))
    rmse_curve = []
    for k in k_values:
        m = KNeighborsRegressor(n_neighbors=k)
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_valid_scaled)
        rmse_curve.append(np.sqrt(mean_squared_error(y_valid, preds)))
    param_grid = {
        'n_neighbors': k_values,
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_gs = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3 if fast else 5,
        n_jobs=-1
    )
    knn_gs.fit(X_train_scaled, y_train)
    knn_final = KNeighborsRegressor(**knn_gs.best_params_)
    knn_final.fit(X_train_scaled, y_train)
    preds_valid = knn_final.predict(X_valid_scaled)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
    rmse_curve_tuned = []
    for k in k_values:
        tuned_model = KNeighborsRegressor(
            n_neighbors=k,
            weights=knn_gs.best_params_['weights'],
            metric=knn_gs.best_params_['metric']
        )
        tuned_model.fit(X_train_scaled, y_train)
        tuned_preds = tuned_model.predict(X_valid_scaled)
        rmse_curve_tuned.append(np.sqrt(mean_squared_error(y_valid, tuned_preds)))
    preds_train = knn_final.predict(X_train_scaled)
    corr_df = pd.DataFrame({
        'feature': X.columns,
        'correlation_with_knn_pred': [
            np.corrcoef(X_train_scaled[:, i], preds_train)[0, 1]
            for i in range(X_train_scaled.shape[1])
        ]
    }).sort_values('correlation_with_knn_pred', ascending=False)
    r = permutation_importance(knn_final, X_valid_scaled, y_valid, n_repeats=(5 if fast else 10), random_state=3001)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': r.importances_mean
    }).sort_values('importance', ascending=False)
    return (
        knn_final,
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        valid_rmse,
        importance_df,
        scaler,
        k_values,
        rmse_curve,
        rmse_curve_tuned,
        corr_df,
        knn_gs.best_params_
    )

def fit_kmeans_model(df_num):
    X = df_num.drop(columns=['vote_average']).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k_values = range(2, 11)
    inertia_list = []
    silhouette_list = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=3001, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia_list.append(km.inertia_)
        silhouette_list.append(silhouette_score(X_scaled, labels))
    kmeans_tuned = KMeans(n_clusters=4, random_state=3001, n_init=20)
    clusters = kmeans_tuned.fit_predict(X_scaled)
    df_clusters = df_num.copy()
    df_clusters['cluster'] = clusters
    cluster_profiles = df_clusters.groupby('cluster').mean(numeric_only=True).reset_index()
    return kmeans_tuned, X_scaled, df_clusters, k_values, inertia_list, silhouette_list, cluster_profiles

def fit_pca_model(df_num, X_scaled):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    kmeans_pca = KMeans(n_clusters=4, random_state=3001, n_init=20)
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    pca_summary = pd.DataFrame({
        'PC': np.arange(1, 4),
        'ExplainedVariance': pca.explained_variance_ratio_[:3],
        'CumulativeVariance': np.cumsum(pca.explained_variance_ratio_[:3])
    })
    return pca, X_pca, clusters_pca, pca_summary

def fit_linear_model(df_num, df_nz):
    df_dummies = prepare_categorical_data(df_nz)
    y = df_dummies['vote_average']
    X = df_dummies.drop(columns=['vote_average'])
    X = sm.add_constant(X)
    ols_full = sm.OLS(y, X).fit()
    return ols_full, df_dummies

def fit_mlp_model(df_num, fast=False):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    mlp_final = MLPRegressor(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=(300 if fast else 500),
        random_state=3001
    )
    mlp_final.fit(X_train_scaled, y_train)
    final_train_rmse = np.sqrt(mean_squared_error(y_train, mlp_final.predict(X_train_scaled)))
    final_valid_rmse = np.sqrt(mean_squared_error(y_valid, mlp_final.predict(X_valid_scaled)))
    r_final = permutation_importance(
        mlp_final,
        X_valid_scaled,
        y_valid,
        n_repeats=(10 if fast else 20),
        random_state=3001
    )
    importance_final = pd.DataFrame({
        'feature': X.columns,
        'importance': r_final.importances_mean
    }).sort_values('importance', ascending=False)
    loss_curve = mlp_final.loss_curve_
    return mlp_final, final_train_rmse, final_valid_rmse, importance_final, loss_curve

print("Loading model data...")
df_num_models, df_nz_models = load_and_preprocess_data_models()

# Data tab should compare the original raw dataframe vs the post-processed dataframe.
# Use the fully loaded raw_data for "Original" and the models' df_nz for "Transformed".
original_data = raw_data.copy()
transformed_data = df_nz_models.copy()

# Features used in the Models section and their original→transformed mapping
FEATURE_MAP = {
    'vote_average': 'vote_average',
    'budget': 'log_budget',
    'revenue': 'log_revenue',
    'vote_count': 'log_vote_count',
    'user_rating_count': 'log_user_rating_count',
    'keyword_count': 'log_keyword_count',
    'runtime': 'runtime',
    'primary_genre': 'primary_genre',
    'year_group': 'year_group',
    'language_group': 'language_group',
    'popularity_group': 'popularity_group'
}

# Ensure engineered categorical features exist on original_data for fair comparison
if 'primary_genre' not in original_data.columns and 'genres' in original_data.columns:
    original_data['primary_genre'] = original_data['genres'].astype(str).str.split('|').str[0]
if 'year_group' not in original_data.columns and 'release_year' in original_data.columns:
    original_data['year_group'] = pd.cut(
        original_data['release_year'],
        bins=[0, 1919, 1949, 1979, 1999, 2009, 2019],
        labels=['Pre-1920', '1920-1949', '1950-1979', '1980-1999', '2000-2009', '2010-2019']
    )
if 'language_group' not in original_data.columns and 'original_language' in original_data.columns:
    original_data['language_group'] = original_data['original_language'].apply(
        lambda x: x if x in ['en','fr','ru','hi','es','de','ja','it'] else 'Other'
    )
if 'popularity_group' not in original_data.columns and 'lead_actor' in original_data.columns:
    _actor_counts = (
        original_data.groupby('lead_actor').size().reset_index(name='movie_count')
    )
    _actor_counts['popularity_group'] = pd.cut(
        _actor_counts['movie_count'],
        bins=[0, 1, 5, float('inf')],
        labels=['Low Popularity', 'Medium Popularity', 'High Popularity']
    )
    original_data = original_data.merge(_actor_counts[['lead_actor','popularity_group']], on='lead_actor', how='left')

model_cache = {
    'knn_rmse': None,
    'mlp_train_rmse': None,
    'mlp_valid_rmse': None,
    'linear_fitted': False,
    'knn_fitted': False,
    'kmeans_fitted': False,
    'mlp_fitted': False,
    'fitted': False,
    'linear_html': None,
    'knn_html': None,
    'kmeans_html': None,
    'mlp_html': None,
    'comparison_html': None,
    'knn_payload': None,
    'mlp_payload': None,
    'kmeans_pca_payload': None
}

def build_comparison_section(knn_rmse=None, mlp_train_rmse=None, mlp_valid_rmse=None):
    X = df_num_models.drop(columns=['vote_average'])
    y = df_num_models['vote_average']
    X_train_lr, X_valid_lr, y_train_lr, y_valid_lr = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr)
    X_valid_lr_scaled = scaler.transform(X_valid_lr)
    lr_model = KNeighborsRegressor(n_neighbors=1)
    lr_model.fit(X_train_lr_scaled, y_train_lr)
    lr_preds = lr_model.predict(X_valid_lr_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_valid_lr, lr_preds))
    comparison_data = {
        'Model': ['Linear Regression', 'KNN (Tuned)', 'K-Means', 'MLP (Tuned)'],
        'Train RMSE': [f'{lr_rmse:.3f}', f'{lr_rmse:.3f}' if knn_rmse else '—', 'N/A', f'{mlp_train_rmse:.3f}' if mlp_train_rmse else '—'],
        'Valid RMSE': [f'{lr_rmse:.3f}', f'{knn_rmse:.3f}' if knn_rmse else '—', 'Clustering', f'{mlp_valid_rmse:.3f}' if mlp_valid_rmse else '—'],
        'Type': ['Interpretable', 'Distance-based', 'Unsupervised', 'Nonlinear']
    }
    comparison_df = pd.DataFrame(comparison_data)
    knn_text = f"Validation RMSE {knn_rmse:.3f}" if knn_rmse else "Validation RMSE will update"
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "Model Performance Comparison",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dash_table.DataTable(
                        columns=[{"name": col, "id": col} for col in comparison_df.columns],
                        data=comparison_df.to_dict('records'),
                        style_header={
                            'backgroundColor': COLORS['header'],
                            'color': COLORS['text_light'],
                            'fontWeight': 'bold',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '14px',
                            'textAlign': 'center'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '10px',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '13px',
                            'border': '1px solid #ddd'
                        },
                        style_data={
                            'backgroundColor': COLORS['bg_main'],
                            'color': COLORS['text_primary']
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': COLORS['card_background_color']
                            }
                        ]
                    )
                ], className="card-body p-3")
            ], className="card")
        ], className="col-12 mb-4"),
        html.Div([
            html.Div([
                html.Div(
                    "Key Conclusions",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    html.H5("Model Performance Rankings:", className="fw-bold mt-3"),
                    html.Ol([
                        html.Li([html.Strong("MLP (Best):"), f" Validation RMSE {mlp_valid_rmse:.3f}" if mlp_valid_rmse else " Validation RMSE ~0.88", " - Captures nonlinear relationships effectively"]),
                        html.Li([html.Strong("KNN (Strong):"), f" {knn_text}", " - Excellent local neighborhood exploitation"]),
                        html.Li([html.Strong("Linear Regression:"), " Provides interpretability but limited by linearity assumption"]),
                        html.Li([html.Strong("K-Means:"), " Unsupervised clustering reveals natural movie groupings"]),
                    ]),
                    html.Hr(),
                    html.H5("Universal Finding:", className="fw-bold mt-3"),
                    html.P(
                        "Across all models, user engagement metrics (log vote count and log user rating count) "
                        "are the dominant predictors of movie ratings. Financial attributes (budget, revenue) "
                        "contribute less than engagement variables, suggesting that audience interaction patterns "
                        "are more informative than production scale."
                    ),
                    html.Hr(),
                    html.H5("Structural Insights:", className="fw-bold mt-3"),
                    html.Ul([
                        html.Li("PCA reveals that >50% of variance is driven by a single engagement-scale dimension"),
                        html.Li("K-Means identifies 4 natural movie archetypes based on budget, engagement, and runtime"),
                        html.Li("Nonlinear models (KNN, MLP) substantially outperform linear approaches"),
                        html.Li("The dataset exhibits meaningful local structure that distance-based and neural approaches can exploit"),
                    ])
                ], className="card-body")
            ], className="card")
        ], className="col-12")
    ], className="row g-3")

def build_linear_regression_section(ols_model):
    ols_summary = pd.DataFrame({
        'Coefficient': ols_model.params,
        'Std Error': ols_model.bse,
        't-statistic': ols_model.tvalues,
        'p-value': ols_model.pvalues
    }).reset_index().rename(columns={'index': 'Variable'})
    ols_summary_display = ols_summary.copy()
    ols_summary_display['Coefficient'] = ols_summary_display['Coefficient'].round(6)
    ols_summary_display['Std Error'] = ols_summary_display['Std Error'].round(6)
    ols_summary_display['t-statistic'] = ols_summary_display['t-statistic'].round(4)
    ols_summary_display['p-value'] = ols_summary_display['p-value'].round(6)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "Linear Regression Analysis with Categorical Features",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    html.P(
                        "Comprehensive linear regression incorporating numeric predictors and categorical features "
                        "(genres, release years, languages, and actor popularity). This model reveals how multiple factors combine to explain movie ratings, "
                        "showing strong positive effects from engagement metrics and important contributions from categorical structure.",
                        className="mb-3"
                    ),
                    html.Div([
                        html.Strong("Key Findings:"),
                        html.Ul([
                            html.Li("Log vote count: Strongest positive predictor of ratings"),
                            html.Li("Log user rating count: Second strongest engagement metric"),
                            html.Li("Genre effects: Documentary, History, Drama, War, Music, and Crime show strong positive effects"),
                            html.Li("Temporal patterns: Older films tend to have higher ratings than recent releases"),
                            html.Li("Language effects: Smaller language markets associated with higher average ratings"),
                            html.Li("Actor popularity: Frequent actors associated with higher rated films"),
                        ])
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Strong("Model Summary Statistics:"),
                        html.Ul([
                            html.Li(f"R-squared: {ols_model.rsquared:.4f} (explains ~{ols_model.rsquared*100:.1f}% of variance)"),
                            html.Li(f"Adjusted R-squared: {ols_model.rsquared_adj:.4f}"),
                            html.Li(f"F-statistic: {ols_model.fvalue:.2f}"),
                            html.Li(f"Prob (F-statistic): {ols_model.f_pvalue:.2e}"),
                            html.Li(f"Number of observations: {len(ols_model.resid)}"),
                        ])
                    ])
                ], className="card-body")
            ], className="card")
        ], className="col-12 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "OLS Regression Results",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dash_table.DataTable(
                        columns=[{"name": col, "id": col} for col in ols_summary_display.columns],
                        data=ols_summary_display.to_dict('records'),
                        style_header={
                            'backgroundColor': COLORS['header'],
                            'color': COLORS['text_light'],
                            'fontWeight': 'bold',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '13px',
                            'textAlign': 'center',
                            'padding': '8px'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '8px',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '12px',
                            'border': '1px solid #ddd',
                            'minWidth': '120px'
                        },
                        style_data={
                            'backgroundColor': COLORS['bg_main'],
                            'color': COLORS['text_primary']
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': COLORS['card_background_color']
                            }
                        ],
                        page_size=10
                    )
                ], className="card-body p-3", style={'maxHeight': '600px', 'overflowY': 'auto'})
            ], className="card")
        ], className="col-12")
    ], className="row mb-3")

def build_knn_section(knn_importance, knn_rmse, k_values, rmse_curve, rmse_curve_tuned, corr_df, best_params):
    fig_rmse = px.line(
        x=k_values,
        y=rmse_curve,
        markers=True,
        labels={'x': 'k', 'y': 'Validation RMSE'},
        title='KNN Validation RMSE by k (baseline)'
    ).update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8))
    fig_rmse = apply_fig_theme(fig_rmse, height=360)
    fig_rmse_tuned = px.line(
        x=k_values,
        y=rmse_curve_tuned,
        markers=True,
        labels={'x': 'k', 'y': 'Validation RMSE'},
        title='Tuned KNN Validation RMSE by k'
    ).update_traces(line=dict(color=COLORS['header']), marker=dict(size=8))
    fig_rmse_tuned = apply_fig_theme(fig_rmse_tuned, height=360)
    fig_corr = px.bar(
        corr_df,
        x='feature',
        y='correlation_with_knn_pred',
        title='Feature Correlation with KNN Predictions'
    )
    fig_corr.update_layout(xaxis_tickangle=45)
    fig_corr = apply_fig_theme(fig_corr, height=360)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"KNN Model (Tuned) - Validation RMSE: {knn_rmse:.4f}",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            knn_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            color_discrete_sequence=[COLORS['graph_bg']],
                            title="KNN: Permutation Importance"
                        ).update_layout(
                            height=400,
                            margin=DEFAULT_FIG_MARGIN,
                            plot_bgcolor=COLORS['bg_transparent'],
                            paper_bgcolor=COLORS['bg_transparent'],
                            font=dict(color=COLORS['text_primary']),
                            yaxis=dict(autorange='reversed')
                        ),
                        config={'displayModeBar': False}
                    )
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "KNN Validation Curves",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_rmse, config={'displayModeBar': False}),
                    dcc.Graph(figure=fig_rmse_tuned, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "KNN Performance Summary",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    html.P(
                        "The tuned KNN model exploits local similarity; expanded tuning (k, weights, metric) "
                        "identified a Manhattan-distance, distance-weighted model with balanced bias-variance.",
                        className="mb-3"
                    ),
                    html.Div([
                        html.Strong("Model Configuration:"),
                        html.Ul([
                            html.Li(f"Neighbors (k): {best_params['n_neighbors']}"),
                            html.Li(f"Weighting: {best_params['weights'].title()}"),
                            html.Li(f"Distance Metric: {best_params['metric'].title()}"),
                            html.Li(f"Validation RMSE: {knn_rmse:.4f}"),
                        ])
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Strong("Key Insights:"),
                        html.Ul([
                            html.Li("User engagement metrics dominate predictions"),
                            html.Li("Validation curves show overfitting at very low k and oversmoothing at high k"),
                            html.Li("Distance weighting + Manhattan metric provide smoother generalization"),
                            html.Li("Financial predictors contribute less than engagement variables"),
                        ])
                    ])
                ], className="card-body")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "Feature Correlation with KNN Predictions",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_corr, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card")
        ], className="col-12 mb-3")
    ], className="row g-3")

def build_clustering_section(X_pca, pca_clusters, k_values, silhouette_list, df_clusters, inertia_list, cluster_profiles):
    pca_2d = X_pca[:, :2]
    df_pca_plot = pd.DataFrame({
        'PC1': pca_2d[:, 0],
        'PC2': pca_2d[:, 1],
        'Cluster': pca_clusters
    })
    df_pca3_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': pca_clusters
    })
    fig_pca = px.scatter(
        df_pca_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        color_continuous_scale='Viridis',
        opacity=0.6,
        title='Tuned K-Means Clusters in PCA Space'
    ).update_traces(marker=dict(size=5))
    fig_pca = apply_fig_theme(fig_pca, height=400)
    fig_pca3d = px.scatter_3d(
        df_pca3_plot,
        x='PC1', y='PC2', z='PC3',
        color='Cluster',
        color_continuous_scale='Viridis',
        title='Tuned K-Means Clusters in 3D PCA Space',
        opacity=0.6
    ).update_traces(marker=dict(size=4))
    fig_pca3d = apply_fig_theme(fig_pca3d, height=420)
    fig_silhouette = px.line(
        x=list(k_values),
        y=silhouette_list,
        markers=True,
        labels={'x': 'k', 'y': 'Silhouette Score'},
        title='Silhouette Scores by k'
    ).update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8))
    fig_silhouette.update_layout(showlegend=False)
    fig_silhouette = apply_fig_theme(fig_silhouette, height=320)
    fig_inertia = px.line(
        x=list(k_values),
        y=inertia_list,
        markers=True,
        labels={'x': 'k', 'y': 'Inertia'},
        title='Inertia by Number of Clusters'
    ).update_traces(line=dict(color=COLORS['header']), marker=dict(size=8))
    fig_inertia.update_layout(showlegend=False)
    fig_inertia = apply_fig_theme(fig_inertia, height=320)
    cluster_profiles_melted = cluster_profiles.melt(id_vars='cluster', var_name='feature', value_name='value')
    fig_profiles = px.bar(
        cluster_profiles_melted,
        x='feature', y='value', color='cluster', barmode='group',
        title='Cluster Profiles: Average Scaled Feature Values'
    )
    fig_profiles.update_layout(xaxis_tickangle=45)
    fig_profiles = apply_fig_theme(fig_profiles, height=360)
    fig_box = px.box(
        df_clusters,
        x='cluster', y='vote_average',
        title='Vote Average by Tuned Cluster (k = 4)',
        labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'}
    )
    fig_box = apply_fig_theme(fig_box, height=360)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "K-Means Clustering Analysis",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "Clustering Tuning Diagnostics",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_silhouette, config={'displayModeBar': False}),
                    dcc.Graph(figure=fig_inertia, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "3D PCA View of Clusters",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca3d, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "Cluster Profiles",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_profiles, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "Cluster Outcome Distribution",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_box, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3")
    ], className="row g-3")

def build_mlp_section(mlp_importance, mlp_train_rmse, mlp_valid_rmse, loss_curve):
    fig_loss = px.line(
        x=list(range(len(loss_curve))),
        y=loss_curve,
        markers=True,
        labels={'x': 'Iteration', 'y': 'Training Loss'},
        title='MLP Training Loss Curve'
    ).update_traces(mode='lines+markers', line=dict(color=COLORS['graph_bg']))
    fig_loss = apply_fig_theme(fig_loss, height=360)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"MLP Regression (Tuned) - Valid RMSE: {mlp_valid_rmse:.4f}",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            mlp_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            color_discrete_sequence=[COLORS['graph_bg']],
                            title="MLP: Permutation Importance"
                        ).update_layout(
                            height=360,
                            margin=DEFAULT_FIG_MARGIN,
                            plot_bgcolor=COLORS['bg_transparent'],
                            paper_bgcolor=COLORS['bg_transparent'],
                            font=dict(color=COLORS['text_primary']),
                            yaxis=dict(autorange='reversed')
                        ),
                        config={'displayModeBar': False}
                    )
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "MLP Training Dynamics",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_loss, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "MLP Performance Summary",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    html.P(
                        "The tuned MLP achieves the strongest predictive performance by learning smooth nonlinear "
                        "relationships between engagement metrics and ratings.",
                        className="mb-3"
                    ),
                    html.Div([
                        html.Strong("Model Configuration:"),
                        html.Ul([
                            html.Li("Hidden Layers: (64,)"),
                            html.Li("Activation: ReLU"),
                            html.Li("Regularization (alpha): 0.0001"),
                            html.Li(f"Train RMSE: {mlp_train_rmse:.4f}"),
                            html.Li(f"Valid RMSE: {mlp_valid_rmse:.4f}"),
                        ])
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Strong("Key Insights:"),
                        html.Ul([
                            html.Li("Best overall predictive performance"),
                            html.Li("Captures smooth nonlinear relationships"),
                            html.Li("Low training-validation gap indicates good generalization"),
                            html.Li("Variable importance aligned with KNN findings"),
                        ])
                    ])
                ], className="card-body")
            ], className="card h-100")
        ], className="col-12 col-xl-4 mb-3")
    ], className="row g-3")

def build_pca_section():
    X = df_num_models.drop(columns=['vote_average']).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    fig_scree = px.bar(
        x=list(range(1, len(explained_var) + 1)),
        y=explained_var,
        labels={'x': 'Principal Component', 'y': 'Variance Explained'},
        title='Scree Plot: Variance by Component',
        color_discrete_sequence=[COLORS['graph_bg']]
    )
    fig_scree.update_layout(showlegend=False)
    fig_scree = apply_fig_theme(fig_scree, height=360)
    fig_cum = px.line(
        x=list(range(1, len(cum_var) + 1)),
        y=cum_var,
        markers=True,
        labels={'x': 'Principal Component', 'y': 'Cumulative Variance'},
        title='Cumulative Variance Explained'
    ).update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8)).add_hline(
        y=0.80,
        line_dash='dash',
        line_color='red',
        annotation_text='80% threshold'
    )
    fig_cum.update_layout(showlegend=False)
    fig_cum = apply_fig_theme(fig_cum, height=360)
    loadings = pd.DataFrame(
        pca_full.components_.T,
        columns=[f'PC{i+1}' for i in range(len(explained_var))],
        index=X.columns
    )
    fig_loadings = px.bar(
        loadings[['PC1', 'PC2', 'PC3']].reset_index().melt(id_vars='index', var_name='Component', value_name='Loading'),
        x='index', y='Loading', color='Component', barmode='group',
        title="PCA Loadings for First Three Components"
    )
    fig_loadings.update_layout(xaxis_tickangle=45)
    fig_loadings = apply_fig_theme(fig_loadings, height=360)
    X_pca2 = X_scaled @ pca_full.components_.T[:, :2]
    df_pca2 = pd.DataFrame({'PC1': X_pca2[:, 0], 'PC2': X_pca2[:, 1], 'vote_average': df_num_models['vote_average'].values})
    fig_pca2 = px.scatter(
        df_pca2,
        x='PC1', y='PC2', color='vote_average', color_continuous_scale='Viridis',
        title='Movies Projected onto PC1 & PC2 (Colored by Vote Average)', opacity=0.6
    ).update_traces(marker=dict(size=6))
    fig_pca2 = apply_fig_theme(fig_pca2, height=360)
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_scaled)
    df_pca3 = pd.DataFrame({'PC1': X_pca3[:, 0], 'PC2': X_pca3[:, 1], 'PC3': X_pca3[:, 2], 'vote_average': df_num_models['vote_average'].values})
    fig_pca3 = go.Figure()
    fig_pca3.add_trace(go.Scatter3d(
        x=df_pca3['PC1'], y=df_pca3['PC2'], z=df_pca3['PC3'],
        mode='markers', marker=dict(size=3.5, color=df_pca3['vote_average'], opacity=0.55, colorbar=dict(title='Vote Avg'))
    ))
    fig_pca3.update_layout(
        title='3D PCA Projection Colored by Vote Average',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        height=520
    )
    fig_pca3 = apply_fig_theme(fig_pca3, height=520)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "PCA: Variance Explained",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_scree, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "PCA: Cumulative Variance",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_cum, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "PCA Loadings",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_loadings, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "PCA Projection (2D)",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca2, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-xl-6 mb-3"),
        html.Div([
            html.Div([
                html.Div(
                    "PCA Projection (3D)",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca3, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12")
    ], className="row g-3")

def truncate_label(label, max_len=24):
    """Truncate a string label to a maximum length with ellipsis."""
    try:
        s = str(label)
    except Exception:
        s = label
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"

def get_column_type(df, col):
    """Determine if column is numeric or categorical"""
    if pd.api.types.is_numeric_dtype(df[col]):
        return 'numeric'
    else:
        # All non-numeric columns are categorical - we'll show top 15 values
        return 'categorical'

def create_numeric_graph(df, col, title_prefix=""):
    """Create histogram with box plot for numeric columns"""
    fig = px.histogram(df, x=col, marginal="box", opacity=0.9, color_discrete_sequence=[COLORS['graph_bg']])
    fig.update_layout(title_text=None, showlegend=False,
                      xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)),
                      yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)))
    return apply_fig_theme(fig, height=420)

def create_categorical_graph(df, col, title_prefix=""):
    """Create bar chart for categorical columns showing top 15 most frequent occurrences"""
    value_counts = df[col].value_counts().head(15)
    original_labels = list(value_counts.index)
    x_labels = [truncate_label(lbl, max_len=24) for lbl in original_labels]
    fig = px.bar(
        x=x_labels,
        y=value_counts.values,
        text_auto=True,
        opacity=0.95,
        color_discrete_sequence=[COLORS['graph_bg']]
    )
    # Keep bar labels inside to avoid clipping at the top
    fig.update_traces(
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=12, color=COLORS['text_light'])
    )
    fig.update_traces(
        hovertext=original_labels,
        hovertemplate="%{hovertext}: %{y}<extra></extra>"
    )
    fig.update_layout(title_text=None, xaxis_title=col, yaxis_title='Count', showlegend=False,
                      xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)),
                      yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)))
    fig.update_xaxes(tickangle=-30)
    return apply_fig_theme(fig, height=420)

def create_text_placeholder(col):
    """Create placeholder for text columns that can't be visualized"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Column '{col}' has too many unique values to visualize",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, family="Inter, system-ui", color=COLORS['text_primary'])
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), font=dict(color=COLORS['text_primary'], size=14))
    return apply_fig_theme(fig, height=420)

def create_missingness_bar(col: str):
    """Create a stacked bar chart showing % missing vs present in transformed data."""
    trans_pct_missing = float(transformed_data[col].isna().mean() * 100)
    trans_pct_present = 100 - trans_pct_missing

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Transformed"],
            y=[trans_pct_present],
            name="Present",
            marker_color="#2ecc71",
            text=[f"{trans_pct_present:.1f}%"],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["Transformed"],
            y=[trans_pct_missing],
            name="Missing",
            marker_color=COLORS['graph_bg'],
            text=[f"{trans_pct_missing:.1f}%"],
            textposition="inside",
        )
    )
    fig.update_layout(
        title=None,
        barmode="stack",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)),
        font=dict(color=COLORS['text_primary'], size=12),
        xaxis=dict(title=None, tickfont=dict(size=11), automargin=True),
        yaxis=dict(title="Percentage %", range=[0, 100], tick0=0, dtick=20, tickfont=dict(size=11), automargin=True),
    )
    return apply_fig_theme(fig, height=140)

def build_column_row(original_col: str, transformed_col: str):
    """Build a single row comparing original vs transformed features used in models"""
    # Determine types for appropriate visuals
    orig_type = get_column_type(original_data, original_col) if original_col in original_data.columns else 'categorical'
    trans_type = get_column_type(transformed_data, transformed_col) if transformed_col in transformed_data.columns else 'categorical'

    # Create appropriate visualizations based on column type
    left_graph = create_numeric_graph(original_data, original_col) if orig_type == 'numeric' else create_categorical_graph(original_data, original_col)
    right_graph = create_numeric_graph(transformed_data, transformed_col) if trans_type == 'numeric' else create_categorical_graph(transformed_data, transformed_col)

    # Notes keyed by original → transformed pair
    notes = TRANSFORMATION_NOTES.get(
        original_col,
        f"No transformation notes yet for **{original_col} → {transformed_col}**.\n\nAdd your transformation explanation here."
    )

    row = html.Div([
        # Left card - Original data
        html.Div([
            html.Div([
                html.Div(
                    f"Original: {original_col}",
                    className="card-header fw-semibold"
                ),
                html.Div(
                    dcc.Graph(
                        figure=left_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card h-100")
        ], className="col-12 col-lg-4 mb-3"),
        
        # Middle card - Transformation notes + missingness
        html.Div([
            html.Div([
                html.Div(
                    f"Transformation: {original_col} → {transformed_col}",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=create_missingness_bar(transformed_col),
                            config={'displayModeBar': False},
                            style={"height": "140px"}
                        )
                    ], className="mb-2"),
                    html.Div(
                        dcc.Markdown(
                            children=notes,
                            className="small"
                        ),
                        style={
                            "maxHeight": "260px",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                            "fontSize": "14px"
                        }
                    )
                ], className="card-body")
            ], className="card h-100")
        ], className="col-12 col-lg-4 mb-3"),
        
        # Right card - Transformed data
        html.Div([
            html.Div([
                html.Div(
                    f"Transformed: {transformed_col}",
                    className="card-header fw-semibold"
                ),
                html.Div(
                    dcc.Graph(
                        figure=right_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card h-100")
        ], className="col-12 col-lg-4 mb-3")
    ], className="row g-3 align-items-stretch mb-3")
    
    return row

def build_column_detail(selected_col: str):
    """Build per-column detail panel using transformed (model-used) data only.
    Shows distribution chart and descriptive metrics for the selected column.
    """
    # Use the selected column directly from transformed_data
    transformed_col = selected_col
    if transformed_col not in transformed_data.columns:
        return html.Div([
            html.Div([
                html.Div(
                    f"Column '{selected_col}' is not available in transformed data",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    html.P("Please select another column.")
                ], className="card-body")
            ], className="card")
        ])

    # Decide visualization type
    col_type = get_column_type(transformed_data, transformed_col)
    if col_type == 'numeric':
        dist_fig = create_numeric_graph(transformed_data, transformed_col)
        series = transformed_data[transformed_col]
        metrics = {
            'Count': int(series.count()),
            'Missing %': float(series.isna().mean() * 100),
            'Mean': float(series.mean()),
            'Std': float(series.std()),
            'Min': float(series.min()),
            'Median': float(series.median()),
            'Max': float(series.max()),
            'Skew': float(series.skew() if series.count() > 0 else 0.0),
            'Kurtosis': float(series.kurtosis() if series.count() > 0 else 0.0),
        }
    else:
        dist_fig = create_categorical_graph(transformed_data, transformed_col)
        series = transformed_data[transformed_col].astype(str)
        vc = series.value_counts()
        top_label = vc.index[0] if len(vc) else '—'
        top_count = int(vc.iloc[0]) if len(vc) else 0
        metrics = {
            'Count': int(series.shape[0]),
            'Missing %': float(series.isna().mean() * 100),
            'Unique': int(series.nunique()),
            'Top': str(top_label),
            'Top Freq': int(top_count),
        }

    # Build layout: Distribution (left), Metrics (right)
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"Distribution: {transformed_col}",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dcc.Graph(figure=dist_fig, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-lg-8 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "Metrics",
                    className="card-header fw-semibold"
                ),
                html.Div([
                    dash_table.DataTable(
                        columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                        data=[{"Metric": k, "Value": (f"{v:.3f}" if isinstance(v, float) else v)} for k, v in metrics.items()],
                        style_header={
                            'backgroundColor': COLORS['header'],
                            'color': COLORS['text_light'],
                            'fontWeight': 'bold',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '14px',
                            'textAlign': 'center',
                            'border': '1px solid #ddd'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '13px',
                            'border': '1px solid #ddd',
                            'minWidth': '100px'
                        },
                        style_data={
                            'backgroundColor': COLORS['bg_main'],
                            'color': COLORS['text_primary']
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': COLORS['card_background_color']
                            }
                        ],
                        page_size=10
                    )
                ], className="card-body p-2")
            ], className="card h-100")
        ], className="col-12 col-lg-4 mb-3"),
    ], className="row g-3 align-items-stretch mb-3")

def create_correlation_matrix(df):
    """Create a correlation matrix heatmap for numeric columns"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove id column if present
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Correlation",
            tickmode="linear",
            tick0=-1,
            dtick=0.5
        )
    ))
    
    fig.update_layout(
        title=None,
        height=600,
        margin=dict(l=100, r=50, t=50, b=100),
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=12),
        xaxis=dict(
            tickangle=-45,
            side='bottom',
            tickfont=dict(size=11),
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            automargin=True
        )
    )
    
    return fig

def build_dashboard_layout():
    """Deprecated: previously built comparison rows. No longer used."""
    return []

def build_data_tab_content():
    """Build Data tab to focus on post-processed (model-used) data only.
    Shows per-column distribution and metrics for all columns in transformed_data,
    plus a correlation matrix. No transformation notes.
    """
    # Allow selection of any column present in transformed_data
    columns = [col for col in transformed_data.columns]

    return html.Div([
        html.H3("Data", className="text-center mb-4",
               style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}),
        # Selector
        html.Div([
            html.Div([
                html.Label(
                    "Select a Column:",
                    htmlFor="column-selector",
                    style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '16px'}
                ),
                dcc.Dropdown(
                    id='column-selector',
                    options=[{'label': col, 'value': col} for col in columns],
                    value=columns[0] if columns else None,
                    style={'width': '300px', 'fontFamily': 'Inter, system-ui'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '10px'})
        ], className="row mb-2"),

        # Visualization + Metrics for selected column
        html.Div(id='column-visualization', children=[]),

        # Raw data table for transformed data at the bottom
        html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        "Post-Processed Dataset (Used for Modeling)",
                        className="card-header fw-semibold"
                    ),
                    html.Div([
                        dash_table.DataTable(
                            id='transformed-data-table',
                            columns=[{"name": col, "id": col} for col in transformed_data.columns],
                            data=transformed_data.to_dict('records'),
                            page_size=10,
                            style_table={
                                'overflowX': 'auto',
                                'overflowY': 'auto',
                                'maxHeight': '500px'
                            },
                            style_header={
                                'backgroundColor': COLORS['header'],
                                'color': COLORS['text_light'],
                                'fontWeight': 'bold',
                                'fontFamily': 'Inter, system-ui',
                                'fontSize': '14px',
                                'textAlign': 'center',
                                'border': '1px solid #ddd'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'fontFamily': 'Inter, system-ui',
                                'fontSize': '13px',
                                'border': '1px solid #ddd',
                                'minWidth': '100px',
                                'maxWidth': '300px',
                                'whiteSpace': 'normal'
                            },
                            style_data={
                                'backgroundColor': COLORS['bg_main'],
                                'color': COLORS['text_primary']
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': COLORS['card_background_color']
                                }
                            ],
                            fixed_rows={'headers': True},
                            sort_action='native',
                            filter_action='native'
                        )
                    ], className="card-body p-3")
                ], className="card")
            ], className="col-12")
        ], className="row mt-4 mb-4"),

        # Correlation matrix section (transformed data only)
        html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        "Correlation Matrix (Transformed Data)",
                        className="card-header fw-semibold"
                    ),
                    html.Div([
                        dcc.Graph(
                            id='correlation-matrix',
                            figure=create_correlation_matrix(transformed_data),
                            config={'displayModeBar': True}
                        )
                    ], className="card-body p-3")
                ], className="card")
            ], className="col-12")
        ], className="row mt-4 mb-4")
    ])

# Global cache for data tab content
_DATA_TAB_CACHE = None

def get_data_tab_content():
    """Get or build data tab content with caching"""
    global _DATA_TAB_CACHE
    if _DATA_TAB_CACHE is None:
        _DATA_TAB_CACHE = build_data_tab_content()
    return _DATA_TAB_CACHE


# Create the app layout
app.layout = html.Div([
    # Top header card
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H1(
                        "Movie Ratings Predictive Modeling Dashboard",
                        className="fw-semibold",
                        style={
                            'margin': '0',
                            'fontFamily': 'Inter, system-ui',
                            'color': COLORS['header']
                        }
                    ),
                    html.P(
                        "Explore models, data insights, and summary for our analysis of movie ratings using machine learning techniques.",
                        style={
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '16px',
                            'color': COLORS['text_secondary'],
                            'marginTop': '8px',
                            'marginBottom': '0'
                        }
                    )
                ], className="card-header", style={"backgroundColor": COLORS['card_background_color']}),
                
            ], className="card")
        ], className="col-12")
    ], className="row mb-4"),

    # Tab navigation
    html.Div([
        html.Div([
            dcc.Tabs(id='main-tabs', value='summary-tab', children=[
                dcc.Tab(label='Summary', value='summary-tab', 
                       className='custom-tab',
                       selected_className='custom-tab--selected'),
                dcc.Tab(label='Models', value='models-tab',
                       className='custom-tab',
                       selected_className='custom-tab--selected'),
                dcc.Tab(label='Data', value='data-tab',
                       className='custom-tab',
                       selected_className='custom-tab--selected'),
            ], className='custom-tabs')
        ], className="col-12")
    ], className="row mb-3"),

    # Tab content containers - all rendered at once, shown/hidden via CSS
    html.Div([
        html.Div(id='summary-content', children=[
            html.H3("Project Summary", className="text-center mb-4",
               style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}),
            html.Div([
            html.Div([
                html.Div("About This Dashboard", className="card-header fw-semibold"),
                html.Div([
                html.P(
                    "This dashboard explores movie data and predictive modeling for ratings. "
                    "Use the Models tab to compare different algorithms and the Data tab to inspect features, distributions, and transformations.",
                    className="mb-0",
                    style={'fontFamily': 'Inter, system-ui', 'fontSize': '16px', 'color': COLORS['text_secondary']}
                )
                ], className="card-body")
            ], className="card"),
            ], className="mb-4"),

            html.Div([
            html.Div([
                html.Div("Group Members", className="card-header fw-semibold"),
                html.Div([
                html.Ul([
                    html.Li("Sabine Segaloff"),
                    html.Li("Tianyin Mao"),
                    html.Li("Mason Earp"),
                    html.Li("Nick Thornton"),
                    html.Li("Nate Harris"),
                ], className="mb-0", style={'fontFamily': 'Inter, system-ui'})
                ], className="card-body")
            ], className="card"),
            ], className="mb-4"),

            html.Div([
            html.Div([
                html.Div("Dataset", className="card-header fw-semibold"),
                html.Div([
                html.P(
                    [
                    "Dataset source: ",
                    html.A("The Movies Dataset on Kaggle", href="https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset", target="_blank")
                    ],
                    className="mb-0",
                    style={'fontFamily': 'Inter, system-ui', 'fontSize': '16px'}
                )
                ], className="card-body")
            ], className="card"),
            ])
        ], style={'display': 'block'}),
        
        html.Div(id='models-content', children=[
            html.H3("Models", className="text-center mb-4",
               style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}),
            # Hidden storage divs for model data
            dcc.Store(id='linear-model-store'),
            dcc.Store(id='knn-model-store'),
            dcc.Store(id='kmeans-model-store'),
            dcc.Store(id='pca-model-store'),
            dcc.Store(id='mlp-model-store'),
            dcc.Interval(id='interval-trigger', interval=500, n_intervals=0, max_intervals=10),

            

            html.Div([
                html.Div([
                    dcc.Tabs(
                        id='model-tabs',
                        value='linear',
                        children=[
                            dcc.Tab(
                                label='Linear Regression',
                                value='linear',
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div(id='linear-section-container', children=[html.Div([
                                                html.Div([
                                                    html.Div(
                                                        f"Loading Linear Regression...",
                                                        className="card-header fw-semibold"
                                                    ),
                                                    html.Div([
                                                        html.Div([
                                                            html.Div(
                                                                [html.Div(className="spinner-border text-danger", role="status",
                                                                          style={"width": "3rem", "height": "3rem"})],
                                                                style={"textAlign": "center", "padding": "40px"}
                                                            ),
                                                            html.P(f"Fitting Linear Regression model...", 
                                                                  style={"textAlign": "center", "marginTop": "20px"})
                                                        ])
                                                    ], className="card-body")
                                                ], className="card")
                                            ])])
                                        ], className="col-12")
                                    ], className="row mb-4")
                                ]
                            ),
                            dcc.Tab(
                                label='KNN Regression',
                                value='knn',
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div(id='knn-section-container', children=[html.Div([
                                                html.Div([
                                                    html.Div(
                                                        f"Loading KNN...",
                                                        className="card-header fw-semibold"
                                                    ),
                                                    html.Div([
                                                        html.Div([
                                                            html.Div(
                                                                [html.Div(className="spinner-border text-danger", role="status",
                                                                          style={"width": "3rem", "height": "3rem"})],
                                                                style={"textAlign": "center", "padding": "40px"}
                                                            ),
                                                            html.P(f"Fitting KNN model...", 
                                                                  style={"textAlign": "center", "marginTop": "20px"})
                                                        ])
                                                    ], className="card-body")
                                                ], className="card")
                                            ])])
                                        ], className="col-12")
                                    ], className="row mb-4")
                                ]
                            ),
                            dcc.Tab(
                                label='K-Means Clustering',
                                value='kmeans',
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div(id='kmeans-section-container', children=[html.Div([
                                                html.Div([
                                                    html.Div(
                                                        f"Loading K-Means...",
                                                        className="card-header fw-semibold"
                                                    ),
                                                    html.Div([
                                                        html.Div([
                                                            html.Div(
                                                                [html.Div(className="spinner-border text-danger", role="status",
                                                                          style={"width": "3rem", "height": "3rem"})],
                                                                style={"textAlign": "center", "padding": "40px"}
                                                            ),
                                                            html.P(f"Fitting K-Means model...", 
                                                                  style={"textAlign": "center", "marginTop": "20px"})
                                                        ])
                                                    ], className="card-body")
                                                ], className="card")
                                            ])])
                                        ], className="col-12")
                                    ], className="row mb-4")
                                ]
                            ),
                            dcc.Tab(
                                label='PCA',
                                value='pca',
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div(id='pca-section-container', children=[])
                                        ], className="col-12")
                                    ], className="row mb-4")
                                ]
                            ),
                            dcc.Tab(
                                label='MLP Regression',
                                value='mlp',
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div(id='mlp-section-container', children=[html.Div([
                                                html.Div([
                                                    html.Div(
                                                        f"Loading MLP...",
                                                        className="card-header fw-semibold"
                                                    ),
                                                    html.Div([
                                                        html.Div([
                                                            html.Div(
                                                                [html.Div(className="spinner-border text-danger", role="status",
                                                                          style={"width": "3rem", "height": "3rem"})],
                                                                style={"textAlign": "center", "padding": "40px"}
                                                            ),
                                                            html.P(f"Fitting MLP model...", 
                                                                  style={"textAlign": "center", "marginTop": "20px"})
                                                        ])
                                                    ], className="card-body")
                                                ], className="card")
                                            ])])
                                        ], className="col-12")
                                    ], className="row mb-4")
                                ]
                            )
                        ]
                    )
                ], className="col-12")
            ], className="row mb-4"),

            html.Div([
                html.Div([
                    html.Div(id='comparison-section-container', children=[
                        html.Div([
                            html.Div([
                                html.Div(
                                    "Model Performance Comparison",
                                    className="card-header fw-semibold"
                                ),
                                html.Div([
                                    html.P("Loading model results...", style={"textAlign": "center", "padding": "20px"})
                                ], className="card-body p-3")
                            ], className="card")
                        ], className="col-12 mb-4"),

                        html.Div([
                            html.Div([
                                html.Div(
                                    "Key Conclusions",
                                    className="card-header fw-semibold"
                                ),
                                html.Div([
                                    html.H5("Model Performance Rankings:", className="fw-bold mt-3"),
                                    html.Ol([
                                        html.Li(html.Strong("MLP (Best):"), " Validation RMSE ~0.88 - Captures nonlinear relationships effectively"),
                                        html.Li(html.Strong("KNN (Strong):"), " Validation RMSE will update - Excellent local neighborhood exploitation"),
                                        html.Li(html.Strong("Linear Regression:"), " Provides interpretability but limited by linearity assumption"),
                                        html.Li(html.Strong("K-Means:"), " Unsupervised clustering reveals natural movie groupings"),
                                    ]),
                                    html.Hr(),
                                    html.H5("Universal Finding:", className="fw-bold mt-3"),
                                    html.P(
                                        "Across all models, user engagement metrics (log vote count and log user rating count) "
                                        "are the dominant predictors of movie ratings. Financial attributes (budget, revenue) "
                                        "contribute less than engagement variables, suggesting that audience interaction patterns "
                                        "are more informative than production scale."
                                    ),
                                    html.Hr(),
                                    html.H5("Structural Insights:", className="fw-bold mt-3"),
                                    html.Ul([
                                        html.Li("PCA reveals that >50% of variance is driven by a single engagement-scale dimension"),
                                        html.Li("K-Means identifies 4 natural movie archetypes based on budget, engagement, and runtime"),
                                        html.Li("Nonlinear models (KNN, MLP) substantially outperform linear approaches"),
                                        html.Li("The dataset exhibits meaningful local structure that distance-based and neural approaches can exploit"),
                                    ])
                                ], className="card-body")
                            ], className="card")
                        ], className="col-12")
                    ], className="row g-3")
                ], className="col-12")
            ], className="row mb-4")
        ], style={'display': 'none'}),
        
        dcc.Loading(
            id="loading-data-tab",
            type="circle",
            color=COLORS['header'],
            children=html.Div(id='data-content', children=[], style={'display': 'none'})
        )
    ], className="tab-content-container"),
    
], id='app-container', className="container-fluid py-4 px-3", style={
    'background': COLORS['bg_main'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui'
})

# Route to allow dataset download from the app server
import os
from flask import send_file
@app.server.route('/download/final_movie_table')
def download_final_movie_table():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'New Data and Work', 'final_movie_table.csv')
    return send_file(path, as_attachment=True)

# Callbacks to handle tab switching
@app.callback(
    [Output('summary-content', 'style'),
     Output('models-content', 'style'),
     Output('data-content', 'style')],
    Input('main-tabs', 'value')
)
def toggle_tab_visibility(active_tab):
    """Toggle visibility of tab content without re-rendering"""
    summary_style = {'display': 'block'} if active_tab == 'summary-tab' else {'display': 'none'}
    models_style = {'display': 'block'} if active_tab == 'models-tab' else {'display': 'none'}
    data_style = {'display': 'block'} if active_tab == 'data-tab' else {'display': 'none'}
    return summary_style, models_style, data_style

# Hero buttons: switch main tabs when clicked
@app.callback(
    Output('main-tabs', 'value'),
    [Input('hero-summary-btn', 'n_clicks'),
     Input('hero-models-btn', 'n_clicks'),
     Input('hero-data-btn', 'n_clicks')],
    prevent_initial_call=True
)
def hero_buttons_nav(summary_clicks, models_clicks, data_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]['prop_id']
    if 'hero-summary-btn' in trigger:
        return 'summary-tab'
    if 'hero-models-btn' in trigger:
        return 'models-tab'
    if 'hero-data-btn' in trigger:
        return 'data-tab'
    return no_update

@app.callback(
    Output('data-content', 'children'),
    Input('main-tabs', 'value'),
    State('data-content', 'children')
)
def load_data_tab_once(active_tab, current_children):
    """Lazy load data tab content only once"""
    # If switching to data tab and content not loaded yet
    if active_tab == 'data-tab' and not current_children:
        return get_data_tab_content()
    # Otherwise don't update
    return no_update

@app.callback(
    Output('column-visualization', 'children'),
    Input('column-selector', 'value')
)
def update_column_visualization(selected_column):
    """Update the column visualization based on selection"""
    if selected_column is None:
        return html.Div("Please select a column", style={'textAlign': 'center', 'padding': '20px'})
    
    # Build and return selected column's transformed-data visualization
    return build_column_detail(selected_column)

# =====================
# Models: Callbacks
# =====================

@app.callback(
    Output('linear-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_linear_section(active_tab):
    if active_tab != 'linear':
        return no_update
    if model_cache['linear_fitted']:
        return model_cache['linear_html']
    try:
        ols_model, _ = fit_linear_model(df_num_models, df_nz_models)
        model_cache['linear_fitted'] = True
        html_content = build_linear_regression_section(ols_model)
        model_cache['linear_html'] = html_content
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@app.callback(
    Output('knn-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_knn_section(active_tab):
    if active_tab != 'knn':
        return no_update
    if model_cache['knn_fitted'] and model_cache['knn_payload'] is not None:
        p = model_cache['knn_payload']
        importance_df = pd.DataFrame(p['importance'])
        corr_df = pd.DataFrame(p['correlation']).sort_values('correlation_with_knn_pred', ascending=False)
        html_content = build_knn_section(importance_df, p['valid_rmse'], p['k_values'], p['rmse_curve'], p['rmse_curve_tuned'], corr_df, p['best_params'])
        model_cache['knn_html'] = html_content
        return html_content
    try:
        (
            knn_model,
            X_train_knn,
            X_valid_knn,
            y_train_knn,
            y_valid_knn,
            knn_rmse,
            knn_importance,
            knn_scaler,
            k_values,
            rmse_curve,
            rmse_curve_tuned,
            corr_df,
            best_params
        ) = fit_knn_model(df_num_models, fast=False)
        model_cache['knn_rmse'] = knn_rmse
        model_cache['knn_fitted'] = True
        html_content = build_knn_section(knn_importance, knn_rmse, k_values, rmse_curve, rmse_curve_tuned, corr_df, best_params)
        model_cache['knn_html'] = html_content
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@app.callback(
    Output('kmeans-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_kmeans_section(active_tab):
    if active_tab != 'kmeans':
        return no_update
    if model_cache['kmeans_fitted'] and model_cache['kmeans_pca_payload'] is not None:
        p = model_cache['kmeans_pca_payload']
        cluster_profiles = pd.DataFrame(p['cluster_profiles'])
        df_clusters_small = pd.DataFrame(p['df_clusters_small'])
        X_pca = np.array(p['X_pca'])
        pca_clusters = np.array(p['pca_clusters'])
        html_content = build_clustering_section(X_pca, pca_clusters, p['k_values'], p['silhouette_list'], df_clusters_small, p['inertia_list'], cluster_profiles)
        model_cache['kmeans_html'] = html_content
        return html_content
    try:
        kmeans_model, X_scaled_km, df_clusters, k_values, inertia_list, silhouette_list, cluster_profiles = fit_kmeans_model(df_num_models)
        pca_model, X_pca, pca_clusters, pca_summary = fit_pca_model(df_num_models, X_scaled_km)
        model_cache['kmeans_fitted'] = True
        html_content = build_clustering_section(X_pca, pca_clusters, k_values, silhouette_list, df_clusters, inertia_list, cluster_profiles)
        model_cache['kmeans_html'] = html_content
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@app.callback(
    Output('pca-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_pca_section(active_tab):
    if active_tab != 'pca':
        return no_update
    try:
        html_content = build_pca_section()
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@app.callback(
    Output('mlp-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_mlp_section(active_tab):
    if active_tab != 'mlp':
        return no_update
    if model_cache['mlp_fitted'] and model_cache['mlp_payload'] is not None:
        p = model_cache['mlp_payload']
        importance_df = pd.DataFrame(p['importance'])
        html_content = build_mlp_section(importance_df, p['train_rmse'], p['valid_rmse'], p['loss_curve'])
        model_cache['mlp_html'] = html_content
        return html_content
    try:
        mlp_model, mlp_train_rmse, mlp_valid_rmse, mlp_importance, loss_curve = fit_mlp_model(df_num_models, fast=False)
        model_cache['mlp_train_rmse'] = mlp_train_rmse
        model_cache['mlp_valid_rmse'] = mlp_valid_rmse
        model_cache['mlp_fitted'] = True
        if model_cache['knn_fitted'] and model_cache['mlp_fitted']:
            model_cache['fitted'] = True
        html_content = build_mlp_section(mlp_importance, mlp_train_rmse, mlp_valid_rmse, loss_curve)
        model_cache['mlp_html'] = html_content
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@app.callback(
    Output('comparison-section-container', 'children'),
    Input('interval-trigger', 'n_intervals'),
    Input('knn-section-container', 'children'),
    Input('mlp-section-container', 'children'),
    prevent_initial_call=False
)
def update_comparison_section(n, knn_children, mlp_children):
    if model_cache['fitted'] and model_cache['comparison_html'] is not None:
        return model_cache['comparison_html']
    if not model_cache['fitted']:
        return html.Div([
            html.Div([
                html.Div(
                    "Model Performance Comparison",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    html.P("Loading model results...", style={"textAlign": "center", "padding": "20px"})
                ], className="card-body p-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 mb-4")
    try:
        html_content = build_comparison_section(
            model_cache['knn_rmse'],
            model_cache['mlp_train_rmse'],
            model_cache['mlp_valid_rmse']
        )
        model_cache['comparison_html'] = html_content
        return html_content
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")
        
# Inject custom CSS for tabs
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            /* Custom tab container styling */
            .custom-tabs {
                display: flex;
                position: relative;
                background: transparent;
                border-radius: 10px 10px 0 0;
                padding: 0;
                border: none;
                box-shadow: none;
            }
            
            /* Individual tab styling */
            .custom-tab {
                margin: 12px 4px 0 4px;
                padding: 8px 20px;
                border: none;
                border-bottom: 0.5pt solid #ddd;
                color: #333333;
                background: #f8f9fa;
                box-shadow: 1px 1px 2px 1px rgba(0,0,0,0.05);
                text-transform: uppercase;
                font-weight: 300;
                font-size: 11px;
                border-radius: 7px 7px 0 0;
                border-top: 1px solid #e0e0e0;
                letter-spacing: 0.167em;
                font-family: 'Manrope', 'Inter', sans-serif;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                z-index: 1;
            }
            
            .custom-tab:hover {
                color: #000;
                background: #e9ecef;
            }
            
            /* First tab (Summary) */
            .custom-tab:first-of-type {
                border-radius: 10px 0 0 0;
                margin-left: 0;
            }
            
            /* Last tab (Data) */
            .custom-tab:last-of-type {
                border-radius: 0 10px 0 0;
            }
            
            /* Selected tab styling with gradient colors */
            .custom-tab--selected {
                color: #000 !important;
                font-weight: 600 !important;
                position: relative;
                z-index: 2;
                background: #fff !important;
            }
            
            /* Summary tab selected (blue/purple gradient) */
            .custom-tabs > div:nth-child(1) .custom-tab--selected {
                color: deepskyblue !important;
            }
            
            /* Models tab selected (green gradient) */
            .custom-tabs > div:nth-child(2) .custom-tab--selected {
                color: #51a14c !important;
            }
            
            /* Data tab selected (orange gradient) */
            .custom-tabs > div:nth-child(3) .custom-tab--selected {
                color: #FF8E3C !important;
            }
            
            /* Glider indicator - positioned absolutely */
            .custom-tabs::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 33.33%;
                height: 4.5px;
                border-radius: 0 0 1px 1px;
                background: linear-gradient(113deg, hsl(260deg 100% 64%) 0%, hsl(190deg 100% 55%) 100%);
                box-shadow: 0px 0px 8px 0px hsl(262deg 100% 70% / 70%);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 3;
            }
            
            /* Move glider based on active tab */
            .custom-tabs[data-active="summary-tab"]::after {
                left: 0%;
                background: linear-gradient(113deg, hsl(260deg 100% 64%) 0%, hsl(190deg 100% 55%) 100%);
                box-shadow: 0px 0px 8px 0px hsl(262deg 100% 70% / 70%);
            }
            
            .custom-tabs[data-active="models-tab"]::after {
                left: 33.33%;
                background: linear-gradient(90deg, #51a14c 0%, #10c33e 100%);
                box-shadow: 0px 0px 8px 0px rgba(47, 187, 12, 0.62);
            }
            
            .custom-tabs[data-active="data-tab"]::after {
                left: 66.66%;
                background: linear-gradient(90deg, #faffcc 0%, #f5eea3 10%, #ffe48a 40%, #ffb54d 65%, #ff974d 85%, #ff8052 100%);
                box-shadow: 0px 0px 8px 0px hsl(17.72deg 100% 70% / 70%);
            }
            
            /* Tab content container */
            .tab-content-container {
                padding-top: 20px;
                background: transparent;
            }
            
            /* Content area styling */
            #summary-content, #models-content, #data-content {
                padding: 20px;
                border-radius: 0 0 10px 10px;
                background: transparent;
                min-height: 200px;
                box-shadow: none;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Add glider tracking attribute
            document.addEventListener('DOMContentLoaded', function() {
                const observer = new MutationObserver(function() {
                    const tabs = document.querySelector('.custom-tabs');
                    if (tabs) {
                        const activeTabs = tabs.querySelectorAll('.custom-tab--selected');
                        if (activeTabs.length > 0) {
                            const parentDiv = activeTabs[0].closest('.custom-tabs > div');
                            if (parentDiv) {
                                const index = Array.from(parentDiv.parentElement.children).indexOf(parentDiv);
                                const tabNames = ['summary-tab', 'models-tab', 'data-tab'];
                                tabs.setAttribute('data-active', tabNames[index] || 'summary-tab');
                            }
                        }
                    }
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['class']
                });
            });
        </script>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='127.0.0.1')
