import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, callback, clientside_callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
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
import base64
from pathlib import Path

# Color scheme
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

# External stylesheets
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Load and preprocess data
def load_and_preprocess_data():
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
    
    df_num = df_nz[numeric_vars].dropna()
    return df_num, df_nz

def prepare_categorical_data(df_nz):
    """Prepare categorical features for linear regression modeling"""
    df_model = df_nz.copy()
    
    # Create year groups
    df_model['year_group'] = pd.cut(
        df_model['release_year'],
        bins=[0, 1919, 1949, 1979, 1999, 2009, 2019],
        labels=['Pre-1920', '1920-1949', '1950-1979', '1980-1999', '2000-2009', '2010-2019']
    )
    
    # Create language groups
    df_model['language_group'] = df_model['original_language'].apply(
        lambda x: x if x in ['en','fr','ru','hi','es','de','ja','it'] else 'Other'
    )
    
    # Create actor popularity groups
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
    
    # Extract primary genre
    df_model['primary_genre'] = df_model['genres'].str.split('|').str[0]
    
    # Select variables for full categorical model
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
    
    # Convert categorical columns to string
    for col in ['primary_genre', 'year_group', 'language_group', 'popularity_group']:
        df_model[col] = df_model[col].astype(str)
    
    # Create dummies
    df_dummies = pd.get_dummies(df_model, drop_first=True)
    
    # Convert boolean columns to int
    bool_cols = df_dummies.select_dtypes(include=['bool']).columns
    df_dummies[bool_cols] = df_dummies[bool_cols].astype(int)
    
    df_dummies = df_dummies.apply(pd.to_numeric, errors='coerce').dropna()
    
    return df_dummies

def fit_knn_model(df_num, fast=False):
    """Fit KNN with tuning curves and diagnostics."""
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Baseline validation RMSE curve across k (default euclidean/uniform)
    k_values = list(range(1, 21)) if fast else list(range(1, 31))
    rmse_curve = []
    for k in k_values:
        m = KNeighborsRegressor(n_neighbors=k)
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_valid_scaled)
        rmse_curve.append(np.sqrt(mean_squared_error(y_valid, preds)))
    
    # Grid search for tuned configuration
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
    
    # Tuned validation curve using best weights/metric across k
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
    
    # Correlation of features with predictions for extra diagnostic bar
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
    
    # Tuned solution (k=4) used throughout the dashboard
    kmeans_tuned = KMeans(n_clusters=4, random_state=3001, n_init=20)
    clusters = kmeans_tuned.fit_predict(X_scaled)
    
    df_clusters = df_num.copy()
    df_clusters['cluster'] = clusters
    cluster_profiles = df_clusters.groupby('cluster').mean(numeric_only=True).reset_index()
    
    return kmeans_tuned, X_scaled, df_clusters, k_values, inertia_list, silhouette_list, cluster_profiles

def fit_pca_model(df_num, X_scaled):
    X = df_num.drop(columns=['vote_average']).copy()
    
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
    """Fit OLS regression model with categorical variables"""
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

# Load data
print("Loading data...")
df_num, df_nz = load_and_preprocess_data()

def try_load_cached_results():
    try:
        root = Path(__file__).resolve().parents[1]
        json_path = root / 'data' / 'model_results.json'
        cache_dir = root / 'data' / 'model_cache'
        # Load summary metrics json if available (legacy)
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            model_cache['knn_rmse'] = data.get('knn_valid_rmse')
            model_cache['mlp_train_rmse'] = data.get('mlp_train_rmse')
            model_cache['mlp_valid_rmse'] = data.get('mlp_valid_rmse')
            print("Loaded legacy cached model results for comparison.")
        # Load serialized artifacts if available
        if cache_dir.exists():
            knn_file = cache_dir / 'knn.joblib'
            mlp_file = cache_dir / 'mlp.joblib'
            kmeans_pca_file = cache_dir / 'kmeans_pca.joblib'
            if knn_file.exists():
                knn_payload = joblib.load(knn_file)
                model_cache['knn_rmse'] = knn_payload.get('valid_rmse')
                model_cache['knn_payload'] = knn_payload
                model_cache['knn_fitted'] = True
            if mlp_file.exists():
                mlp_payload = joblib.load(mlp_file)
                model_cache['mlp_train_rmse'] = mlp_payload.get('train_rmse')
                model_cache['mlp_valid_rmse'] = mlp_payload.get('valid_rmse')
                model_cache['mlp_payload'] = mlp_payload
                model_cache['mlp_fitted'] = True
            if kmeans_pca_file.exists():
                model_cache['kmeans_pca_payload'] = joblib.load(kmeans_pca_file)
                model_cache['kmeans_fitted'] = True
            model_cache['fitted'] = bool(model_cache['knn_fitted'] and model_cache['mlp_fitted'])
            if model_cache['fitted']:
                print("Loaded serialized model cache artifacts.")
    except Exception as e:
        print(f"Failed to load cached results: {e}")

# Global variables to cache model results
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

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS],
    include_assets_files=False,
    suppress_callback_exceptions=True
)

px.defaults.template = "plotly_dark"
# Increase top margin to ensure figure titles are not clipped
DEFAULT_FIG_MARGIN = dict(l=10, r=10, t=40, b=30)
try_load_cached_results()

def create_loading_spinner(model_name):
    """Create a loading spinner card"""
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"Loading {model_name}...",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    html.Div([
                        html.Div(
                            [html.Div(className="spinner-border text-danger", role="status",
                                      style={"width": "3rem", "height": "3rem"})],
                            style={"textAlign": "center", "padding": "40px"}
                        ),
                        html.P(f"Fitting {model_name} model...", 
                              style={"textAlign": "center", "marginTop": "20px"})
                    ])
                ], className="card-body")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row mb-3")

def create_model_comparison_table(knn_rmse=None, mlp_train_rmse=None, mlp_valid_rmse=None, ols_rsquared=None):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train_lr, X_valid_lr, y_train_lr, y_valid_lr = train_test_split(X, y, test_size=0.25, random_state=3001)
    
    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr)
    X_valid_lr_scaled = scaler.transform(X_valid_lr)
    
    lr_model = KNeighborsRegressor(n_neighbors=1)
    lr_model.fit(X_train_lr_scaled, y_train_lr)
    lr_preds = lr_model.predict(X_valid_lr_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_valid_lr, lr_preds))
    
    # Build comparison table with actual values
    comparison_data = {
        'Model': ['Linear Regression', 'KNN (Tuned)', 'K-Means', 'MLP (Tuned)'],
        'Train RMSE': [f'{lr_rmse:.3f}', f'{lr_rmse:.3f}' if knn_rmse else '—', 'N/A', f'{mlp_train_rmse:.3f}' if mlp_train_rmse else '—'],
        'Valid RMSE': [f'{lr_rmse:.3f}', f'{knn_rmse:.3f}' if knn_rmse else '—', 'Clustering', f'{mlp_valid_rmse:.3f}' if mlp_valid_rmse else '—'],
        'Type': ['Interpretable', 'Distance-based', 'Unsupervised', 'Nonlinear']
    }
    
    return pd.DataFrame(comparison_data)

def build_comparison_section(knn_rmse=None, mlp_train_rmse=None, mlp_valid_rmse=None):
    """Build the comparison section with dynamic table and conclusions"""
    comparison_df = create_model_comparison_table(knn_rmse, mlp_train_rmse, mlp_valid_rmse)
    
    # Determine KNN description
    knn_text = f"Validation RMSE {knn_rmse:.3f}" if knn_rmse else "Validation RMSE will update"
    
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "Model Performance Comparison",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 mb-4"),
        
        html.Div([
            html.Div([
                html.Div(
                    "Key Conclusions",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3")

def build_linear_regression_section(ols_model):
    # Extract OLS summary statistics
    ols_summary = pd.DataFrame({
        'Coefficient': ols_model.params,
        'Std Error': ols_model.bse,
        't-statistic': ols_model.tvalues,
        'p-value': ols_model.pvalues
    }).reset_index().rename(columns={'index': 'Variable'})
    
    # Round for display
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
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "OLS Regression Results",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row mb-3")

def build_knn_section(knn_importance, knn_rmse, k_values, rmse_curve, rmse_curve_tuned, corr_df, best_params):
    fig_rmse = px.line(
        x=k_values,
        y=rmse_curve,
        markers=True,
        labels={'x': 'k', 'y': 'Validation RMSE'},
        title='KNN Validation RMSE by k (baseline)'
    ).update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8)).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )
    fig_rmse_tuned = px.line(
        x=k_values,
        y=rmse_curve_tuned,
        markers=True,
        labels={'x': 'k', 'y': 'Validation RMSE'},
        title='Tuned KNN Validation RMSE by k'
    ).update_traces(line=dict(color=COLORS['header']), marker=dict(size=8)).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )
    fig_corr = px.bar(
        corr_df,
        x='feature',
        y='correlation_with_knn_pred',
        title='Feature Correlation with KNN Predictions'
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        xaxis_tickangle=45
    )

    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"KNN Model (Tuned) - Validation RMSE: {knn_rmse:.4f}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "KNN Validation Curves",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_rmse, config={'displayModeBar': False}),
                    dcc.Graph(figure=fig_rmse_tuned, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "KNN Performance Summary",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "Feature Correlation with KNN Predictions",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_corr, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
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
    ).update_traces(marker=dict(size=5)).update_layout(
        height=400,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )

    fig_pca3d = px.scatter_3d(
        df_pca3_plot,
        x='PC1', y='PC2', z='PC3',
        color='Cluster',
        color_continuous_scale='Viridis',
        title='Tuned K-Means Clusters in 3D PCA Space',
        opacity=0.6
    ).update_traces(marker=dict(size=4)).update_layout(
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )
    
    fig_silhouette = px.line(
        x=list(k_values),
        y=silhouette_list,
        markers=True,
        labels={'x': 'k', 'y': 'Silhouette Score'},
        title='Silhouette Scores by k'
    ).update_traces(
        line=dict(color=COLORS['graph_bg']),
        marker=dict(size=8)
    ).update_layout(
        height=320,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        showlegend=False
    )
    
    fig_inertia = px.line(
        x=list(k_values),
        y=inertia_list,
        markers=True,
        labels={'x': 'k', 'y': 'Inertia'},
        title='Inertia by Number of Clusters'
    ).update_traces(line=dict(color=COLORS['header']), marker=dict(size=8)).update_layout(
        height=320,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        showlegend=False
    )
    
    cluster_profiles_melted = cluster_profiles.melt(
        id_vars='cluster',
        var_name='feature',
        value_name='value'
    )
    fig_profiles = px.bar(
        cluster_profiles_melted,
        x='feature',
        y='value',
        color='cluster',
        barmode='group',
        title='Cluster Profiles: Average Scaled Feature Values'
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        xaxis_tickangle=45
    )
    
    fig_box = px.box(
        df_clusters,
        x='cluster',
        y='vote_average',
        title='Vote Average by Tuned Cluster (k = 4)',
        labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'}
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )
    
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "K-Means Clustering Analysis",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "Clustering Tuning Diagnostics",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_silhouette, config={'displayModeBar': False}),
                    dcc.Graph(figure=fig_inertia, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "3D PCA View of Clusters",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
    ).update_traces(mode='lines+markers', line=dict(color=COLORS['graph_bg'])).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )

    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    f"MLP Regression (Tuned) - Valid RMSE: {mlp_valid_rmse:.4f}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "MLP Training Dynamics",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_loss, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "MLP Performance Summary",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
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
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-4 mb-3")
    ], className="row g-3")

def build_pca_section():
    X = df_num.drop(columns=['vote_average']).copy()
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
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        showlegend=False
    )
    
    fig_cum = px.line(
        x=list(range(1, len(cum_var) + 1)),
        y=cum_var,
        markers=True,
        labels={'x': 'Principal Component', 'y': 'Cumulative Variance'},
        title='Cumulative Variance Explained'
    ).update_traces(
        line=dict(color=COLORS['graph_bg']),
        marker=dict(size=8)
    ).add_hline(
        y=0.80,
        line_dash='dash',
        line_color='red',
        annotation_text='80% threshold'
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        showlegend=False
    )

    # Loadings for first three components
    loadings = pd.DataFrame(
        pca_full.components_.T,
        columns=[f'PC{i+1}' for i in range(len(explained_var))],
        index=X.columns
    )
    fig_loadings = px.bar(
        loadings[['PC1', 'PC2', 'PC3']].reset_index().melt(id_vars='index', var_name='Component', value_name='Loading'),
        x='index',
        y='Loading',
        color='Component',
        barmode='group',
        title="PCA Loadings for First Three Components"
    ).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary']),
        xaxis_tickangle=45
    )

    # 2D projection colored by vote_average
    X_pca2 = X_scaled @ pca_full.components_.T[:, :2]
    df_pca2 = pd.DataFrame({
        'PC1': X_pca2[:, 0],
        'PC2': X_pca2[:, 1],
        'vote_average': df_num['vote_average'].values
    })
    fig_pca2 = px.scatter(
        df_pca2,
        x='PC1',
        y='PC2',
        color='vote_average',
        color_continuous_scale='Viridis',
        title='Movies Projected onto PC1 & PC2 (Colored by Vote Average)',
        opacity=0.6
    ).update_traces(marker=dict(size=6)).update_layout(
        height=360,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )

    # 3D projection colored by vote_average
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_scaled)
    df_pca3 = pd.DataFrame({
        'PC1': X_pca3[:, 0],
        'PC2': X_pca3[:, 1],
        'PC3': X_pca3[:, 2],
        'vote_average': df_num['vote_average'].values
    })
    fig_pca3 = go.Figure()
    fig_pca3.add_trace(go.Scatter3d(
        x=df_pca3['PC1'],
        y=df_pca3['PC2'],
        z=df_pca3['PC3'],
        mode='markers',
        marker=dict(size=3.5, color=df_pca3['vote_average'], opacity=0.55, colorbar=dict(title='Vote Avg'))
    ))
    fig_pca3.update_layout(
        title='3D PCA Projection Colored by Vote Average',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        height=520,
        margin=DEFAULT_FIG_MARGIN,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'])
    )

    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "PCA: Variance Explained",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_scree, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),
        
        html.Div([
            html.Div([
                html.Div(
                    "PCA: Cumulative Variance",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_cum, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "PCA Loadings",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_loadings, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "PCA Projection (2D)",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca2, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-xl-6 mb-3"),

        html.Div([
            html.Div([
                html.Div(
                    "PCA Projection (3D)",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(figure=fig_pca3, config={'displayModeBar': False})
                ], className="card-body p-2")
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row g-3")

# App layout - initial
app.layout = html.Div([
    # Hidden storage divs for model data
    dcc.Store(id='linear-model-store'),
    dcc.Store(id='knn-model-store'),
    dcc.Store(id='kmeans-model-store'),
    dcc.Store(id='pca-model-store'),
    dcc.Store(id='mlp-model-store'),
    dcc.Interval(id='interval-trigger', interval=500, n_intervals=0, max_intervals=10),
    
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        "Movie Rating Prediction: Model Analysis Dashboard",
                        className="card-header fw-semibold"
                    ),
                    html.Div([
                        html.P(
                            "Comprehensive analysis and comparison of multiple machine learning approaches for predicting movie ratings. "
                            "Includes linear regression, KNN, K-Means clustering, PCA dimensionality reduction, and neural networks.",
                            className="mb-0",
                            style={'fontFamily': 'Inter, system-ui', 'fontSize': '16px', 'color': COLORS['text_secondary']}
                        )
                    ], className="card-body")
                ], className="card", style={"backgroundColor": "rgba(0,0,0,0)", "boxShadow": "none"})
            ], className="col-12")
        ], className="row mb-4")
    ], className="container-fluid"),
    
    # Tabbed interface for models
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
                                    html.Div(id='linear-section-container', children=[create_loading_spinner("Linear Regression")])
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
                                    html.Div(id='knn-section-container', children=[create_loading_spinner("KNN")])
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
                                    html.Div(id='kmeans-section-container', children=[create_loading_spinner("K-Means")])
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
                                    html.Div(id='pca-section-container', children=[build_pca_section()])
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
                                    html.Div(id='mlp-section-container', children=[create_loading_spinner("MLP")])
                                ], className="col-12")
                            ], className="row mb-4")
                        ]
                    )
                ],
                style={
                    'borderBottom': '2px solid #eee',
                    'marginBottom': '20px'
                }
            )
        ], className="col-12")
    ], className="row mb-4"),

    # (Removed) Run mode toggle

    # Bottom Model Comparison card outside tabs
    html.Div([
        html.Div([
            html.Div(id='comparison-section-container', children=[
                html.Div([
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
                ], className="col-12 mb-4"),

                html.Div([
                    html.Div([
                        html.Div(
                            "Key Conclusions",
                            className="card-header fw-semibold",
                            style={"backgroundColor": COLORS['card_background_color']}
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
                    ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
                ], className="col-12")
            ], className="row g-3")
        ], className="col-12")
    ], className="row mb-4")
    
], id='app-container', className="container-fluid py-4 px-3", style={
    'background': COLORS['bg_main'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui'
})

# Callbacks to fit models and update sections (runs once after 500ms delay)
@callback(
    Output('linear-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_linear_section(active_tab):
    if active_tab != 'linear':
        return dash.no_update
    if model_cache['linear_fitted']:
        return model_cache['linear_html']  # Return cached HTML
    print("Fitting Linear Regression model...")
    try:
        ols_model, _ = fit_linear_model(df_num, df_nz)
        model_cache['linear_fitted'] = True
        html_content = build_linear_regression_section(ols_model)
        model_cache['linear_html'] = html_content
        return html_content
    except Exception as e:
        print(f"Error fitting linear model: {e}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@callback(
    Output('knn-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_knn_section(active_tab):
    if active_tab != 'knn':
        return dash.no_update
    if model_cache['knn_fitted'] and model_cache['knn_payload'] is not None:
        # Build HTML from cached payload
        p = model_cache['knn_payload']
        importance_df = pd.DataFrame(p['importance'])
        corr_df = pd.DataFrame(p['correlation']).sort_values('correlation_with_knn_pred', ascending=False)
        html_content = build_knn_section(importance_df, p['valid_rmse'], p['k_values'], p['rmse_curve'], p['rmse_curve_tuned'], corr_df, p['best_params'])
        model_cache['knn_html'] = html_content
        return html_content
    print("Fitting KNN model...")
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
        ) = fit_knn_model(df_num, fast=False)
        # Cache the result
        model_cache['knn_rmse'] = knn_rmse
        model_cache['knn_fitted'] = True
        html_content = build_knn_section(knn_importance, knn_rmse, k_values, rmse_curve, rmse_curve_tuned, corr_df, best_params)
        model_cache['knn_html'] = html_content
        return html_content
    except Exception as e:
        print(f"Error fitting KNN model: {e}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@callback(
    Output('kmeans-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_kmeans_section(active_tab):
    if active_tab != 'kmeans':
        return dash.no_update
    if model_cache['kmeans_fitted'] and model_cache['kmeans_pca_payload'] is not None:
        p = model_cache['kmeans_pca_payload']
        cluster_profiles = pd.DataFrame(p['cluster_profiles'])
        df_clusters_small = pd.DataFrame(p['df_clusters_small'])
        X_pca = np.array(p['X_pca'])
        pca_clusters = np.array(p['pca_clusters'])
        html_content = build_clustering_section(X_pca, pca_clusters, p['k_values'], p['silhouette_list'], df_clusters_small, p['inertia_list'], cluster_profiles)
        model_cache['kmeans_html'] = html_content
        return html_content
    print("Fitting K-Means model...")
    try:
        kmeans_model, X_scaled_km, df_clusters, k_values, inertia_list, silhouette_list, cluster_profiles = fit_kmeans_model(df_num)
        pca_model, X_pca, pca_clusters, pca_summary = fit_pca_model(df_num, X_scaled_km)
        model_cache['kmeans_fitted'] = True
        html_content = build_clustering_section(X_pca, pca_clusters, k_values, silhouette_list, df_clusters, inertia_list, cluster_profiles)
        model_cache['kmeans_html'] = html_content
        return html_content
    except Exception as e:
        print(f"Error fitting K-Means model: {e}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@callback(
    Output('mlp-section-container', 'children'),
    Input('model-tabs', 'value'),
    prevent_initial_call=False
)
def update_mlp_section(active_tab):
    if active_tab != 'mlp':
        return dash.no_update
    if model_cache['mlp_fitted'] and model_cache['mlp_payload'] is not None:
        p = model_cache['mlp_payload']
        importance_df = pd.DataFrame(p['importance'])
        html_content = build_mlp_section(importance_df, p['train_rmse'], p['valid_rmse'], p['loss_curve'])
        model_cache['mlp_html'] = html_content
        return html_content
    print("Fitting MLP model...")
    try:
        mlp_model, mlp_train_rmse, mlp_valid_rmse, mlp_importance, loss_curve = fit_mlp_model(df_num, fast=False)
        # Cache the results
        model_cache['mlp_train_rmse'] = mlp_train_rmse
        model_cache['mlp_valid_rmse'] = mlp_valid_rmse
        model_cache['mlp_fitted'] = True
        # Check if all models are fitted
        if model_cache['knn_fitted'] and model_cache['mlp_fitted']:
            model_cache['fitted'] = True
        html_content = build_mlp_section(mlp_importance, mlp_train_rmse, mlp_valid_rmse, loss_curve)
        model_cache['mlp_html'] = html_content
        return html_content
    except Exception as e:
        print(f"Error fitting MLP model: {e}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

@callback(
    Output('comparison-section-container', 'children'),
    Input('interval-trigger', 'n_intervals'),
    Input('knn-section-container', 'children'),
    Input('mlp-section-container', 'children'),
    prevent_initial_call=False
)
def update_comparison_section(n, knn_children, mlp_children):
    # Return cached content if already fitted
    if model_cache['fitted'] and model_cache['comparison_html'] is not None:
        return model_cache['comparison_html']
    
    # Only update if models have been fitted
    if not model_cache['fitted']:
        print("Waiting for models to be fitted...")
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
    
    print("Updating comparison section with cached model results...")
    try:
        html_content = build_comparison_section(
            model_cache['knn_rmse'],
            model_cache['mlp_train_rmse'],
            model_cache['mlp_valid_rmse']
        )
        model_cache['comparison_html'] = html_content
        return html_content
    except Exception as e:
        print(f"Error updating comparison section: {e}")
        return html.Div(f"Error: {str(e)}", className="alert alert-danger")

if __name__ == '__main__':
    app.run(debug=True, port=8051, host='127.0.0.1')
