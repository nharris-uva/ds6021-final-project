#!/usr/bin/env python3
"""
Generate and cache model and graph HTML files for the Dash dashboards.
Reads precomputed model payloads from data/model_cache and writes Plotly HTML
files to Dashboard/stored_graphs to be embedded later by a lightweight dashboard.
"""
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
OUTPUT_DIR = ROOT / "Dashboard" / "stored_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'graph_bg': '#D9376E',
    'header': '#FF8E3C',
    'text_primary': '#222222',
    'bg_transparent': 'rgba(0,0,0,0)'
}
DEFAULT_FIG_MARGIN = dict(l=20, r=20, t=60, b=40)
px.defaults.template = "plotly_dark"

# Utility

def apply_fig_theme(fig, *, height=None):
    layout_updates = {
        'margin': DEFAULT_FIG_MARGIN,
        'plot_bgcolor': COLORS['bg_transparent'],
        'paper_bgcolor': COLORS['bg_transparent'],
        'font': dict(color=COLORS['text_primary'])
    }
    if height is not None:
        fig.update_layout(height=height)
    fig.update_layout(**layout_updates)
    return fig


def write_html(fig: go.Figure, filename: str):
    path = OUTPUT_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"Wrote {path}")


def write_simple_table_html(df: pd.DataFrame, filename: str, title: str = None):
    html = [
        "<html><head><meta charset='utf-8'><title>Table</title>",
        "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\">",
        "<style>body{font-family:Inter,system-ui;color:#222;margin:12px} h3{color:#FF8E3C;margin:8px 0 12px}</style>",
        "</head><body>"
    ]
    if title:
        html.append(f"<h3 style='font-family:Inter,system-ui;color:{COLORS['header']}'>{title}</h3>")
    html.append(df.to_html(index=False, border=0, classes=["table","table-striped","table-bordered","table-sm","align-middle","text-center"]))
    html.append("</body></html>")
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {path}")


def write_comparison_table_html(df: pd.DataFrame, filename: str):
    head = (
        "<html><head><meta charset='utf-8'><title>Model Comparison</title>"
        "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\">"
        "<style>body{font-family:Inter,system-ui;color:#222;margin:12px} h3{color:#FF8E3C;margin:8px 0 12px}</style>"
        "</head><body>"
    )
    header = "<h3>Model Performance Comparison</h3>"
    table_html = df.to_html(index=False, border=0, classes=["table","table-striped","table-bordered","table-sm","align-middle","text-center"]) 
    doc = head + header + table_html + "</body></html>"
    path = OUTPUT_DIR / filename
    path.write_text(doc, encoding='utf-8')
    print(f"Wrote {path}")


def to_dataframe(payload, expected_cols: list[str]) -> pd.DataFrame | None:
    """Robustly convert various payload structures into a DataFrame with expected columns.
    Returns None if conversion is not possible or results in empty data.
    """
    if payload is None:
        return None
    # Already a DataFrame
    if isinstance(payload, pd.DataFrame):
        # Ensure columns exist
        if all(col in payload.columns for col in expected_cols) and len(payload) > 0:
            return payload
        # Attempt to rename if common mismatches
        df = payload.copy()
        col_map = {
            'features': 'feature',
            'importances': 'importance',
            'corr': 'correlation_with_knn_pred',
        }
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df = df.rename(columns={src: dst})
        return df if all(c in df.columns for c in expected_cols) and len(df) > 0 else None
    # Dict-like with arrays
    if isinstance(payload, dict):
        df_candidate = pd.DataFrame(payload)
        if len(df_candidate) == 0:
            # Handle dict of lists with equal length
            try:
                keys = list(payload.keys())
                lens = {k: len(v) for k, v in payload.items() if hasattr(v, '__len__')}
                if lens and len(set(lens.values())) == 1:
                    df_candidate = pd.DataFrame(payload)
            except Exception:
                pass
        if all(c in df_candidate.columns for c in expected_cols) and len(df_candidate) > 0:
            return df_candidate
        # Could be nested under a key
        for v in payload.values():
            try:
                df_nested = pd.DataFrame(v)
                if all(c in df_nested.columns for c in expected_cols) and len(df_nested) > 0:
                    return df_nested
            except Exception:
                continue
        return None
    # List of dicts
    if isinstance(payload, list):
        try:
            df_candidate = pd.DataFrame(payload)
            return df_candidate if all(c in df_candidate.columns for c in expected_cols) and len(df_candidate) > 0 else None
        except Exception:
            return None
    return None


# Load data (for clustering profiles if needed)
final_csv = ROOT / "New Data and Work" / "final_movie_table.csv"
raw_df = None
if final_csv.exists():
    try:
        raw_df = pd.read_csv(final_csv)
    except Exception:
        raw_df = None


# KNN payload
try:
    knn_payload = joblib.load(MODEL_CACHE_DIR / "knn.joblib")
except Exception:
    knn_payload = None

if knn_payload:
    # Expect keys based on simple_style_dashboard.py: importance_df, k_values, rmse_curve, rmse_curve_tuned, corr_df, best_params, valid_rmse
    importance_df = to_dataframe(knn_payload.get('importance_df'), ['feature', 'importance'])
    corr_df = to_dataframe(knn_payload.get('corr_df'), ['feature', 'correlation_with_knn_pred'])
    k_values = knn_payload.get('k_values') or list(range(1, 21))
    rmse_curve = knn_payload.get('rmse_curve') or [None]*len(k_values)
    rmse_curve_tuned = knn_payload.get('rmse_curve_tuned') or [None]*len(k_values)
    best_params = knn_payload.get('best_params') or {}

    if importance_df is not None and {'feature','importance'}.issubset(importance_df.columns) and len(importance_df) > 0:
        fig_imp = px.bar(
            importance_df,
            x='importance', y='feature', orientation='h',
            color_discrete_sequence=[COLORS['graph_bg']],
            title='KNN: Permutation Importance'
        ).update_layout(yaxis=dict(autorange='reversed'))
        fig_imp = apply_fig_theme(fig_imp, height=400)
        write_html(fig_imp, 'knn_importance.html')

    fig_rmse = px.line(x=k_values, y=rmse_curve, markers=True,
                       labels={'x': 'k', 'y': 'Validation RMSE'},
                       title='KNN Validation RMSE by k (baseline)')
    fig_rmse.update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8))
    fig_rmse = apply_fig_theme(fig_rmse, height=360)
    write_html(fig_rmse, 'knn_rmse_curve.html')

    fig_rmse_tuned = px.line(x=k_values, y=rmse_curve_tuned, markers=True,
                             labels={'x': 'k', 'y': 'Validation RMSE'},
                             title='Tuned KNN Validation RMSE by k')
    fig_rmse_tuned.update_traces(line=dict(color=COLORS['header']), marker=dict(size=8))
    fig_rmse_tuned = apply_fig_theme(fig_rmse_tuned, height=360)
    write_html(fig_rmse_tuned, 'knn_rmse_tuned.html')

    if corr_df is not None and {'feature','correlation_with_knn_pred'}.issubset(corr_df.columns) and len(corr_df) > 0:
        fig_corr = px.bar(corr_df, x='feature', y='correlation_with_knn_pred',
                          title='Feature Correlation with KNN Predictions')
        fig_corr.update_layout(xaxis_tickangle=45)
        fig_corr = apply_fig_theme(fig_corr, height=360)
        write_html(fig_corr, 'knn_feature_corr.html')

    # Summary table
    summary_rows = {
        'Neighbors (k)': [best_params.get('n_neighbors')],
        'Weighting': [str(best_params.get('weights', '')).title()],
        'Distance Metric': [str(best_params.get('metric', '')).title()],
    }
    write_simple_table_html(pd.DataFrame(summary_rows), 'knn_summary.html', title='KNN Configuration')

# KMeans/PCA payload
try:
    kmeans_payload = joblib.load(MODEL_CACHE_DIR / "kmeans_pca.joblib")
except Exception:
    kmeans_payload = None

if kmeans_payload:
    # Expect keys: X_pca, pca_clusters, k_values, silhouette_list, df_clusters, inertia_list, cluster_profiles
    # Convert to DataFrame/ndarray if necessary
    X_pca = np.asarray(kmeans_payload.get('X_pca')) if kmeans_payload.get('X_pca') is not None else None
    pca_clusters = np.asarray(kmeans_payload.get('pca_clusters')) if kmeans_payload.get('pca_clusters') is not None else None
    k_values = kmeans_payload.get('k_values') or list(range(2, 11))
    silhouette_list = kmeans_payload.get('silhouette_list') or [None]*len(k_values)
    inertia_list = kmeans_payload.get('inertia_list') or [None]*len(k_values)
    df_clusters_payload = kmeans_payload.get('df_clusters')
    cluster_profiles_payload = kmeans_payload.get('cluster_profiles')
    df_clusters = pd.DataFrame(df_clusters_payload) if (df_clusters_payload is not None and not isinstance(df_clusters_payload, pd.DataFrame)) else df_clusters_payload
    cluster_profiles = pd.DataFrame(cluster_profiles_payload) if (cluster_profiles_payload is not None and not isinstance(cluster_profiles_payload, pd.DataFrame)) else cluster_profiles_payload

    # 2D PCA scatter
    if X_pca is not None and pca_clusters is not None and X_pca.ndim == 2 and X_pca.shape[0] == pca_clusters.shape[0] and X_pca.shape[1] >= 2:
        df_pca_plot = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': pca_clusters})
        fig_pca = px.scatter(df_pca_plot, x='PC1', y='PC2', color='Cluster', color_continuous_scale='Viridis',
                             opacity=0.6, title='Tuned K-Means Clusters in PCA Space')
        fig_pca.update_traces(marker=dict(size=5))
        fig_pca = apply_fig_theme(fig_pca, height=400)
        write_html(fig_pca, 'kmeans_pca_2d.html')

    # 3D PCA
    if X_pca is not None and pca_clusters is not None and X_pca.ndim == 2 and X_pca.shape[0] == pca_clusters.shape[0] and X_pca.shape[1] >= 3:
        df_pca3_plot = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2], 'Cluster': pca_clusters})
        fig_pca3d = px.scatter_3d(df_pca3_plot, x='PC1', y='PC2', z='PC3', color='Cluster', color_continuous_scale='Viridis',
                                  title='Tuned K-Means Clusters in 3D PCA Space', opacity=0.6)
        fig_pca3d.update_traces(marker=dict(size=4))
        fig_pca3d = apply_fig_theme(fig_pca3d, height=420)
        write_html(fig_pca3d, 'kmeans_pca_3d.html')

    # Silhouette and inertia
    fig_sil = px.line(x=list(k_values), y=silhouette_list, markers=True,
                      labels={'x': 'k', 'y': 'Silhouette Score'}, title='Silhouette Scores by k')
    fig_sil.update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8))
    fig_sil.update_layout(showlegend=False)
    fig_sil = apply_fig_theme(fig_sil, height=320)
    write_html(fig_sil, 'kmeans_silhouette.html')

    fig_inertia = px.line(x=list(k_values), y=inertia_list, markers=True,
                          labels={'x': 'k', 'y': 'Inertia'}, title='Inertia by Number of Clusters')
    fig_inertia.update_traces(line=dict(color=COLORS['header']), marker=dict(size=8))
    fig_inertia.update_layout(showlegend=False)
    fig_inertia = apply_fig_theme(fig_inertia, height=320)
    write_html(fig_inertia, 'kmeans_inertia.html')

    # Profiles and box plot
    if isinstance(cluster_profiles, pd.DataFrame) and 'cluster' in cluster_profiles.columns and len(cluster_profiles) > 0:
        try:
            cluster_profiles_melted = cluster_profiles.melt(id_vars='cluster', var_name='feature', value_name='value')
            fig_profiles = px.bar(cluster_profiles_melted, x='feature', y='value', color='cluster', barmode='group',
                                  title='Cluster Profiles: Average Scaled Feature Values')
            fig_profiles.update_layout(xaxis_tickangle=45)
            fig_profiles = apply_fig_theme(fig_profiles, height=360)
            write_html(fig_profiles, 'kmeans_cluster_profiles.html')
        except Exception:
            pass

    if isinstance(df_clusters, pd.DataFrame) and {'cluster','vote_average'}.issubset(df_clusters.columns) and len(df_clusters) > 0:
        try:
            fig_box = px.box(df_clusters, x='cluster', y='vote_average',
                             title='Vote Average by Tuned Cluster (k = 4)', labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'})
            fig_box = apply_fig_theme(fig_box, height=360)
            write_html(fig_box, 'kmeans_vote_box.html')
        except Exception:
            pass

# MLP payload
try:
    mlp_payload = joblib.load(MODEL_CACHE_DIR / "mlp.joblib")
except Exception:
    mlp_payload = None

if mlp_payload:
    importance_df = to_dataframe(mlp_payload.get('importance_df'), ['feature', 'importance'])
    loss_curve = mlp_payload.get('loss_curve') or []

    if importance_df is not None and {'feature','importance'}.issubset(importance_df.columns) and len(importance_df) > 0:
        fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h',
                         color_discrete_sequence=[COLORS['graph_bg']], title='MLP: Permutation Importance')
        fig_imp.update_layout(yaxis=dict(autorange='reversed'))
        fig_imp = apply_fig_theme(fig_imp, height=400)
        write_html(fig_imp, 'mlp_importance.html')

    fig_loss = px.line(x=list(range(len(loss_curve))), y=loss_curve, markers=True,
                       labels={'x': 'Iteration', 'y': 'Training Loss'}, title='MLP Training Loss Curve')
    fig_loss.update_traces(mode='lines+markers', line=dict(color=COLORS['graph_bg']))
    fig_loss = apply_fig_theme(fig_loss, height=360)
    write_html(fig_loss, 'mlp_loss_curve.html')

# Comparison table
# Deprecated: JSON-driven comparison. We now always rebuild from CSV below.

print("Graph cache generation complete.")

# --- Linear OLS (computed directly from final_movie_table.csv) ---
try:
    if raw_df is not None and {'budget','revenue','vote_count','user_rating_count','keyword_count','runtime','vote_average'}.issubset(raw_df.columns):
        df = raw_df.copy()
        # Filter budget > 0 to match modeling
        df = df[df['budget'] > 0].copy()
        # Engineer log features
        df['log_budget'] = np.log1p(df['budget'])
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_vote_count'] = np.log1p(df['vote_count'])
        df['log_user_rating_count'] = np.log1p(df['user_rating_count'])
        df['log_keyword_count'] = np.log1p(df['keyword_count'])
        # Categorical features
        if 'genres' in df.columns:
            df['primary_genre'] = df['genres'].astype(str).str.split('|').str[0]
        if 'release_year' in df.columns:
            df['year_group'] = pd.cut(
                df['release_year'],
                bins=[0, 1919, 1949, 1979, 1999, 2009, 2019],
                labels=['Pre-1920', '1920-1949', '1950-1979', '1980-1999', '2000-2009', '2010-2019']
            )
        if 'original_language' in df.columns:
            df['language_group'] = df['original_language'].apply(
                lambda x: x if x in ['en','fr','ru','hi','es','de','ja','it'] else 'Other'
            )
        if 'lead_actor' in df.columns:
            counts = df.groupby('lead_actor').size().reset_index(name='movie_count')
            counts['popularity_group'] = pd.cut(
                counts['movie_count'],
                bins=[0, 1, 5, float('inf')],
                labels=['Low Popularity', 'Medium Popularity', 'High Popularity']
            )
            df = df.merge(counts[['lead_actor','popularity_group']], on='lead_actor', how='left')

        model_vars = [
            'vote_average',
            'log_budget','log_revenue','log_vote_count','log_user_rating_count','log_keyword_count','runtime',
            'primary_genre','year_group','language_group','popularity_group'
        ]
        df_model = df[[c for c in model_vars if c in df.columns]].dropna().copy()
        # Factorize categoricals into string type to ensure get_dummies works
        for col in ['primary_genre','year_group','language_group','popularity_group']:
            if col in df_model.columns:
                df_model[col] = df_model[col].astype(str)
        dummies = pd.get_dummies(df_model, drop_first=True)
        # Ensure all bools -> int
        bool_cols = dummies.select_dtypes(include=['bool']).columns
        dummies[bool_cols] = dummies[bool_cols].astype(int)
        dummies = dummies.apply(pd.to_numeric, errors='coerce').dropna()

        y = dummies['vote_average']
        X = dummies.drop(columns=['vote_average'])
        X = sm.add_constant(X)
        ols = sm.OLS(y, X).fit()

        # Build summary table similar to dashboard
        ols_summary = pd.DataFrame({
            'Variable': ols.params.index,
            'Coefficient': ols.params.values,
            'Std Error': ols.bse.values,
            't-statistic': ols.tvalues.values,
            'p-value': ols.pvalues.values,
        })
        ols_summary['Coefficient'] = ols_summary['Coefficient'].round(6)
        ols_summary['Std Error'] = ols_summary['Std Error'].round(6)
        ols_summary['t-statistic'] = ols_summary['t-statistic'].round(4)
        ols_summary['p-value'] = ols_summary['p-value'].round(6)

        # Compose HTML with stats header + table
        # Compose styled HTML with inline CSS to ensure proper table rendering
        css = """
        <style>
        body{font-family:Inter,system-ui;color:#222;margin:12px}
        table{border-collapse:collapse;width:100%;font-size:13px}
        th,td{border:1px solid #ddd;padding:8px;text-align:center}
        thead th{background:#FF8E3C;color:#fff;font-weight:bold}
        tbody tr:nth-child(odd){background:#EFF0F3}
        h3{color:#FF8E3C;margin:8px 0 12px}
        .stat-label{font-weight:600;margin-right:6px}
        .stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:8px 0 16px}
        .stat-item{padding:6px 8px;border:1px solid #ddd;background:#fafafa;text-align:center}
        </style>
        """
        stats_html = f"""
        <h3>Linear Regression Analysis with Categorical Features</h3>
        <div class='stat-grid'>
            <div class='stat-item'><span class='stat-label'>R-squared</span>{ols.rsquared:.4f}</div>
            <div class='stat-item'><span class='stat-label'>Adj. R-squared</span>{ols.rsquared_adj:.4f}</div>
            <div class='stat-item'><span class='stat-label'>F-statistic</span>{ols.fvalue:.2f}</div>
            <div class='stat-item'><span class='stat-label'>Prob(F)</span>{ols.f_pvalue:.2e}</div>
            <div class='stat-item'><span class='stat-label'>Observations</span>{len(ols.resid)}</div>
        </div>
        """
        table_html = ols_summary.to_html(index=False, border=0)
        doc = "<html><head><meta charset='utf-8'><title>OLS Summary</title>" + css + "</head><body>" + stats_html + table_html + "</body></html>"
        path = OUTPUT_DIR / 'linear_ols_table.html'
        path.write_text(doc, encoding='utf-8')
        print(f"Wrote {path}")
except Exception as e:
    print(f"Skipping OLS generation due to error: {e}")

# --- PCA Summary (computed if raw_df available) ---
try:
    if raw_df is not None and {'budget','revenue','vote_count','user_rating_count','keyword_count','runtime','vote_average'}.issubset(raw_df.columns):
        df = raw_df.copy()
        df = df[df['budget'] > 0].copy()
        df['log_budget'] = np.log1p(df['budget'])
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_vote_count'] = np.log1p(df['vote_count'])
        df['log_user_rating_count'] = np.log1p(df['user_rating_count'])
        df['log_keyword_count'] = np.log1p(df['keyword_count'])
        numeric_vars = [
            'vote_average','log_budget','log_revenue','log_vote_count','log_user_rating_count','log_keyword_count','runtime'
        ]
        df_num = df[numeric_vars].dropna().copy()
        X = df_num.drop(columns=['vote_average'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        pca_summary = pd.DataFrame({
            'PC': np.arange(1, 4),
            'ExplainedVariance': pca.explained_variance_ratio_[:3],
            'CumulativeVariance': np.cumsum(pca.explained_variance_ratio_[:3])
        })
        # Write simple HTML table
        write_simple_table_html(pca_summary, 'pca_summary.html', title='PCA Explained Variance (Top 3 PCs)')
        # Explained variance bar
        fig_ev = px.bar(pca_summary, x='PC', y='ExplainedVariance', title='Explained Variance by Principal Component')
        fig_ev = apply_fig_theme(fig_ev, height=320)
        write_html(fig_ev, 'pca_explained_variance.html')
        # PCA 2D scatter (PC1 vs PC2)
        df_pca2 = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1]})
        fig_pca2 = px.scatter(df_pca2, x='PC1', y='PC2', opacity=0.6, title='PCA Scatter: PC1 vs PC2')
        fig_pca2.update_traces(marker=dict(size=5))
        fig_pca2 = apply_fig_theme(fig_pca2, height=380)
        write_html(fig_pca2, 'pca_scatter_2d.html')
        # PCA 3D scatter (PC1, PC2, PC3)
        df_pca3 = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1], 'PC3': X_pca[:,2]})
        fig_pca3 = px.scatter_3d(df_pca3, x='PC1', y='PC2', z='PC3', title='PCA Scatter: 3D (PC1, PC2, PC3)', opacity=0.6)
        fig_pca3.update_traces(marker=dict(size=4))
        fig_pca3 = apply_fig_theme(fig_pca3, height=400)
        write_html(fig_pca3, 'pca_scatter_3d.html')
        # Loadings heatmap for PC1..PC3
        loadings = pd.DataFrame(pca.components_[:3], columns=X.columns, index=['PC1','PC2','PC3'])
        loadings_melt = loadings.reset_index().melt(id_vars='index', var_name='feature', value_name='loading').rename(columns={'index':'PC'})
        fig_load = px.imshow(loadings.values, x=X.columns, y=['PC1','PC2','PC3'], color_continuous_scale='RdBu', origin='lower',
                             title='PCA Loadings (PC1..PC3)')
        fig_load = apply_fig_theme(fig_load, height=380)
        write_html(fig_load, 'pca_loadings.html')
except Exception as e:
    print(f"Skipping PCA summary/plots due to error: {e}")

# --- Fallbacks from CSV to ensure required graphs exist ---
try:
    # If required files missing, compute minimal models from CSV
    required_files = [
        'knn_importance.html', 'knn_feature_corr.html',
        'kmeans_vote_box.html', 'comparison_table.html',
        'mlp_importance.html',
        'kmeans_pca_2d.html','kmeans_pca_3d.html','kmeans_silhouette.html','kmeans_inertia.html','kmeans_cluster_profiles.html'
    ]
    missing = [f for f in required_files if not (OUTPUT_DIR / f).exists()]
    if raw_df is not None and missing:
        df = raw_df.copy()
        df = df[df['budget'] > 0].copy()
        # Engineer features
        df['log_budget'] = np.log1p(df['budget'])
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_vote_count'] = np.log1p(df['vote_count'])
        df['log_user_rating_count'] = np.log1p(df['user_rating_count'])
        df['log_keyword_count'] = np.log1p(df['keyword_count'])
        numeric_vars = ['vote_average','log_budget','log_revenue','log_vote_count','log_user_rating_count','log_keyword_count','runtime']
        df_num = df[numeric_vars].dropna().copy()
        X = df_num.drop(columns=['vote_average'])
        y = df_num['vote_average']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KNN fallback
        # Always compute comparison table from CSV (and generate KNN artifacts if missing)
        if True:
            try:
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
                X_train_s = scaler.fit_transform(X_train)
                X_valid_s = scaler.transform(X_valid)
                knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
                knn.fit(X_train_s, y_train)
                preds_train = knn.predict(X_train_s)
                preds_valid = knn.predict(X_valid_s)
                valid_rmse = float(np.sqrt(mean_squared_error(y_valid, preds_valid)))
                # Importance via permutation
                r = permutation_importance(knn, X_valid_s, y_valid, n_repeats=5, random_state=3001)
                importance_df_fb = pd.DataFrame({'feature': X.columns, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
                fig_imp = px.bar(importance_df_fb, x='importance', y='feature', orientation='h', color_discrete_sequence=[COLORS['graph_bg']], title='KNN: Permutation Importance')
                fig_imp.update_layout(yaxis=dict(autorange='reversed'))
                fig_imp = apply_fig_theme(fig_imp, height=400)
                write_html(fig_imp, 'knn_importance.html')
                # Correlation with predictions
                corr_vals = [np.corrcoef(X_train_s[:, i], preds_train)[0, 1] for i in range(X_train_s.shape[1])]
                corr_df_fb = pd.DataFrame({'feature': X.columns, 'correlation_with_knn_pred': corr_vals}).sort_values('correlation_with_knn_pred', ascending=False)
                fig_corr = px.bar(corr_df_fb, x='feature', y='correlation_with_knn_pred', title='Feature Correlation with KNN Predictions')
                fig_corr.update_layout(xaxis_tickangle=45)
                fig_corr = apply_fig_theme(fig_corr, height=360)
                write_html(fig_corr, 'knn_feature_corr.html')
                # Compute Linear Regression baseline
                lr = LinearRegression()
                lr.fit(X_train_s, y_train)
                lr_train_rmse = float(np.sqrt(mean_squared_error(y_train, lr.predict(X_train_s))))
                lr_valid_rmse = float(np.sqrt(mean_squared_error(y_valid, lr.predict(X_valid_s))))

                # If MLP importance was created, also compute its RMSE
                mlp_train_rmse = '—'
                mlp_valid_rmse = '—'
                if (OUTPUT_DIR / 'mlp_importance.html').exists():
                    try:
                        from sklearn.neural_network import MLPRegressor
                        mlp = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001, max_iter=300, random_state=3001)
                        mlp.fit(X_train_s, y_train)
                        mlp_train_rmse = f"{np.sqrt(mean_squared_error(y_train, mlp.predict(X_train_s))):.3f}"
                        mlp_valid_rmse = f"{np.sqrt(mean_squared_error(y_valid, mlp.predict(X_valid_s))):.3f}"
                    except Exception:
                        pass

                # Build comparison table (always overwrite)
                df_comp = pd.DataFrame({
                    'Model': ['Linear Regression', 'KNN (Fallback)', 'K-Means', 'MLP (Fallback)'],
                    'Train RMSE': [f'{lr_train_rmse:.3f}', '—', 'N/A', mlp_train_rmse],
                    'Valid RMSE': [f'{lr_valid_rmse:.3f}', f'{valid_rmse:.3f}', 'Clustering', mlp_valid_rmse],
                    'Type': ['Interpretable', 'Distance-based', 'Unsupervised', 'Nonlinear']
                })
                write_comparison_table_html(df_comp, 'comparison_table.html')
            except Exception:
                pass

        # KMeans/PCA fallbacks
        if any(f in missing for f in ['kmeans_vote_box.html','kmeans_pca_2d.html','kmeans_pca_3d.html','kmeans_silhouette.html','kmeans_inertia.html','kmeans_cluster_profiles.html']):
            try:
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled)
                # Tune k by silhouette on 2..10
                k_values = list(range(2, 11))
                inertia_list = []
                silhouette_list = []
                for k in k_values:
                    km = KMeans(n_clusters=k, random_state=3001, n_init=10)
                    labels = km.fit_predict(X_scaled)
                    inertia_list.append(float(km.inertia_))
                    try:
                        silhouette_list.append(float(silhouette_score(X_scaled, labels)))
                    except Exception:
                        silhouette_list.append(None)
                # Use k=4 as in original
                km4 = KMeans(n_clusters=4, random_state=3001, n_init=20)
                clusters = km4.fit_predict(X_scaled)
                df_clusters = df_num.copy()
                df_clusters['cluster'] = clusters
                cluster_profiles = pd.DataFrame(X_scaled, columns=X.columns)
                cluster_profiles['cluster'] = clusters
                cluster_profiles = cluster_profiles.groupby('cluster').mean(numeric_only=True).reset_index()
                # Write plots
                df_pca_plot = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': clusters})
                fig_pca = px.scatter(df_pca_plot, x='PC1', y='PC2', color='Cluster', color_continuous_scale='Viridis', opacity=0.6, title='Tuned K-Means Clusters in PCA Space')
                fig_pca.update_traces(marker=dict(size=5))
                fig_pca = apply_fig_theme(fig_pca, height=400)
                write_html(fig_pca, 'kmeans_pca_2d.html')

                df_pca3_plot = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2], 'Cluster': clusters})
                fig_pca3d = px.scatter_3d(df_pca3_plot, x='PC1', y='PC2', z='PC3', color='Cluster', color_continuous_scale='Viridis', title='Tuned K-Means Clusters in 3D PCA Space', opacity=0.6)
                fig_pca3d.update_traces(marker=dict(size=4))
                fig_pca3d = apply_fig_theme(fig_pca3d, height=420)
                write_html(fig_pca3d, 'kmeans_pca_3d.html')

                fig_sil = px.line(x=k_values, y=silhouette_list, markers=True, labels={'x': 'k', 'y': 'Silhouette Score'}, title='Silhouette Scores by k')
                fig_sil.update_traces(line=dict(color=COLORS['graph_bg']), marker=dict(size=8))
                fig_sil.update_layout(showlegend=False)
                fig_sil = apply_fig_theme(fig_sil, height=320)
                write_html(fig_sil, 'kmeans_silhouette.html')

                fig_inertia = px.line(x=k_values, y=inertia_list, markers=True, labels={'x': 'k', 'y': 'Inertia'}, title='Inertia by Number of Clusters')
                fig_inertia.update_traces(line=dict(color=COLORS['header']), marker=dict(size=8))
                fig_inertia.update_layout(showlegend=False)
                fig_inertia = apply_fig_theme(fig_inertia, height=320)
                write_html(fig_inertia, 'kmeans_inertia.html')

                cluster_profiles_melted = cluster_profiles.melt(id_vars='cluster', var_name='feature', value_name='value')
                fig_profiles = px.bar(cluster_profiles_melted, x='feature', y='value', color='cluster', barmode='group', title='Cluster Profiles: Average Scaled Feature Values')
                fig_profiles.update_layout(xaxis_tickangle=45)
                fig_profiles = apply_fig_theme(fig_profiles, height=360)
                write_html(fig_profiles, 'kmeans_cluster_profiles.html')

                fig_box = px.box(df_clusters, x='cluster', y='vote_average', title='Vote Average by Tuned Cluster (k = 4)', labels={'cluster': 'Cluster', 'vote_average': 'Vote Average'})
                fig_box = apply_fig_theme(fig_box, height=360)
                write_html(fig_box, 'kmeans_vote_box.html')
            except Exception:
                pass

        # MLP fallback (importance only)
        if 'mlp_importance.html' in missing:
            try:
                from sklearn.neural_network import MLPRegressor
                from sklearn.model_selection import train_test_split
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
                X_train_s = scaler.fit_transform(X_train)
                X_valid_s = scaler.transform(X_valid)
                mlp = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001, max_iter=300, random_state=3001)
                mlp.fit(X_train_s, y_train)
                r = permutation_importance(mlp, X_valid_s, y_valid, n_repeats=10, random_state=3001)
                importance_df_fb = pd.DataFrame({'feature': X.columns, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
                fig_imp = px.bar(importance_df_fb, x='importance', y='feature', orientation='h', color_discrete_sequence=[COLORS['graph_bg']], title='MLP: Permutation Importance')
                fig_imp.update_layout(yaxis=dict(autorange='reversed'))
                fig_imp = apply_fig_theme(fig_imp, height=400)
                write_html(fig_imp, 'mlp_importance.html')
            except Exception:
                pass
except Exception as e:
    print(f"Fallback generation error: {e}")

# --- Always rebuild Model Comparison from CSV ---
try:
    if raw_df is not None:
        df = raw_df.copy()
        df = df[df['budget'] > 0].copy()
        df['log_budget'] = np.log1p(df['budget'])
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_vote_count'] = np.log1p(df['vote_count'])
        df['log_user_rating_count'] = np.log1p(df['user_rating_count'])
        df['log_keyword_count'] = np.log1p(df['keyword_count'])
        numeric_vars = ['vote_average','log_budget','log_revenue','log_vote_count','log_user_rating_count','log_keyword_count','runtime']
        df_num = df[numeric_vars].dropna().copy()
        X = df_num.drop(columns=['vote_average'])
        y = df_num['vote_average']
        scaler = StandardScaler()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
        X_train_s = scaler.fit_transform(X_train)
        X_valid_s = scaler.transform(X_valid)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_s, y_train)
        lr_train_rmse = float(np.sqrt(mean_squared_error(y_train, lr.predict(X_train_s))))
        lr_valid_rmse = float(np.sqrt(mean_squared_error(y_valid, lr.predict(X_valid_s))))

        # KNN
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
        knn.fit(X_train_s, y_train)
        knn_valid_rmse = float(np.sqrt(mean_squared_error(y_valid, knn.predict(X_valid_s))))

        # Optional MLP
        mlp_train_rmse = '—'
        mlp_valid_rmse = '—'
        try:
            from sklearn.neural_network import MLPRegressor
            mlp = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001, max_iter=300, random_state=3001)
            mlp.fit(X_train_s, y_train)
            mlp_train_rmse = f"{np.sqrt(mean_squared_error(y_train, mlp.predict(X_train_s))):.3f}"
            mlp_valid_rmse = f"{np.sqrt(mean_squared_error(y_valid, mlp.predict(X_valid_s))):.3f}"
        except Exception:
            pass

        df_comp = pd.DataFrame({
            'Model': ['Linear Regression', 'KNN (Fallback)', 'K-Means', 'MLP (Fallback)'],
            'Train RMSE': [f'{lr_train_rmse:.3f}', '—', 'N/A', mlp_train_rmse],
            'Valid RMSE': [f'{lr_valid_rmse:.3f}', f'{knn_valid_rmse:.3f}', 'Clustering', mlp_valid_rmse],
            'Type': ['Interpretable', 'Distance-based', 'Unsupervised', 'Nonlinear']
        })
        write_comparison_table_html(df_comp, 'comparison_table.html')
except Exception as e:
    print(f"Comparison rebuild error: {e}")
