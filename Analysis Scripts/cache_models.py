import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Re-implement minimal model computations to serialize artifacts
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'New Data and Work' / 'final_movie_table.csv'
CACHE_DIR = ROOT / 'data' / 'model_cache'


def load_df(num_rows=1000):
    df = pd.read_csv(DATA_PATH).head(num_rows)
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


def cache_knn(df_num):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    k_values = list(range(1, 31))
    param_grid = {
        'n_neighbors': k_values,
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_gs = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    knn_gs.fit(X_train_scaled, y_train)

    knn_final = KNeighborsRegressor(**knn_gs.best_params_)
    knn_final.fit(X_train_scaled, y_train)
    preds_valid = knn_final.predict(X_valid_scaled)
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, preds_valid)))

    rmse_curve = []
    for k in k_values:
        m = KNeighborsRegressor(n_neighbors=k)
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_valid_scaled)
        rmse_curve.append(float(np.sqrt(mean_squared_error(y_valid, preds))))

    rmse_curve_tuned = []
    for k in k_values:
        tuned_model = KNeighborsRegressor(
            n_neighbors=k,
            weights=knn_gs.best_params_['weights'],
            metric=knn_gs.best_params_['metric']
        )
        tuned_model.fit(X_train_scaled, y_train)
        tuned_preds = tuned_model.predict(X_valid_scaled)
        rmse_curve_tuned.append(float(np.sqrt(mean_squared_error(y_valid, tuned_preds))))

    preds_train = knn_final.predict(X_train_scaled)
    corr = [
        float(np.corrcoef(X_train_scaled[:, i], preds_train)[0, 1])
        for i in range(X_train_scaled.shape[1])
    ]
    r = permutation_importance(knn_final, X_valid_scaled, y_valid, n_repeats=10, random_state=3001)

    payload = {
        'best_params': knn_gs.best_params_,
        'valid_rmse': valid_rmse,
        'k_values': k_values,
        'rmse_curve': rmse_curve,
        'rmse_curve_tuned': rmse_curve_tuned,
        'importance': {
            'feature': list(X.columns),
            'importance': list(r.importances_mean)
        },
        'correlation': {
            'feature': list(X.columns),
            'correlation_with_knn_pred': corr
        },
        'scaler': scaler,
        'model': knn_final
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, CACHE_DIR / 'knn.joblib')
    return valid_rmse


def cache_kmeans_pca(df_num):
    X = df_num.drop(columns=['vote_average']).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = list(range(2, 11))
    inertia_list, silhouette_list = [], []
    from sklearn.metrics import silhouette_score
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=3001, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia_list.append(float(km.inertia_))
        silhouette_list.append(float(silhouette_score(X_scaled, labels)))

    kmeans_tuned = KMeans(n_clusters=4, random_state=3001, n_init=20)
    clusters = kmeans_tuned.fit_predict(X_scaled)

    df_clusters = df_num.copy()
    df_clusters['cluster'] = clusters
    cluster_profiles = df_clusters.groupby('cluster').mean(numeric_only=True).reset_index()
    df_clusters_small = df_clusters[['cluster', 'vote_average']]

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    kmeans_pca = KMeans(n_clusters=4, random_state=3001, n_init=20)
    clusters_pca = kmeans_pca.fit_predict(X_pca)

    payload = {
        'k_values': k_values,
        'inertia_list': inertia_list,
        'silhouette_list': silhouette_list,
        'cluster_profiles': cluster_profiles,
        'df_clusters_small': df_clusters_small,
        'X_pca': X_pca,
        'pca_clusters': clusters_pca,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, CACHE_DIR / 'kmeans_pca.joblib')


def cache_mlp(df_num):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    mlp_final = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001, max_iter=500, random_state=3001)
    mlp_final.fit(X_train_scaled, y_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, mlp_final.predict(X_train_scaled))))
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, mlp_final.predict(X_valid_scaled))))
    r_final = permutation_importance(mlp_final, X_valid_scaled, y_valid, n_repeats=20, random_state=3001)
    payload = {
        'train_rmse': train_rmse,
        'valid_rmse': valid_rmse,
        'importance': {
            'feature': list(X.columns),
            'importance': list(r_final.importances_mean)
        },
        'loss_curve': list(mlp_final.loss_curve_),
        'scaler': scaler,
        'model': mlp_final
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, CACHE_DIR / 'mlp.joblib')


def main():
    df_num, df_nz = load_df(num_rows=1000)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print('Caching KNN artifacts...')
    knn_rmse = cache_knn(df_num)
    print(f'KNN valid RMSE: {knn_rmse:.4f}')
    print('Caching KMeans/PCA artifacts...')
    cache_kmeans_pca(df_num)
    print('Caching MLP artifacts...')
    cache_mlp(df_num)
    meta = {
        'source_rows': len(df_num),
        'generated': True
    }
    with open(CACHE_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Cache written to {CACHE_DIR}')


if __name__ == '__main__':
    main()
