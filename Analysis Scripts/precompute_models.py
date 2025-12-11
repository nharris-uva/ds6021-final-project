import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'model_results.json'
DATA_PATH = Path(__file__).resolve().parents[1] / 'New Data and Work' / 'final_movie_table.csv'


def load_df_num(n_rows=1000):
    df = pd.read_csv(DATA_PATH).head(n_rows)
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
    return df_num


def precompute_knn(df_num):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    # Smaller grid for speed
    k_values = list(range(1, 21))
    param_grid = {
        'n_neighbors': k_values,
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_gs = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
    knn_gs.fit(X_train_scaled, y_train)
    knn_final = KNeighborsRegressor(**knn_gs.best_params_)
    knn_final.fit(X_train_scaled, y_train)
    preds_valid = knn_final.predict(X_valid_scaled)
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, preds_valid)))
    return {
        'knn_params': knn_gs.best_params_,
        'knn_valid_rmse': valid_rmse
    }


def precompute_mlp(df_num):
    X = df_num.drop(columns=['vote_average'])
    y = df_num['vote_average']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=3001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    mlp_final = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001, max_iter=300, random_state=3001)
    mlp_final.fit(X_train_scaled, y_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, mlp_final.predict(X_train_scaled))))
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, mlp_final.predict(X_valid_scaled))))
    return {
        'mlp_train_rmse': train_rmse,
        'mlp_valid_rmse': valid_rmse
    }


def main():
    df_num = load_df_num(n_rows=1000)
    results = {}
    results.update(precompute_knn(df_num))
    results.update(precompute_mlp(df_num))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote precomputed results to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
