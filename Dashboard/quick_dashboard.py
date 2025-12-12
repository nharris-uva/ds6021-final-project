#!/usr/bin/env python3
"""
A lightweight Dash app that mirrors the layout of simple_style_dashboard.py
but loads pre-rendered HTML graphs from Dashboard/stored_graphs instead of
running heavy model computations.
"""
import dash
from dash import dcc, html
from pathlib import Path

# Colors and style constants (matching simple_style_dashboard.py)
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

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "Dashboard" / "stored_graphs"

app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS],
    include_assets_files=True,
    suppress_callback_exceptions=True
)
server = app.server

# Helper to embed stored HTML graphs

def graph_iframe(filename: str, height: int = 380):
    path = GRAPH_DIR / filename
    src_doc = path.read_text(encoding='utf-8') if path.exists() else f"<html><body><p style='color:red'>Missing {filename}</p></body></html>"
    return html.Iframe(srcDoc=src_doc, style={"width": "100%", "height": f"{height}px", "border": "0"})


# Layout mirroring structure of simple_style_dashboard.py but using iframes
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H1(
                        "Movie Ratings Predictive Modeling Dashboard",
                        className="fw-semibold",
                        style={'margin': '0', 'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}
                    ),
                    html.P(
                        "Explore models, data insights, and summary for our analysis of movie ratings using cached visualizations.",
                        style={'fontFamily': 'Inter, system-ui', 'fontSize': '16px', 'color': COLORS['text_secondary'], 'marginTop': '8px', 'marginBottom': '0'}
                    )
                ], className="card-header", style={"backgroundColor": COLORS['card_background_color']}),
            ], className="card")
        ], className="col-12")
    ], className="row mb-4"),

    html.Div([
        html.Div([
            dcc.Tabs(id='main-tabs', value='summary-tab', children=[
                dcc.Tab(label='Summary', value='summary-tab', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Models', value='models-tab', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Data', value='data-tab', className='custom-tab', selected_className='custom-tab--selected'),
            ], className='custom-tabs')
        ], className="col-12")
    ], className="row mb-3"),

    html.Div([
        # Summary content
        html.Div(id='summary-content', children=[
            html.H3("Project Summary", className="text-center mb-4", style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}),
            html.Div([
                html.Div([
                    html.Div("About This Dashboard", className="card-header fw-semibold"),
                    html.Div([
                        html.P(
                            "This dashboard uses precomputed HTML graphs to drastically reduce compute while keeping the visual style intact.",
                            className="mb-0", style={'fontFamily': 'Inter, system-ui', 'fontSize': '16px', 'color': COLORS['text_secondary']}
                        )
                    ], className="card-body")
                ], className="card"),
            ], className="mb-4"),

            # Copied markdown cards from simple_style_dashboard.py
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

        # Models content (iframes for each section)
        html.Div(id='models-content', children=[
            html.H3("Models", className="text-center mb-4", style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']}),

            html.Div([
                html.Div([
                    dcc.Tabs(id='model-tabs', value='linear', children=[
                        dcc.Tab(label='Linear Regression', value='linear', children=[
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("Linear Regression Analysis", className="card-header fw-semibold"),
                                        html.Div([
                                            graph_iframe('linear_ols_table.html', height=600)
                                        ], className="card-body p-2")
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div(
                                            "Linear Regression Analysis with Categorical Features",
                                            className="card-header fw-semibold"
                                        ),
                                        html.Div([
                                            html.P(
                                                "Comprehensive linear regression incorporating numeric predictors and categorical features (genres, release years, languages, and actor popularity). This model reveals how multiple factors combine to explain movie ratings, showing strong positive effects from engagement metrics and important contributions from categorical structure.",
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
                                            ])
                                        ], className="card-body")
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4")
                        ]),
                        dcc.Tab(label='KNN Regression', value='knn', children=[
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("KNN: Permutation Importance", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('knn_importance.html')], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-4 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("KNN Validation Curves", className="card-header fw-semibold"),
                                        html.Div([
                                            graph_iframe('knn_rmse_curve.html'),
                                            graph_iframe('knn_rmse_tuned.html')
                            ], className="row g-3"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div(
                                            "MLP Performance Summary",
                                            className="card-header fw-semibold"
                                        ),
                                        html.Div([
                                            html.P(
                                                "The tuned MLP achieves the strongest predictive performance by learning smooth nonlinear relationships between engagement metrics and ratings.",
                                                className="mb-3"
                                            ),
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
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-4 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("KNN Performance Summary", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('knn_summary.html', height=240)], className="card-body")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-4 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("Feature Correlation with KNN Predictions", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('knn_feature_corr.html')], className="card-body p-2")
                                    ], className="card")
                                ], className="col-12 mb-3"),
                            ], className="row g-3")
                        ]),
                        dcc.Tab(label='K-Means Clustering', value='kmeans', children=[
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("K-Means Clustering Analysis", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('kmeans_pca_2d.html', height=420)], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("Clustering Tuning Diagnostics", className="card-header fw-semibold"),
                                        html.Div([
                                            graph_iframe('kmeans_silhouette.html', height=340),
                                            graph_iframe('kmeans_inertia.html', height=340)
                                        ], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("3D PCA View of Clusters", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('kmeans_pca_3d.html', height=440)], className="card-body p-2")
                                    ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
                                ], className="col-12 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("Cluster Profiles", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                                        html.Div([graph_iframe('kmeans_cluster_profiles.html')], className="card-body p-2")
                                    ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
                                ], className="col-12 col-xl-6 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("Cluster Outcome Distribution", className="card-header fw-semibold", style={"backgroundColor": COLORS['card_background_color']}),
                                        html.Div([graph_iframe('kmeans_vote_box.html')], className="card-body p-2")
                                    ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
                                ], className="col-12 col-xl-6 mb-3"),
                            ], className="row g-3")
                        ]),
                        dcc.Tab(label='PCA', value='pca', children=[
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("PCA Summary", className="card-header fw-semibold"),
                                        html.Div([
                                            graph_iframe('pca_summary.html'),
                                            graph_iframe('pca_explained_variance.html', height=340)
                                        ], className="card-body p-2")
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("PCA Scatter (2D)", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('pca_scatter_2d.html', height=400)], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("PCA Scatter (3D)", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('pca_scatter_3d.html', height=420)], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                            ], className="row g-3"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("PCA Loadings", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('pca_loadings.html', height=380)], className="card-body p-2")
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4"),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div(
                                            "PCA Interpretation",
                                            className="card-header fw-semibold"
                                        ),
                                        html.Div([
                                            html.P(
                                                "Principal Component Analysis compresses the numeric features into orthogonal components that capture most variance with fewer dimensions. The first component is largely an engagement-scale axis (vote/user rating counts), followed by budget/revenue scale and runtime effects.",
                                                className="mb-3"
                                            ),
                                            html.Div([
                                                html.Strong("Highlights:"),
                                                html.Ul([
                                                    html.Li("Variance is concentrated in the first few components (see Scree and Cumulative plots)"),
                                                    html.Li("Loadings identify which original features drive each component"),
                                                    html.Li("2D/3D projections show smooth gradients in `vote_average` across PCA space"),
                                                    html.Li("PCA space facilitates clustering (K-Means) and visualization without heavy overplotting"),
                                                ])
                                            ])
                                        ], className="card-body")
                                    ], className="card")
                                ], className="col-12")
                            ], className="row mb-4")
                        ]),
                        dcc.Tab(label='MLP Regression', value='mlp', children=[
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Div("MLP: Permutation Importance", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('mlp_importance.html')], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div("MLP Training Loss Curve", className="card-header fw-semibold"),
                                        html.Div([graph_iframe('mlp_loss_curve.html')], className="card-body p-2")
                                    ], className="card h-100")
                                ], className="col-12 col-xl-6 mb-3"),
                            ], className="row g-3")
                        ]),
                    ])
                ], className="col-12")
            ], className="row mb-4"),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Model Performance Comparison", className="card-header fw-semibold"),
                        html.Div([graph_iframe('comparison_table.html', height=420)], className="card-body p-3")
                    ], className="card")
                ], className="col-12")
            ], className="row mb-4"),

            # Copied "Key Conclusions" markdown from simple_style_dashboard.py
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(
                            "Key Conclusions",
                            className="card-header fw-semibold"
                        ),
                        html.Div([
                            html.H5("Model Performance Rankings:", className="fw-bold mt-3"),
                            html.Ol([
                                html.Li([html.Strong("MLP (Best):"), " Validation RMSE ~0.88 - Captures nonlinear relationships effectively"]),
                                html.Li([html.Strong("KNN (Strong):"), " Validation RMSE will update - Excellent local neighborhood exploitation"]),
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
            ], className="row mb-4"),
        ], style={'display': 'none'}),

        # Data content placeholder
        html.Div(id='data-content', children=[
            html.Div([
                html.Div([
                    html.Div("Data Tab (Cached)", className="card-header fw-semibold"),
                    html.Div([
                        html.P("This lightweight dashboard focuses on model visuals. Data tab can be added with cached plots if needed.")
                    ], className="card-body")
                ], className="card")
            ], className="row mb-4")
        ], style={'display': 'none'})
    ], className="tab-content-container"),
], id='app-container', className="container-fluid py-4 px-3", style={'background': COLORS['bg_main'], 'minHeight': '100vh', 'fontFamily': 'Inter, system-ui'})

# Tab switching (mirror behavior without heavy callbacks)
from dash.dependencies import Input, Output

@app.callback(
    [Output('summary-content', 'style'), Output('models-content', 'style'), Output('data-content', 'style')],
    Input('main-tabs', 'value')
)
def switch_tabs(tab):
    if tab == 'summary-tab':
        return ({'display': 'block'}, {'display': 'none'}, {'display': 'none'})
    if tab == 'models-tab':
        return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'})
    return ({'display': 'none'}, {'display': 'none'}, {'display': 'block'})

# Inject the same custom CSS for tabs as the original
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
            .custom-tabs { display: flex; position: relative; background: transparent; border-radius: 10px 10px 0 0; padding: 0; border: none; box-shadow: none; }
            .custom-tab { margin: 12px 4px 0 4px; padding: 8px 20px; border: none; border-bottom: 0.5pt solid #ddd; color: #333333; background: #f8f9fa; box-shadow: 1px 1px 2px 1px rgba(0,0,0,0.05); text-transform: uppercase; font-weight: 300; font-size: 11px; border-radius: 7px 7px 0 0; border-top: 1px solid #e0e0e0; letter-spacing: 0.167em; font-family: 'Manrope', 'Inter', sans-serif; cursor: pointer; transition: all 0.3s ease; position: relative; z-index: 1; }
            .custom-tab:hover { color: #000; background: #e9ecef; }
            .custom-tab--selected { color: #000 !important; font-weight: 600 !important; position: relative; z-index: 2; background: #fff !important; }
            .custom-tabs::after { content: ''; position: absolute; bottom: 0; left: 0; width: 33.33%; height: 4.5px; border-radius: 0 0 1px 1px; background: linear-gradient(113deg, hsl(260deg 100% 64%) 0%, hsl(190deg 100% 55%) 100%); box-shadow: 0px 0px 8px 0px hsl(262deg 100% 70% / 70%); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); z-index: 3; }
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
                observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
            });
        </script>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
