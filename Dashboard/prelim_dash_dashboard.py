import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Modern stylesheet (Bootstrap 5)
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Load data
raw_data = pd.read_csv("data/all_movie_features_15_to_19.csv").drop(columns=["Unnamed: 0"])

metadata = raw_data[["id", "title", "adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "video"]]
original_data = raw_data.drop(columns=["adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "title", "original_title", "video", "homepage", "production_companies", "production_countries", "status", "spoken_languages"] + [col for col in raw_data.columns if "belongs_to_collection" in col], axis=1)
transformed_data = original_data.copy()

# Initialize the Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS])

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --hue-primary: 223;
                --hue-secondary: 178;
                --primary500: hsl(var(--hue-primary), 90%, 50%);
                --primary600: hsl(var(--hue-primary), 90%, 60%);
                --primary700: hsl(var(--hue-primary), 90%, 70%);
                --secondary800: hsl(var(--hue-secondary), 90%, 80%);
                --white0: hsla(0, 0%, 100%, 0);
                --white50: hsla(0, 0%, 100%, 0.05);
                --white100: hsla(0, 0%, 100%, 0.1);
                --white200: hsla(0, 0%, 100%, 0.2);
                --white300: hsla(0, 0%, 100%, 0.3);
                --white500: hsla(0, 0%, 100%, 0.5);
                --white: hsl(0, 0%, 100%);
                /* Theme-aware card header colors */
                --card-header-bg-light: #f5f5f7;
                --card-header-text-light: #1f2937;
                --card-header-border-light: #e5e7eb;
                --card-header-bg-dark: #0f172a;
                --card-header-text-dark: #e5e7eb;
                --card-header-border-dark: #1f2937;
            }

            .glass-card {
                backdrop-filter: blur(12px) !important;
                -webkit-backdrop-filter: blur(12px) !important;
                background: #ffffff1a !important;
                border: 1px solid #ffffff40 !important;
                border-radius: 1em !important;
                position: relative !important;
                overflow: visible !important;
                box-shadow: 0 8px 32px 0 #00000020 !important;
            }

            .glass-card::before {
                content: "" !important;
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                border-radius: 1em !important;
                padding: 1px !important;
                background: linear-gradient(135deg, #ffffff80, #ffffff00) !important;
                -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
                -webkit-mask-composite: xor !important;
                mask-composite: exclude !important;
                pointer-events: none !important;
                z-index: 0 !important;
            }

            .glass-card::after {
                content: "" !important;
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                border-radius: 1em !important;
                padding: 1px !important;
                background: linear-gradient(135deg, #ffffff00, #ffffff40) !important;
                -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
                -webkit-mask-composite: xor !important;
                mask-composite: exclude !important;
                pointer-events: none !important;
                z-index: 0 !important;
            }

            .glass-card .card-header {
                background: var(--card-header-bg-light) !important;
                color: var(--card-header-text-light) !important;
                font-weight: 600 !important;
                position: relative !important;
                z-index: 2 !important;
                border-top-left-radius: 1em !important;
                border-top-right-radius: 1em !important;
                border-bottom: 1px solid var(--card-header-border-light) !important;
                padding: 0.75rem 1rem !important;
            }

            .glass-card .card-body {
                position: relative !important;
                z-index: 2 !important;
                color: #333333 !important;
                background: transparent !important;
            }

            .glass-card .card-body * {
                color: #333333 !important;
            }

            .glass-card .card-body h1,
            .glass-card .card-body h2,
            .glass-card .card-body h3,
            .glass-card .card-body h4,
            .glass-card .card-body h5,
            .glass-card .card-body h6,
            .glass-card .card-body strong,
            .glass-card .card-body b {
                color: #1a1a1a !important;
                font-weight: 600 !important;
            }

            .glass-card .card-body p,
            .glass-card .card-body li {
                color: #444444 !important;
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
    </body>
</html>
'''

# Plotly visual style
px.defaults.template = "plotly_dark"
DEFAULT_FIG_MARGIN = dict(l=10, r=10, t=80, b=20)

# Dictionary to store transformation notes for each column
TRANSFORMATION_NOTES = {
    # Add your transformation notes here
    # Example: "budget": "Applied log transformation to reduce skewness\nReplaced 0 values with NaN"
}

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
    fig = px.histogram(df, x=col, marginal="box", opacity=0.9)
    fig.update_layout(
        title_text=None,
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#222222", size=14),
        xaxis=dict(title_font=dict(color="#222222", size=16), tickfont=dict(color="#222222", size=12), automargin=True),
        yaxis=dict(title_font=dict(color="#222222", size=16), tickfont=dict(color="#222222", size=12), automargin=True)
    )
    return fig

def create_categorical_graph(df, col, title_prefix=""):
    """Create bar chart for categorical columns showing top 15 most frequent occurrences"""
    value_counts = df[col].value_counts().head(15)
    original_labels = list(value_counts.index)
    x_labels = [truncate_label(lbl, max_len=24) for lbl in original_labels]
    fig = px.bar(
        x=x_labels,
        y=value_counts.values,
        text_auto=True,
        opacity=0.95
    )
    # Keep bar labels inside to avoid clipping at the top
    fig.update_traces(
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=12, color='#ffffff')
    )
    fig.update_traces(
        hovertext=original_labels,
        hovertemplate="%{hovertext}: %{y}<extra></extra>"
    )
    fig.update_layout(
        title_text=None,
        xaxis_title=col,
        yaxis_title='Count',
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#222222", size=14),
        xaxis=dict(title_font=dict(color="#222222", size=16), tickfont=dict(color="#222222", size=12), automargin=True),
        yaxis=dict(title_font=dict(color="#222222", size=16), tickfont=dict(color="#222222", size=12), automargin=True)
    )
    fig.update_xaxes(tickangle=-30)
    return fig

def create_text_placeholder(col):
    """Create placeholder for text columns that can't be visualized"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Column '{col}' has too many unique values to visualize",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, family="Inter, system-ui", color="#222222")
    )
    fig.update_layout(
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color="#222222", size=14)
    )
    return fig

def build_column_row(col):
    """Build a single row for one column with original, notes, and transformed cards"""
    col_type = get_column_type(original_data, col)
    
    # Skip id column
    if col == 'id':
        return None
    
    # Create appropriate visualizations based on column type
    if col_type == 'numeric':
        left_graph = create_numeric_graph(original_data, col)
        right_graph = create_numeric_graph(transformed_data, col)
    else:  # categorical
        left_graph = create_categorical_graph(original_data, col)
        right_graph = create_categorical_graph(transformed_data, col)
    
    # Get transformation notes if available
    notes = TRANSFORMATION_NOTES.get(col, f"No transformation notes yet for **{col}**.\n\nAdd your transformation explanation here.")
    
    row = html.Div([
        # Left card - Original data
        html.Div([
            html.Div([
                html.Div(f"Original: {col}", className="card-header fw-semibold"),
                html.Div(
                    dcc.Graph(
                        figure=left_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card glass-card shadow-lg h-100")
        ], className="col-12 col-lg-4 mb-3"),
        
        # Middle card - Transformation notes with glassmorphism
        html.Div([
            html.Div([
                html.Div(f"Transformation: {col}", className="card-header fw-semibold"),
                html.Div(
                    dcc.Markdown(
                        children=notes,
                        className="small"
                    ),
                    className="card-body",
                    style={
                        "height": "280px",
                        "overflowY": "auto",
                        "whiteSpace": "pre-wrap",
                        "fontSize": "14px"
                    }
                )
            ], className="card glass-card shadow-lg h-100")
        ], className="col-12 col-lg-4 mb-3"),
        
        # Right card - Transformed data
        html.Div([
            html.Div([
                html.Div(f"Transformed: {col}", className="card-header fw-semibold"),
                html.Div(
                    dcc.Graph(
                        figure=right_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card glass-card shadow-lg h-100")
        ], className="col-12 col-lg-4 mb-3")
    ], className="row g-3 align-items-stretch mb-3")
    
    return row

def build_dashboard_layout():
    """Build all rows for all columns"""
    rows = []
    for col in original_data.columns:
        row = build_column_row(col)
        if row is not None:
            rows.append(row)
    return rows

# Create the app layout
app.layout = html.Div([
    # Theme toggle controls
    html.Div([
        html.Div([
            html.Label("Theme", className="fw-semibold me-3", style={'color': '#ffffffe6'}),
            dcc.RadioItems(
                id='theme-toggle',
                options=[
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'}
                ],
                value='light',
                inline=True,
                inputStyle={'marginRight': '6px'},
                labelStyle={'marginRight': '14px', 'color': '#ffffffe6'}
            )
        ], className="col")
    ], className="row mb-3"),
    # Top header card
    html.Div([
        html.Div([
            html.Div([
                html.Div("Data Transformation Dashboard", className="card-header fw-semibold"),
                html.Div([
                    html.P(
                        "Compare original vs transformed features and document your transformation steps.",
                        className="mb-0",
                        style={
                            'fontFamily': 'Inter, system-ui',
                            'fontSize': '16px',
                            'color': '#333333'
                        }
                    )
                ], className="card-body")
            ], className="card glass-card shadow-lg")
        ], className="col-12")
    ], className="row mb-4"),
    
    # Column headers
    html.Div([
        html.Div([
            html.Div(
                "Original Data",
                className="text-center fw-bold fs-5",
                id={'type': 'header-color', 'index': 0},
                style={'fontFamily': 'Inter, system-ui', 'color': '#60a5fa'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformation Notes",
                className="text-center fw-bold fs-5",
                id={'type': 'header-color', 'index': 1},
                style={'fontFamily': 'Inter, system-ui', 'color': '#fbbf24'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformed Data",
                className="text-center fw-bold fs-5",
                id={'type': 'header-color', 'index': 2},
                style={'fontFamily': 'Inter, system-ui', 'color': '#34d399'}
            )
        ], className="col-12 col-lg-4")
    ], className="row g-3 mb-3 pb-2", style={'borderBottom': '2px solid #ffffff33'}),
    
    # All column rows
    html.Div(build_dashboard_layout(), className="dashboard-rows")
    
], id='app-container', className="container-fluid py-4 px-3")

# Theme callback
@app.callback(
    Output('app-container', 'style'),
    Output({'type': 'header-color', 'index': 0}, 'style'),
    Output({'type': 'header-color', 'index': 1}, 'style'),
    Output({'type': 'header-color', 'index': 2}, 'style'),
    Input('theme-toggle', 'value')
)
def update_theme(theme):
    if theme == 'dark':
        background = (
            'radial-gradient(1600px 1200px at 50% 30%, hsla(223, 90%, 18%, 0.35), hsla(223, 90%, 8%, 0.6) 60%), '
            'linear-gradient(135deg, hsla(223, 90%, 12%, 0.8), hsla(223, 90%, 6%, 0.9))'
        )
        header_styles = [
            {'color': '#60a5fa'},
            {'color': '#fbbf24'},
            {'color': '#34d399'}
        ]
    else:
        background = (
            'linear-gradient(#444cf722 1px, transparent 1px), '
            'linear-gradient(to right, #444cf722 1px, transparent 1px), '
            'radial-gradient(1400px 1000px at 25% 25%, rgba(230, 213, 168, 0.35), rgba(230, 213, 168, 0) 65%), '
            'radial-gradient(1200px 900px at 80% 30%, rgba(168, 213, 240, 0.30), rgba(168, 213, 240, 0) 60%), '
            'radial-gradient(1100px 1100px at 50% 80%, rgba(255, 232, 163, 0.28), rgba(255, 232, 163, 0) 55%), '
            'linear-gradient(135deg, #f7f4ea, #fbf7ef)'
        )
        header_styles = [
            {'color': '#1a4d8f'},
            {'color': '#8b6f00'},
            {'color': '#156647'}
        ]

    container_style = {
        'background': background,
        'backgroundSize': '50px 50px, 50px 50px, auto, auto, auto, auto',
        'minHeight': '100vh',
        'fontFamily': 'Inter, system-ui'
    }
    return container_style, header_styles[0], header_styles[1], header_styles[2]

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='127.0.0.1')