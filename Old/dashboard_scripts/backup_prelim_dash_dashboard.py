
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

COLORS = {
    # Colors derived from dashboard background gradients
    'yellow_bg': '#e6d5a8',
    'blue_bg': '#a8d5f0',
    'gold_bg': '#ffe8a3',
    'header_original': '#1a4d8f',
    'header_notes': '#8b6f00',
    'header_transformed': '#156647',
    'header_text': '#333333',
    'body_text': '#333333',
    'body_strong': '#1a1a1a',
    'body_secondary': '#444444',
    'plot_font': '#222222',
    'plot_bg': 'rgba(0,0,0,0)',
    'plot_bar_text': '#ffffff',
    'dashboard_bg': 'linear-gradient(#444cf722 1px, transparent 1px), linear-gradient(to right, #444cf722 1px, transparent 1px), radial-gradient(1400px 1000px at 25% 25%, {yellow_bg_alpha}, rgba(230, 213, 168, 0) 65%), radial-gradient(1200px 900px at 80% 30%, {blue_bg_alpha}, rgba(168, 213, 240, 0) 60%), radial-gradient(1100px 1100px at 50% 80%, {gold_bg_alpha}, rgba(255, 232, 163, 0) 55%), linear-gradient(135deg, #f7f4ea, #fbf7ef)',
    'dashboard_border': '2px solid #eee',
}

# Modern stylesheet (Bootstrap 5)
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Load data
raw_data = pd.read_csv("data/all_movie_features_15_to_19.csv").drop(columns=["Unnamed: 0"]).head(1000)

metadata = raw_data[["id", "title", "adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "video"]]
original_data = raw_data.drop(columns=["adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "title", "original_title", "video", "homepage", "production_companies", "production_countries", "status", "spoken_languages"] + [col for col in raw_data.columns if "belongs_to_collection" in col], axis=1)
transformed_data = original_data.copy()

#Transformations:
#Transform budget
transformed_data['budget'] = transformed_data['budget'].replace(0, np.nan)
transformed_data['budget'] = np.log1p(transformed_data['budget'])
#Transform revenue
transformed_data['revenue'] = transformed_data['revenue'].replace(0, np.nan)
transformed_data['revenue'] = np.log1p(transformed_data['revenue'])
#Transform runtime
transformed_data['runtime'] = transformed_data['runtime'].replace(0, np.nan)
#Transform release_date
transformed_data['release_date'] = pd.to_datetime(transformed_data['release_date'], errors='coerce')
#transform vote_count
transformed_data['vote_count'] = transformed_data['vote_count'].replace(0, np.nan)
transformed_data['vote_count'] = np.log1p(transformed_data['vote_count'])

#transform vote_average
transformed_data['vote_average'] = transformed_data['vote_average'].replace(0, np.nan)

# Initialize the Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS])


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
        plot_bgcolor=COLORS['plot_bg'],
        paper_bgcolor=COLORS['plot_bg'],
        font=dict(color=COLORS['plot_font'], size=14),
        xaxis=dict(title_font=dict(color=COLORS['plot_font'], size=16), tickfont=dict(color=COLORS['plot_font'], size=12), automargin=True),
        yaxis=dict(title_font=dict(color=COLORS['plot_font'], size=16), tickfont=dict(color=COLORS['plot_font'], size=12), automargin=True)
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
        opacity=0.95,
    )
    # Keep bar labels inside to avoid clipping at the top
    fig.update_traces(
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(size=12, color=COLORS['plot_bar_text'])
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
        plot_bgcolor=COLORS['plot_bg'],
        paper_bgcolor=COLORS['plot_bg'],
        font=dict(color=COLORS['plot_font'], size=14),
        xaxis=dict(title_font=dict(color=COLORS['plot_font'], size=16), tickfont=dict(color=COLORS['plot_font'], size=12), automargin=True),
        yaxis=dict(title_font=dict(color=COLORS['plot_font'], size=16), tickfont=dict(color=COLORS['plot_font'], size=12), automargin=True)
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
        font=dict(size=14, family="Inter, system-ui", color=COLORS['plot_font'])
    )
    fig.update_layout(
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color=COLORS['plot_font'], size=14)
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


# Create the app layout (light mode only, no theme toggle)
dashboard_bg = COLORS['dashboard_bg'].format(
    yellow_bg_alpha=hex_to_rgba(COLORS['yellow_bg'], 0.35),
    blue_bg_alpha=hex_to_rgba(COLORS['blue_bg'], 0.30),
    gold_bg_alpha=hex_to_rgba(COLORS['gold_bg'], 0.28)
)

app.layout = html.Div([
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
                            'color': COLORS['header_text']
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
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_original']}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformation Notes",
                className="text-center fw-bold fs-5",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_notes']}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformed Data",
                className="text-center fw-bold fs-5",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_transformed']}
            )
        ], className="col-12 col-lg-4")
    ], className="row g-3 mb-3 pb-2", style={'borderBottom': COLORS['dashboard_border']}),

    # All column rows
    html.Div(build_dashboard_layout(), className="dashboard-rows")
], id='app-container', className="container-fluid py-4 px-3", style={
    'background': dashboard_bg,
    'backgroundSize': '50px 50px, 50px 50px, auto, auto, auto, auto',
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui'
})


if __name__ == '__main__':
    app.run(debug=True, port=8050, host='127.0.0.1')