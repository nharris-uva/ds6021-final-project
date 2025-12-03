
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

COLORS = {
    'graph_bg': '#D9376E',

    # Background
    'bg_main': '#EFF0F3',
    'grid_color': "#8AAAC0",
    
    # Header colors
    'header_original': '#FF8E3C',
    'header_notes': '#FF8E3C',
    'header_transformed': '#FF8E3C',
    
    # Text colors
    'text_dark': '#1a1a1a',
    'text_primary': '#222222',
    'text_secondary': '#333333',
    'text_tertiary': '#444444',
    'text_light': '#ffffff',
    
    # Misc
    'bg_transparent': 'rgba(0,0,0,0)',
    'border_light': '2px solid #eee',
    # Cards
    'card_background_color': '#FFFFFF',
}

# Modern stylesheet (Bootstrap 5)
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Load data
raw_data = pd.read_csv("data/all_movie_features_15_to_19.csv").drop(columns=["Unnamed: 0"]).head(100)

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

# Initialize the Dash app with external stylesheets and ignore local assets
app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP_CSS, GOOGLE_FONTS],
    include_assets_files=False
)

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
    fig = px.histogram(df, x=col, marginal="box", opacity=0.9, color_discrete_sequence=[COLORS['graph_bg']])
    fig.update_layout(
        title_text=None,
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        showlegend=False,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14),
        xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12), automargin=True),
        yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12), automargin=True)
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
    fig.update_layout(
        title_text=None,
        xaxis_title=col,
        yaxis_title='Count',
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        showlegend=False,
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=14),
        xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12), automargin=True),
        yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12), automargin=True)
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
        font=dict(size=14, family="Inter, system-ui", color=COLORS['text_primary'])
    )
    fig.update_layout(
        height=420,
        margin=DEFAULT_FIG_MARGIN,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color=COLORS['text_primary'], size=14)
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
                html.Div(
                    f"Original: {col}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div(
                    dcc.Graph(
                        figure=left_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-lg-4 mb-3"),
        
        # Middle card - Transformation notes
        html.Div([
            html.Div([
                html.Div(
                    f"Transformation: {col}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
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
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12 col-lg-4 mb-3"),
        
        # Right card - Transformed data
        html.Div([
            html.Div([
                html.Div(
                    f"Transformed: {col}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div(
                    dcc.Graph(
                        figure=right_graph,
                        config={'displayModeBar': False}
                    ),
                    className="card-body p-2"
                )
            ], className="card h-100", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
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
                            'color': COLORS['text_secondary']
                        }
                    )
                ], className="card-body")
            ], className="card", style={"backgroundColor": "rgba(0,0,0,0)", "boxShadow": "none"})
        ], className="col-12")
    ], className="row mb-4"),

    # Column headers
    html.Div([
        html.Div([
            html.Div(
                "Raw Data",
                className="text-center fw-bold",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_original'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformation Notes",
                className="text-center fw-bold",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_notes'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformed Data",
                className="text-center fw-bold",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header_transformed'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4")
    ], className="row g-3 mb-3 pb-2", style={'borderBottom': COLORS['border_light']}),

    # All column rows
    html.Div(build_dashboard_layout(), className="dashboard-rows")
], id='app-container', className="container-fluid py-4 px-3", style={
    'background': COLORS['bg_main'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui'
})


if __name__ == '__main__':
    app.run(debug=True, port=8050, host='127.0.0.1')
