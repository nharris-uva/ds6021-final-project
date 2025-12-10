
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
# metadata = raw_data[["id", "title", "adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "video"]]
# original_data = raw_data.drop(columns=["adult", "backdrop_path", "poster_path", "imdb_id", "overview", "tagline", "title", "original_title", "video", "homepage", "production_companies", "production_countries", "status", "spoken_languages"] + [col for col in raw_data.columns if "belongs_to_collection" in col], axis=1)
# transformed_data = original_data.copy()
original_data = raw_data.copy()
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
DEFAULT_FIG_MARGIN = dict(l=10, r=10, t=10, b=20)

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
        height=140,
        margin=dict(l=10, r=10, t=30, b=10),
        barmode="stack",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font=dict(color=COLORS['text_primary'], size=12),
        xaxis=dict(
            title=None,
            tickfont=dict(size=11),
            automargin=True,
        ),
        yaxis=dict(
            title="Percentage %",
            range=[0, 100],
            tick0=0,
            dtick=20,
            tickfont=dict(size=11),
            automargin=True,
        ),
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
        
        # Middle card - Transformation notes + missingness
        html.Div([
            html.Div([
                html.Div(
                    f"Transformation: {col}",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=create_missingness_bar(col),
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
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformation Notes",
                className="text-center fw-bold",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4"),
        html.Div([
            html.Div(
                "Transformed Data",
                className="text-center fw-bold",
                style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header'], 'fontSize': '42px'}
            )
        ], className="col-12 col-lg-4")
    ], className="row g-3 mb-3 pb-2", style={'borderBottom': COLORS['border_light']}),

    # All column rows
    html.Div(build_dashboard_layout(), className="dashboard-rows"),
    
    # Data table section at the bottom
    html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "Original Dataset",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": col, "id": col} for col in original_data.columns],
                        data=original_data.to_dict('records'),
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
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row mt-5 mb-4"),
    
    # Correlation matrix section
    html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "Correlation Matrix (Transformed Data)",
                    className="card-header fw-semibold",
                    style={"backgroundColor": COLORS['card_background_color']}
                ),
                html.Div([
                    dcc.Graph(
                        id='correlation-matrix',
                        figure=create_correlation_matrix(transformed_data),
                        config={'displayModeBar': True}
                    )
                ], className="card-body p-3")
            ], className="card", style={"backgroundColor": COLORS['card_background_color'], "boxShadow": "none"})
        ], className="col-12")
    ], className="row mt-4 mb-4")
], id='app-container', className="container-fluid py-4 px-3", style={
    'background': COLORS['bg_main'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui'
})

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='127.0.0.1')
