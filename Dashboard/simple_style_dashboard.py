
import pandas as pd
import dash
from dash import dcc, html, dash_table, no_update
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
    include_assets_files=False,
    suppress_callback_exceptions=True
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
        xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)),
        yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12))
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
        xaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12)),
        yaxis=dict(title_font=dict(color=COLORS['text_primary'], size=16), tickfont=dict(color=COLORS['text_primary'], size=12))
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

def build_data_tab_content():
    """Build the entire Data tab content"""
    # Get list of columns excluding 'id'
    columns = [col for col in original_data.columns if col != 'id']
    
    return html.Div([
        # Column selector dropdown
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
                    style={
                        'width': '300px',
                        'fontFamily': 'Inter, system-ui'
                    }
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'gap': '10px',
                'marginBottom': '20px'
            })
        ], className="row mb-3"),

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

        # Dynamic column visualization container
        html.Div(id='column-visualization', className="dashboard-rows"),
        
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
            html.H3("Summary", className="text-center mb-4", 
                   style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']})
        ], style={'display': 'block'}),
        
        html.Div(id='models-content', children=[
            html.H3("Models", className="text-center mb-4",
                   style={'fontFamily': 'Inter, system-ui', 'color': COLORS['header']})
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
    
    # Build and return only the selected column's visualization
    return build_column_row(selected_column)
        
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
