import dash
from dash import html, dcc
import dash_table
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from networkx.algorithms.centrality import harmonic_centrality

# Load node and edge data from CSV files
nodes = pd.read_csv('InputFileNodes.csv')
# nodes = nodes.rename(columns={'NodeID': 'ID'})
edges = pd.read_csv('InputFileEdges.csv')

# to split each column of nodes and edges by comma *,*
separator = ','
node_columns = nodes.columns.str.split(separator, expand=True)
edge_columns = edges.columns.str.split(separator, expand=True)

G = nx.Graph()
nodes_array = nodes.to_dict(orient='records')
edges_array = edges.to_dict(orient='records')

i = 0
for node_attrs in nodes_array:
    G.add_node(i, **node_attrs)
    i = i+1

for edge_attrs in edges_array:
    G.add_edge(edge_attrs['source'], edge_attrs['target'], **edge_attrs)

pos = nx.spring_layout(G)

pr_scores = nx.pagerank(G)
print(pr_scores)
# Create edge and node traces
edge_trace = go.Scatter(
    x=(), y=(), line={'width': 0.5, 'color': 'red'}, hoverinfo='none', mode='lines')
node_trace = go.Scatter(x=(), y=(), text=(), mode='markers+text',
                        hoverinfo='text', marker={'size': 10, 'color': '#888'})

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple([str(node)])

# Create initial figure
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=True, hovermode='closest', margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, xaxis={
                'showgrid': False, 'zeroline': False, 'showticklabels': False}, yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}))

# Create a Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Button('Remove edge with largest betweenness',
                    className='btn btn-danger', id='remove-button', n_clicks=0),
        html.Div(id='output-box'),
        html.Button('Show betweenness centrality',
                    className='btn btn-danger', id='bc-button', n_clicks=0),
        
        html.Button('Show closeness centrality',
                    className='btn btn-danger', id='cc-button', n_clicks=0),
        html.Button('Show harmonic closeness centrality',
                    className='btn btn-danger', id='hc-button', n_clicks=0),
        html.Div(id='bc-table'),
        html.Div(id='cc-table'),
        html.Div(id='hc-table')
    ], className=''),
    html.Div([
        dcc.Graph(id='network-graph', figure=fig)
    ], className=''),
])


@app.callback(
    Output('network-graph', 'figure'),
    Output('output-box', 'children'),
    Input('remove-button', 'n_clicks'),
    State('network-graph', 'figure'),
    prevent_initial_call=True
)
def remove_edge(n_clicks, figure):
    # Get edges with highest betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    max_betweenness = max(edge_betweenness.values())
    max_edges = [edge for edge, centrality in edge_betweenness.items()
                 if centrality == max_betweenness]
    # Remove edge with highest betweenness centrality
    if len(max_edges) > 0:
        max_edge = max_edges[0]
        G.remove_edge(*max_edge)
        pos = nx.spring_layout(G)
        updated_edge_trace = go.Scatter(
            x=[], y=[], line={'width': 0.5, 'color': 'red'}, hoverinfo='none', mode='lines')
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            updated_edge_trace['x'] += tuple([x0, x1, None])
            updated_edge_trace['y'] += tuple([y0, y1, None])
        updated_node_trace = go.Scatter(x=[], y=[], text=[
        ], mode='markers+text', hoverinfo='text', marker={'size': 10, 'color': '#888'})
        for node in G.nodes():
            x, y = pos[node]
            updated_node_trace['x'] += tuple([x])
            updated_node_trace['y'] += tuple([y])
            updated_node_trace['text'] += tuple([str(node)])
        # Create updated figure
        updated_fig = go.Figure(data=[updated_edge_trace, updated_node_trace], layout=go.Layout(showlegend=True, hovermode='closest', margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, xaxis={
                                'showgrid': False, 'zeroline': False, 'showticklabels': False}, yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}))
        output_message = f"Removed edge: {max_edge}"
        return updated_fig, output_message
    else:
        output_message = "Cannot remove any more edges"
        return figure, output_message


@app.callback(
    Output('bc-table', 'children'),
    Output('cc-table', 'children'),
    Output('hc-table', 'children'),
    Input('hc-button', 'n_clicks'),
    Input('bc-button', 'n_clicks'),
    Input('cc-button', 'n_clicks'),
    State('bc-button', 'n_clicks'),
    State('cc-button', 'n_clicks'),
    State('hc-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_centrality_tables(bc_clicks, cc_clicks, hc_clicks,bc_n_clicks,cc_n_clicks,hc_n_clicks):
    # Calculate betweenness and closeness centrality
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = ''
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    hc = harmonic_centrality(G)
    show_bc_table = (bc_n_clicks or 0) % 2 == 1
    show_cc_table = (cc_n_clicks or 0) % 2 == 1
    show_hc_table = (hc_n_clicks or 0) % 2 == 1
    if button_id == 'bc-button':
        if show_bc_table:
            bc_df = pd.DataFrame.from_dict(bc, orient='index', columns=[
                                        'Betweenness Centrality'])
            bc_table = dash_table.DataTable(data=bc_df.to_dict('records'), columns=[
                                            {'name': i, 'id': i} for i in bc_df.columns])
        else:
            bc_table = dash.html.Div()
        return bc_table, dash.html.Div(), dash.html.Div()
    elif button_id == 'cc-button':
        if show_cc_table:
            cc_df = pd.DataFrame.from_dict(cc, orient='index', columns=['Closeness Centrality'])
            cc_table = dash_table.DataTable(data=cc_df.to_dict('records'), columns=[{'name': i, 'id': i} for i in cc_df.columns])
        else:
            cc_table = dash.html.Div()
        return cc_table, dash.html.Div(), dash.html.Div()
    elif button_id == 'hc-button':
        if show_hc_table:
            hc_df = pd.DataFrame.from_dict(hc, orient='index', columns=[
                                        'Harmonic Closeness Centrality'])
            hc_table = dash_table.DataTable(data=hc_df.to_dict('records'), columns=[
                                            {'name': i, 'id': i} for i in hc_df.columns])
        else:
            hc_table=dash.html.Div()
        return hc_table, dash.html.Div(), dash.html.Div()
    else:
        return dash.html.Div(), dash.html.Div(), dash.html.Div()

    # Create dataframes for centrality measures

    return bc_table, cc_table, hc_table


if __name__ == '__main__':
    app.run_server(debug=True)
