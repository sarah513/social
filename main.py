
import dash
from dash import html, dcc
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

# Load node and edge data from CSV files
nodes = pd.read_csv('nodes.csv')
# nodes = nodes.rename(columns={'NodeID': 'ID'})
edges = pd.read_csv('edges.csv')

# to split each column of nodes and edges by comma *,*
separator = ','
node_columns = nodes.columns.str.split(separator, expand=True)
edge_columns = edges.columns.str.split(separator, expand=True)

G = nx.Graph()
nodes_array = nodes.to_dict(orient='records')
edges_array = edges.to_dict(orient='records')

i=0
for  node_attrs in  nodes_array:
    G.add_node(i, **node_attrs)
    i=i+1

for edge_attrs in edges_array:
    G.add_edge(edge_attrs['source'],edge_attrs['target'],**edge_attrs)

pos = nx.spring_layout(G)

# Create edge and node traces
edge_trace = go.Scatter(x=(), y=(), line={'width': 0.5, 'color': 'red'}, hoverinfo='none', mode='lines')
node_trace = go.Scatter(x=(), y=(), text=(), mode='markers+text', hoverinfo='text', marker={'size': 10, 'color': '#888'})

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
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=True, hovermode='closest', margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}, yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}))

# Calculate betweenness centrality for each edge
edge_betweenness = nx.edge_betweenness_centrality(G)

# Get edges with highest betweenness centrality
max_betweenness = max(edge_betweenness.values())
max_edges = [edge for edge, centrality in edge_betweenness.items() if centrality == max_betweenness]

# Create a list of edge traces for edges with highest betweenness centrality
max_edge_traces = []
for edge in max_edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    max_edge_trace = go.Scatter(x=[x0, x1], y=[y0, y1], line={'width': 5, 'color': 'orange'}, hoverinfo='text',
                                text='Edge betweenness: {:.4f}'.format(max_betweenness))
    max_edge_traces.append(max_edge_trace)

# Add edge traces to figure
fig.add_traces(max_edge_traces)

# Create a Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='network-graph', figure=fig),
    html.Br(),
    html.Button('Remove edge with largest betweenness', id='remove-button', n_clicks=0),
    html.Div(id='output-box')
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
    max_edges = [edge for edge, centrality in edge_betweenness.items() if centrality == max_betweenness]
    # Remove edge with highest betweenness centrality
    if len(max_edges) > 0:
        max_edge = max_edges[0]
        G.remove_edge(*max_edge)
        pos = nx.spring_layout(G)
        updated_edge_trace = go.Scatter(x=[], y=[], line={'width': 0.5, 'color': 'red'}, hoverinfo='none', mode='lines')
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            updated_edge_trace['x'] += tuple([x0, x1, None])
            updated_edge_trace['y'] += tuple([y0, y1, None])
        updated_fig = go.Figure(data=[updated_edge_trace, node_trace], layout=go.Layout(showlegend=True, hovermode='closest', margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}, yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}))
        return updated_fig, f'Removed edge between nodes {max_edge[0]} and {max_edge[1]} with betweenness centrality of {max_betweenness:.4f}'
    else:
        return figure, 'All edges have been removed'

if __name__ == '__main__':
    app.run_server(debug=True)