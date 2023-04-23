import sys
import networkx as nx
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPen, QColor
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer 
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
from sklearn.metrics.cluster import normalized_mutual_info_score
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem
import matplotlib.colors as mcolors


class GraphWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def draw_graph(self, G):
        self.axes.clear()
        pos = nx.spring_layout(G)
        for node in G.nodes():
            G.nodes[node]['color'] = 'red'
        degrees = dict(nx.degree(G))
        node_sizes = [300 * degrees[node] for node in G.nodes()]
        nx.draw(G, pos, ax=self.axes)
        labels = nx.get_node_attributes(G, 'id')
        node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=self.axes)
        nx.draw_networkx_labels(G, pos, labels, font_size=20, font_color='black', verticalalignment='center', horizontalalignment='center')

        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]
        edge_widths = [3 * w for w in weights] # Set edge width proportional to weight
        colors = [G[u][v].get('color', 'black') for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='black', width=edge_widths, edge_vmax=max(weights)) 
        
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self, G):
        super().__init__()
        self.setWindowTitle('NetworkX Graph')
        self.graph_widget = GraphWidget(self)
        self.setCentralWidget(self.graph_widget)
        self.graph_widget.draw_graph(G)
        
        # for nmi 
        # self.nmi_button = QPushButton('Calculate NMI')
        # self.nmi_label = QLabel('NMI score:')
        # self.nmi_button.clicked.connect(self.calculate_nmi)
        
        # Create a button to apply the Girvan-Newman algorithm
        self.gn_button = QPushButton('Apply Girvan-Newman')
        self.gn_button.clicked.connect(self.apply_girvan_newman)

        # Create a button to stop the Girvan-Newman algorithm
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_girvan_newman)

        # Add the buttons to a horizontal layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.gn_button)
        # button_layout.addWidget(self.stop_button)
        self.centrality_combo = QComboBox()
        self.centrality_combo.addItems(['Closeness Centrality', 'Betweenness Centrality', 'Harmonic Centrality'])

        self.calculate_button = QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate_centrality)
        button_layout.addWidget(self.centrality_combo)
        button_layout.addWidget(self.calculate_button)

        self.rankpage_button=QPushButton('Rank page')
        self.rankpage_button.clicked.connect(self.calculate_pagerank)
        button_layout.addWidget(self.rankpage_button)

        self.table = QTableWidget()

        self.table_pagerank = QTableWidget()
        self.table_pagerank.setColumnCount(2)
        self.table_pagerank.setHorizontalHeaderLabels(['Node', 'PageRank'])
        header = self.table_pagerank.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        # Add the table showing the centrality values
        self.table_centrality = QTableWidget()
        self.table_centrality.setColumnCount(2)
        self.table_centrality.setHorizontalHeaderLabels(['Node', 'Centrality'])
        header = self.table_centrality.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        

        # # Create a main widget and set the layout
        # main_widget = QWidget(self)
        # main_widget.setLayout(layout)
        # self.setCentralWidget(main_widget)
        # Add the horizontal layout to a vertical layout
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.table_centrality)
        layout.addWidget(self.table_pagerank)
        # layout.addLayout(button_layout)
        layout.addWidget(self.graph_widget)
        # layout.addLayout(button_layout)
        # layout.addWidget(self.graph_widget)
        # layout.addWidget(self.table)

        # Create a main widget and set the layout
        main_widget = QWidget(self)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.G = G.copy()
        self.removed_edges = []
    
    def calculate_pagerank(self):
        # Calculate PageRank scores
        pagerank_scores = nx.pagerank(self.G)
        #(pagerank_scores)
        # Clear the table and add the new PageRank scores
        self.table_pagerank.setRowCount(0)
        for node, score in pagerank_scores.items():
            row = self.table_pagerank.rowCount()
            self.table_pagerank.insertRow(row)
            self.table_pagerank.setItem(row, 0, QTableWidgetItem(str(node)))
            self.table_pagerank.setItem(row, 1, QTableWidgetItem(str(score)))
            if score == max(pagerank_scores.values()):
                # Highlight the row with the largest centrality value
                for col in range(2):
                    self.table_pagerank.item(row, col).setBackground(QColor(255, 255, 100))
    


    def calculate_centrality(self):
        # Get the selected centrality measure from the combo box
        centrality_type = str(self.centrality_combo.currentText())
        #(centrality_type)
        if centrality_type == 'Closeness Centrality':
            centrality = nx.closeness_centrality(self.G)
        elif centrality_type == 'Betweenness Centrality':
            centrality = nx.betweenness_centrality(self.G)
        elif centrality_type == 'Harmonic Centrality':
            centrality = nx.harmonic_centrality(self.G)

        # Sort the centrality values in descending order
        centrality = {k: v for k, v in sorted(centrality.items(), key=lambda item: -item[1])}

        # Display the centrality values in the table
        self.table_centrality.setRowCount(len(centrality))
        row = 0
        for node, value in centrality.items():
            #(node,value)
            self.table_centrality.setItem(row, 0, QTableWidgetItem(str(node)))
            self.table_centrality.setItem(row, 1, QTableWidgetItem(str(value)))
            if value == max(centrality.values()):
                # Highlight the row with the largest centrality value
                for col in range(2):
                    self.table_centrality.item(row, col).setBackground(QColor(255, 255, 100))
            row += 1

        # Resize the table columns to fit the contents
        # self.table.resizeColumnsToContents()

    def apply_girvan_newman(self):
        print(self.G.nodes())
        if len(self.removed_edges) == len(self.G.edges()):
            # All edges have been removed, stop the algorithm
            return

        # Compute the betweenness centrality of the edges
        edge_centrality = nx.edge_betweenness_centrality(self.G)
        #("Betweenness centrality values:", edge_centrality)

        # Remove the edge with the highest betweenness centrality
        max_centrality = max(edge_centrality.values())
        for edge, centrality in edge_centrality.items():
            if centrality == max_centrality:
                self.G.remove_edge(*edge)
                self.removed_edges.append(edge)
                break

        # Draw the updated graph
        self.graph_widget.draw_graph(self.G)

        # Display a message with the removed edges
        if len(self.removed_edges) > 0:
            last_edge = self.removed_edges[-1]
            message = f"Next edge to remove: {edge}"
        else:
            message = f"Edge {edge} removed. No more edges to remove."
        #(message)
        self.display_message(message)

    def stop_girvan_newman(self):
        # Reset the graph to its original state
        self.G = nx.karate_club_graph()
        self.removed_edges = []
        self.graph_widget.draw_graph(self.G)

    def display_message(self, message):
        # Show a message box with the specified message
        msg_box = QMessageBox()
        # msg_box.setWindowTitle("Message")
        # msg_box.setText(message)
        # msg_box.setIcon(QMessageBox.Information)
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.setDefaultButton(QMessageBox.Ok)
        # msg_box.exec_()

        # # Create a timer to hide the message box after 3 seconds
        # timer = QTimer(self)
        # timer.timeout.connect(lambda: msg_box.hide())
        # timer.start(3000)

if __name__ == '__main__':
    # Load node and edge data from CSV files
    nodes = pd.read_csv('InputFileNodes.csv')
    edges = pd.read_csv('InputFileEdges.csv')

    G = nx.Graph()
    nodes_array = nodes.to_dict(orient='records')
    edges_array = edges.to_dict(orient='records')

    for node_attrs in nodes_array:
        G.add_node(node_attrs['id'], **node_attrs)

    for edge_attrs in edges_array:
        G.add_edge(edge_attrs['source'], edge_attrs['target'], **edge_attrs)

    app = QApplication(sys.argv)
    main_window = MainWindow(G)
    main_window.show()
    sys.exit(app.exec_())

