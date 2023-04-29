import sys
import networkx as nx
import pandas as pd
import numpy as np
import os
from networkx.algorithms.community.quality import modularity
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,QDesktopWidget,QCheckBox
from PyQt5.QtGui import QPalette, QColor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support,normalized_mutual_info_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPen, QColor
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer 
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QVBoxLayout, QWidget,QLineEdit
from sklearn.metrics.cluster import contingency_matrix, entropy
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


class GraphWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.is_directed=False
        self.is_weight=False

    def draw_graph(self, G , partition=None):
        self.axes.clear()
        pos = nx.spring_layout(G)
        # print('here',partition)
        for node in G.nodes():
            G.nodes[node]['color'] = 'red'
        if partition is not None:
            for node in G.nodes():
                print(partition[node])
                G.nodes[node]['color'] = partition[node]
        
        degrees = dict(nx.degree(G))
        node_sizes = [300 * degrees[node] for node in G.nodes()]
        nx.draw(G, pos, ax=self.axes,with_labels=True)
        labels = nx.get_node_attributes(G, 'id')
        node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=self.axes)
        nx.draw_networkx_labels(G, pos, labels, font_size=20, font_color='black', verticalalignment='center', horizontalalignment='center')

        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]

        if len(weights) == 0:
            edge_vmax = 1.0
        else:
            edge_vmax = max(weights)
         # Set edge width proportional to weight
        # print(edge_widths)
        edge_widths = [ 1 for w in weights]
        if self.is_weight:
            edge_widths = [ 0.75*w for w in weights]

        if self.is_directed:
            print(self.is_directed)
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='black', width=edge_widths, edge_vmax=edge_vmax,ax=self.axes, arrows=True,arrowstyle='->', arrowsize=10) 
        else:
            print(self.is_directed)
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='black', width=edge_widths,  edge_vmax=edge_vmax,ax=self.axes) 

        self.fig.tight_layout()
        self.draw()
    def set_directed(self, directed):
        self.is_directed = directed
    def set_weight(self,weight):
        self.is_weight=weight


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_directed = False
        self.G = nx.Graph()
        self.setWindowTitle('NetworkX Graph')
        self.graph_widget = GraphWidget(self)
        self.setCentralWidget(self.graph_widget)
        self.graph_widget.draw_graph(self.G)
        
        # Create a button to apply the Girvan-Newman algorithm
        self.gn_button = QPushButton('Apply Girvan-Newman')
        self.gn_button.clicked.connect(self.apply_girvan_newman)

        #quality (evaluation) part
        quality_box=QHBoxLayout()
        self.result_label = QLabel(self)
        self.calculate_button = QPushButton("Calculate quality", self)
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItem("NMI")
        self.algorithm_selector.addItem("Modularity")
        self.algorithm_selector.addItem("coverage")
        self.algorithm_selector.addItem("Conductance")
        self.calculate_button.clicked.connect(self.calculate_quality)
        quality_box.addWidget(self.algorithm_selector)
        quality_box.addWidget(self.calculate_button)
        quality_box.addWidget(self.result_label)

        # Add the buttons to a horizontal layout
        girvan_rank=QVBoxLayout()
        self.rankpage_button=QPushButton('Rank page')
        self.rankpage_button.clicked.connect(self.calculate_pagerank)
        girvan_rank.addWidget(self.gn_button)
        girvan_rank.addWidget(self.rankpage_button)

        centeralities=QVBoxLayout()
        self.centrality_combo = QComboBox()
        self.centrality_combo.addItems(['Closeness Centrality', 'Betweenness Centrality', 'Harmonic Centrality'])
        self.centerality_calculate_button = QPushButton('Calculate')
        self.centerality_calculate_button.clicked.connect(self.calculate_centrality) 
        self.textbox = QLineEdit()
        # self.filter=QPushButton('filter') 
        # self.filter.clicked.connect(self.filteration)
        centeralities.addWidget(self.textbox)
        # centeralities.addWidget(self.filter)
        centeralities.addWidget(self.centrality_combo)
        centeralities.addWidget(self.centerality_calculate_button)

        label = QtWidgets.QLabel('select nodes spreadsheet:')
        # Create a button for uploading the file
        self.nodes_btn = QtWidgets.QPushButton('Upload nodes')
        self.nodes_btn.clicked.connect(self.upload_nodes_file)
        self.edges_btn = QtWidgets.QPushButton('Upload edges')
        self.edges_btn.clicked.connect(self.upload_edges_file)
        self.directed_checkbox = QCheckBox('Convert graph to directed')
        self.weight_checkbox = QCheckBox('Convert graph to wieghted')
        self.start_btn=QPushButton('start')
        self.start_btn.clicked.connect(self.create_graph)
        self.directed_checkbox.stateChanged.connect(self.create_graph)
        self.weight_checkbox.stateChanged.connect(self.create_graph)
        # self.start.clicked.connect(self.show_graph)
        right_bar=QVBoxLayout()
        right_bar.addWidget(self.start_btn)
        right_bar.addWidget(self.directed_checkbox)
        right_bar.addWidget(self.weight_checkbox)
        right_bar.addWidget(label)
        right_bar.addWidget(self.nodes_btn)
        right_bar.addWidget(self.edges_btn)
        right_bar.addLayout(girvan_rank)
        right_bar.addLayout(quality_box)
        right_bar.addLayout(centeralities)
        right_bar.setSpacing(0)
        tables=QHBoxLayout()
        self.table_girvan=QTableWidget()


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

        self.table_modularity=QTableWidget()
        self.table_modularity.setColumnCount(2)
        self.table_modularity.setHorizontalHeaderLabels(['Node', 'Modularity'])
        header = self.table_modularity.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # row = self.table_modularity.rowCount()
        # self.table_modularity.insertRow(row)
        # self.table_modularity.setItem(row, 0, QTableWidgetItem(str(modularity_score)))


        tables.addWidget(self.table_girvan)
        tables.addWidget(self.table_modularity)
        tables.addWidget(self.table_centrality)
        tables.addWidget(self.table_pagerank)
        # side_bar=QHBoxLayout()
        # side_bar.addLayout(button_layout)
        # side_bar.addLayout(quality_box)
        tables.setContentsMargins(0, 0, 0, 0)
        tables.setSpacing(0)
        screen = QDesktopWidget().screenGeometry()
        height = int(screen.height() * 0.2)
        self.table_centrality.setFixedHeight(height)
        self.table_pagerank.setFixedHeight(height)
        self.table_girvan.setFixedHeight(height)
        self.table_modularity.setFixedHeight(height)
        tables.setGeometry(self.table_pagerank.rect())

        hlayout=QHBoxLayout()

        layout = QVBoxLayout()
        # layout.addLayout(button_layout)
        # layout.addLayout(quality_box)
        
        # layout.addWidget(self.table_pagerank)
        layout.addWidget(self.graph_widget)
        layout.addLayout(tables)
        hlayout.addLayout(layout)
        hlayout.addLayout(right_bar)


        # Create a main widget and set the layout
        main_widget = QWidget(self)
        main_widget.setLayout(hlayout)
        self.setCentralWidget(main_widget)

        self.G = self.G.copy()
        self.removed_edges = []
    
        # return community_structure

    # def show_graph(self, G):
    def set_graphweight(self):
        if self.weight_checkbox.isChecked():
            self.graph_widget.set_weight(True)
        else:
            self.graph_widget.set_weight(False)

        if self.directed_checkbox.isChecked():
            self.G = nx.DiGraph()
            self.graph_widget.set_directed(True)
            self.G = self.G.to_directed()
        else:
           self. G = nx.Graph()
           self.graph_widget.set_directed(False)
           self.G = self.G.to_undirected()

        self.nodes_array = self.nodes.to_dict(orient='records')
        self.edges_array = self.edges.to_dict(orient='records')
        for node_attrs in self.nodes_array:
            self.G.add_node(node_attrs['id'], **node_attrs)

        for edge_attrs in self.edges_array:
            self.G.add_edge(edge_attrs['source'], edge_attrs['target'], **edge_attrs)
        self.graph_widget.draw_graph(self.G)

    def create_graph(self):
        if self.directed_checkbox.isChecked():
            self.G = nx.DiGraph()
            self.graph_widget.set_directed(True)
            self.G = self.G.to_directed()
        else:
           self. G = nx.Graph()
           self.graph_widget.set_directed(False)
           self.G = self.G.to_undirected()
        if self.weight_checkbox.isChecked():
            self.graph_widget.set_weight(True)
        else:
            self.graph_widget.set_weight(False)
        # self.nodes = pd.read_csv('InputFileNodes.csv')
        # self.edges = pd.read_csv('InputFileEdges.csv')
        # self.G = nx.Graph()
        self.nodes_array = self.nodes.to_dict(orient='records')
        self.edges_array = self.edges.to_dict(orient='records')
        for node_attrs in self.nodes_array:
            self.G.add_node(node_attrs['id'], **node_attrs)

        for edge_attrs in self.edges_array:
            self.G.add_edge(edge_attrs['source'], edge_attrs['target'], **edge_attrs)
        self.graph_widget.draw_graph(self.G)

    def set_directed(self, directed):
        self.is_directed = directed   
    def convert_graph(self, state):
        # Check the state of the checkbox
        if state == 2:
            # Convert the graph to directed
            self.graph_widget.set_directed(True)
            self.G = self.G.to_directed()
        else:
            # Convert the graph to undirected
            self.graph_widget.set_directed(False)
            self.G = self.G.to_undirected()

        # Redraw the graph
        self.graph_widget.draw_graph(self.G)
    
    def upload_nodes_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Spreadsheet', os.path.expanduser('~'), 'Spreadsheet Files (*.csv *.xlsx *.xls)')
        self.nodes = pd.read_csv(filename)
        self.nodes_array = self.nodes.to_dict(orient='records')
        print(self.nodes_array)

        # self.uploaded.emit(nodes)

    def upload_edges_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Spreadsheet', os.path.expanduser('~'), 'Spreadsheet Files (*.csv *.xlsx *.xls)')
        self.edges = pd.read_csv(filename)
        self.edges_array = self.edges.to_dict(orient='records')
        print(self.edges_array)
        # self.uploaded.emit(edges)


    def calculate_coverage(self,G, communities):
        # Calculate the total number of nodes in the graph
        num_nodes = len(self.G.nodes())
        # Calculate the sum of the sizes of the communities
        # community_sizes = [len(community) for community in communities]

        # total_community_size = sum(community_sizes)
        total_community_size=len(communities)
        print(num_nodes, total_community_size)
        # Calculate the coverage of the partition
        coverage = total_community_size / num_nodes

        return coverage
    def calculate_quality(self):
        algorithm = self.algorithm_selector.currentText()
        if algorithm=="Conductance":
            communities_generator = nx.algorithms.community.girvan_newman(self.G)
            communities = list(next(communities_generator))
            result=""
            for i, community in enumerate(communities):
                conductance = nx.algorithms.cuts.conductance(self.G, community)
                result+=f"Community {i}: Conductance = {conductance} \n"
                print(f"Community {i}: Conductance = {conductance}")
            self.result_label.setText(f"{result}")
            return
        elif algorithm=="coverage":
            detected_communities_generator = nx.algorithms.community.girvan_newman(self.G)
            detected_communities = list(next(detected_communities_generator))
            # Calculate the coverage of the partition
            print(detected_communities)
            result=""
            for i, community in enumerate(detected_communities):
                coverage = self.calculate_coverage(self.G, community)
                print(f"Coverage: {coverage}")
                result+=f"Community {i}: Coverage = {coverage} \n"
                # print(f"Community {i}: Conductance = {conductance}")
            self.result_label.setText(f"{result}")
            # self.result_label.setText(f"coverage = {coverage}")
            return
        elif algorithm=="Modularity":
            # Detect communities using the Girvan-Newman algorithm
            communities = next(nx.algorithms.community.girvan_newman(self.G))
            node_to_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = i
            mod_no= 0
            node_modularity = {}
            for node in self.G.nodes():
                community = node_to_community.get(node, None)
                if community is not None:
                    subgraph = self.G.subgraph(communities[community])
                    node_modularity[node] = community
                    mod_no=modularity(self.G, communities, subgraph)
                else:
                    node_modularity[node] = 0.0

            # print(node_modularity)
            # Clear the table and add the modularity values
            self.table_modularity.setRowCount(0)
            for node, modularity_score in node_modularity.items():
                print(node,modularity_score)
                row = self.table_modularity.rowCount()
                self.table_modularity.insertRow(row)
                self.table_modularity.setItem(row, 0, QTableWidgetItem(str(node)))
                self.table_modularity.setItem(row, 1, QTableWidgetItem(str(modularity_score)))
                
            self.result_label.setText(f' modularity : {mod_no}')

        elif algorithm=="NMI":
            print(self.nodes)
            # first get the true classes of nodes
            ground_truth = self.nodes['class'].values
            # apply the community detection to know communities 
            detected_communities_generator = nx.algorithms.community.girvan_newman(self.G)
            # b list kul wahed le wa7du 
            detected_communities = list(next(detected_communities_generator))
            # bged 3dd el communities eli et3mlt 
            n_communities = len(detected_communities)
            # number of nodes
            n = self.nodes.shape[0]

            if n_communities > 0:
                for i in range(n_communities):
                    if i < n_communities:
                        # table for know the nodes that found in both true and detected communities
                        contingency_table = pd.crosstab(ground_truth, [node in detected_communities[i] for node in self.G.nodes()])
                        #summition of true nodes
                        ground_truth_marginal = contingency_table.sum(axis=1)
                        #summition of false nodes
                        detected_marginal = contingency_table.sum(axis=0)

                        pointwise_mi = np.zeros_like(contingency_table, dtype=np.float64)

                        for j in range(contingency_table.shape[0]):
                            for k in range(contingency_table.shape[1]):

                                pij = contingency_table.iloc[j, k] / n

                                pi = ground_truth_marginal.iloc[j] / n

                                pj = detected_marginal.iloc[k] / n

                                if (pi != 0 and pj != 0):
                                    pointwise_mi[j, k] = np.log2(pij / (pi*pj))
                                else:
                                    pointwise_mi[j, k] = np.log2(0)

                        mi = np.sum(np.nan_to_num(contingency_table / n * pointwise_mi))
                        eps = 1e-10  # small constant to add to entropy terms
                        nmi = mi / np.sqrt((entropy(ground_truth_marginal / n) + eps) * (entropy(detected_marginal / n) + eps))

                self.result_label.setText(f"NMI = {nmi}")
                print(f"NMI = {nmi}")
            return 

        # Calculate the conductance of the set
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
        value = self.textbox.text()
        if not value:
            False  
        else:
            try:
                value = round(float(value),1)   # Convert the input to a float
            except ValueError:
                # If the input cannot be converted to a float, show an error message
                print('invalid error')
        centrality_type = str(self.centrality_combo.currentText())
        if centrality_type == 'Closeness Centrality':
            centrality = nx.closeness_centrality(self.G)
        elif centrality_type == 'Betweenness Centrality':
            centrality = nx.betweenness_centrality(self.G)
        elif centrality_type == 'Harmonic Centrality':
            centrality = nx.harmonic_centrality(self.G)
            n = len(self.G.nodes)
            for node in centrality:
                centrality[node] /= (n-1)

        # Sort the centrality values in descending order
        if value:
            centrality = {k: v for k, v in sorted(centrality.items(), key=lambda item: -item[1]) if round(float(v),1)==value}
        else:
            centrality = {k: v for k, v in sorted(centrality.items(), key=lambda item: -item[1]) }
        print(centrality , value)
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
        


    # def calculate_centrality(self):
    #     # Get the selected centrality measure from the combo box
    #     centrality_type = str(self.centrality_combo.currentText())
    #     #(centrality_type)
    #     if centrality_type == 'Closeness Centrality':
    #         centrality = nx.closeness_centrality(self.G)
    #     elif centrality_type == 'Betweenness Centrality':
    #         centrality = nx.betweenness_centrality(self.G)
    #     elif centrality_type == 'Harmonic Centrality':
    #         centrality = nx.harmonic_centrality(self.G)
    #         n = len(self.G.nodes)
    #         for node in centrality:
    #             centrality[node] /= (n-1)

    #     # Sort the centrality values in descending order
    #     centrality = {k: v for k, v in sorted(centrality.items(), key=lambda item: -item[1])}

    #     # Display the centrality values in the table
    #     self.table_centrality.setRowCount(len(centrality))
    #     row = 0
    #     for node, value in centrality.items():
    #         #(node,value)
    #         self.table_centrality.setItem(row, 0, QTableWidgetItem(str(node)))
    #         self.table_centrality.setItem(row, 1, QTableWidgetItem(str(value)))
    #         if value == max(centrality.values()):
    #             # Highlight the row with the largest centrality value
    #             for col in range(2):
    #                 self.table_centrality.item(row, col).setBackground(QColor(255, 255, 100))
    #         row += 1

    #     # Resize the table columns to fit the contents
    #     # self.table.resizeColumnsToContents()

    def apply_girvan_newman(self):
        #(self.G.nodes())
        if len(self.removed_edges) == len(self.G.edges()):
            # All edges have been removed, stop the algorithm
            return
        # Compute the betweenness centrality of the edges
        edge_centrality = nx.edge_betweenness_centrality(self.G)
        max_centrality = max(edge_centrality.values())
        for edge, centrality in edge_centrality.items():
            if centrality == max_centrality:
                self.G.remove_edge(*edge)
                self.removed_edges.append(edge)
                break

        # Draw the updated graph
        communities_generator = nx.community.girvan_newman(self.G)
        community_color=['red','orange','yellow','blue','green','gold','black','brown','cyan','gray']
        new_colors={}
        partition = tuple(sorted(c) for c in next(communities_generator))
        num_communities = len(partition)
        self.table_girvan.setRowCount(num_communities)
        self.table_girvan.setColumnCount(2)
        for i, community in enumerate(partition):
            nodes = ', '.join(str(node) for node in community)
            for node in community:
                new_colors[node]=community_color[i%len(community_color)]
            print(new_colors)
            self.table_girvan.setItem(i, 0, QTableWidgetItem(f"Community {i+1}"))
            self.table_girvan.setItem(i, 1, QTableWidgetItem(nodes))
        # Display a message with the removed edges
        self.graph_widget.draw_graph(self.G,partition=new_colors)
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


if __name__ == '__main__':
    # Load node and edge data from CSV files
    # nodes = pd.read_csv('InputFileNodes.csv')
    # edges = pd.read_csv('InputFileEdges.csv')

    # G = nx.DiGraph()
    # nodes_array = nodes.to_dict(orient='records')
    # edges_array = edges.to_dict(orient='records')

    # for node_attrs in nodes_array:
    #     G.add_node(node_attrs['id'], **node_attrs)

    # for edge_attrs in edges_array:
    #     G.add_edge(edge_attrs['source'], edge_attrs['target'], **edge_attrs)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

