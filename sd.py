import networkx as nx
import matplotlib.pyplot as plt

# Définir les nœuds (articulations de la main) et leurs connexions (arêtes)
nodes = {
    1: "Poignet",
    2: "Base du pouce",
    3: "Pouce 1",
    4: "Pouce 2",
    5: "Extrémité pouce",
    6: "Base de l'index",
    7: "Index 1",
    8: "Index 2",
    9: "Extrémité index",
    10: "Base du majeur",
    11: "Majeur 1",
    12: "Majeur 2",
    13: "Extrémité majeur",
    14: "Base de l'annulaire",
    15: "Annulaire 1",
    16: "Annulaire 2",
    17: "Extrémité annulaire",
    18: "Base de l'auriculaire",
    19: "Auriculaire 1",
    20: "Auriculaire 2",
    21: "Extrémité auriculaire"
}

# Arêtes reliant les nœuds (structure de la main)
edges = [
    (1, 2), (2, 3), (3, 4), (4, 5),  # Pouce
    (1, 6), (6, 7), (7, 8), (8, 9),  # Index
    (1, 10), (10, 11), (11, 12), (12, 13),  # Majeur
    (1, 14), (14, 15), (15, 16), (16, 17),  # Annulaire
    (1, 18), (18, 19), (19, 20), (20, 21)   # Auriculaire
]

# Créer un graphe
G = nx.Graph()
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Ajouter des étiquettes aux nœuds pour la visualisation
node_labels = {key: value for key, value in nodes.items()}

# Visualisation du graphe
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Générer une disposition des nœuds

# Dessiner le graphe
nx.draw(G, pos, with_labels=False, node_color="skyblue", node_size=1000, edge_color="gray", linewidths=1, font_size=15)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black")

plt.title("Graphe représentant les articulations d'une main", fontsize=15)
plt.show()
