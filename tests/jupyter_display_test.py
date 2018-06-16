
from IPython.display import display
import IPython.core.formatters

import networkx as nx, graphviz, networkx.algorithms.dag

dg = nx.DiGraph()
# G.add_node(1)
dg.add_nodes_from(['ALG', 'ANL', 'MECH', 'STAT', 'VECT'])
dg.add_edges_from([
    ['STAT', 'ANL'],
    ['STAT', 'ALG'],
    ['ANL', 'ALG'],
    ['ALG', 'MECH'],
    ['ALG', 'VECT'],
    ['VECT', 'MECH'],
])

dg_dot = graphviz.Digraph(comment='disease-network')
for node in dg.nodes():
    dg_dot.node(node)

for edge in dg.edges():
    dg_dot.edge(edge[0],edge[1], dir='none')

dg_dot

display(dg_dot)

dg_dot._repr_svg_()
dg_dot.render()

fm = IPython.core.formatters.IPythonDisplayFormatter()
