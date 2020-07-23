
import os
from graphviz import Digraph


def draw_genotype(genotype, n_nodes, filename, concat=None, format="svg"):
    g = Digraph(
        format=format,
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center',
                       fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("-2", fillcolor='darkseagreen2')
    g.node("-1", fillcolor='darkseagreen2')
    steps = n_nodes

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for op, source, target in genotype:
        if source == 0:
            u = "-2"
        elif source == 1:
            u = "-1"
        else:
            u = str(source - 2)
        v = str(target-2)
        g.edge(u, v, label=op, fillcolor="gray")

    g.node("out", fillcolor='palegoldenrod')
    if concat is not None:
        for i in concat:
            if i-2 >= 0:
                g.edge(str(i-2), "out", fillcolor="gray")
    else:
        for i in range(steps):
            g.edge(str(i), "out", fillcolor="gray")

    g.render(filename, view=False)
    os.remove(filename)
