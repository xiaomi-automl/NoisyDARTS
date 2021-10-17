import os
import sys
from graphviz import Digraph

import genotypes
from genotypes import PRIMITIVES

COLORS = ["#E53935", "#FF6700", "#83C44E", "#2196F3", "#00C0A5", "#0071BC", "#004776"]
OP_COLORS = {p:c for p,c in zip(PRIMITIVES, COLORS)}

SHORT_LABELS = ['max3','avg3','skip','sep3','sep5','dil3','dil5']
OP_SHORT_LABELS = {p:l for p,l in zip(PRIMITIVES, SHORT_LABELS)}

def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='left', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR']) #LR


  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  with g.subgraph(name='inter') as c:
    c.attr(rank='same')
    for i in range(steps):
      g.node(str(i), fillcolor='#FF6700')

  with g.subgraph(name='output') as c:
    g.node("c_{k}", fillcolor='#DEDEDE')

    g.node("c_{k-2}", fillcolor='#DEDEDE')
    g.node("c_{k-1}", fillcolor='#DEDEDE')

  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="#DEDEDE", color='#aaaaaa')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=OP_SHORT_LABELS[op], fillcolor=OP_COLORS[op], color=OP_COLORS[op])



  g.render('./figures/' + filename, view=True)


if __name__ == '__main__':

  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  if not os.path.exists('figures'):
    os.makedirs('figures')

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name))
    sys.exit(1)

  plot(genotype.normal, "%s_normal"%genotype_name)
  plot(genotype.reduce, "%s_reduction"%genotype_name)