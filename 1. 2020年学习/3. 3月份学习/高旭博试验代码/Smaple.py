import igraph
#from igraph import *
print(igraph.__version__)
g = igraph.Graph()
g.add_vertices(3)
g.add_edges([(1,2)])
g.add_edges([(2,0)])
g.add_vertices(3)
g.add_edges([(2,3),(3,4),(4,5),(5,3),(0,1)])
igraph.summary(g)
print(g.degree())
karate = igraph.Graph.Famous("Zachary")
#igraph.plot(karate)
#mc = karate.community_multilevel()  # NOT HIERARCHICAL: Returns clustering object directly
wdend = karate.community_walktrap() # Returns Dendogram
communities1 = wdend.as_clustering() # Convert to clustering object
communities2 = wdend.as_clustering(2) # Convert to clustering object //  With desired community NUMBER
igraph.plot(communities1)

number1 = karate.modularity(communities1)
number2 = karate.modularity(communities2)
print(number1)
print(number2)

#communities.modularity()
#membership(wc)