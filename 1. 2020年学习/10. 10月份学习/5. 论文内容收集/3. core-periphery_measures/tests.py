import networkx as nx
import matplotlib.pyplot as plt
from cp_methods import *


# Helper Methods for testing 

def generate_figure(G, nodelist, colorlist, title="", saveas=""):
	# Generate graph figure using Kamada Kawai Layout
	pos = nx.kamada_kawai_layout(G)
	nx.draw_networkx_edges(G, pos)
	nodes = nx.draw_networkx_nodes(G, pos, node_list=nodelist, 
		node_size=100, node_color=colorlist, cmap=plt.cm.jet)
	plt.colorbar(nodes)
	plt.axis('off')
	plt.title(title)
	plt.savefig('plots/'+saveas)

def sbm_figure(scores, N, title, saveas):
	# Graph of scores versus node index
	plt.bar(range(N), scores)
	plt.xlabel('Node index')
	plt.ylabel('Core score')
	plt.title(title)
	plt.savefig('plots/'+saveas)


def core_score_vs_index(k, p, size_c, size_p):
	# Generate core score vs index plots for all methods using SBM 
	G = sbm(k, p, size_c, size_p)
	N = size_c + size_p

	t_suffix=', N='+str(N)+', k='+str(k)
	f_suffix='_sbm_kappa='+str(k)+'.png'

	plt.figure(1) # Random Walk persistences
	[profile, persistences] = periphery_profile(G) # Random walk
	sbm_figure(list(map(lambda i: persistences[profile.index(i)], range(N))), N, 
		'Persistence'+t_suffix, 'pprofile'+f_suffix)

	plt.figure(2) # Path core
	sbm_figure(path_core(G), N, "Path-Core"+t_suffix, 'pathcores'+f_suffix)

	plt.figure(3) # Degree
	sbm_figure(list(map(lambda x: G.degree(x), range(N))), N, "Degree"+t_suffix, 
		'degrees'+f_suffix)

	plt.figure(4) # Betweenness Centralities
	C = nx.betweenness_centrality(G)
	sbm_figure([C[i] for i in range(N)], N, 'Betweenness'+t_suffix, 'betweenness'+f_suffix)

	plt.show()


def colormap_test(G, name):
	# Test all methods by graphing colormaps of corescores
	# name is name of graph for saving purposes
	N = nx.number_of_nodes(G)
	
	[profile, persistences] = periphery_profile(G)
	plt.figure(1) # Random walker method
	generate_figure(G, range(N), list(map(lambda i: persistences[profile.index(i)], range(N))), 
				"Random Walker Persistence", 'cm_rw_'+name+'.png')

	plt.figure(2) # Path-core method
	generate_figure(G, range(N), path_core(G),
		"Path-Core", 'cm_pathcore_'+name+'.png')

	plt.figure(3) # Degree Method
	generate_figure(G, range(N), list(map(lambda x: G.degree(x), range(N))), 
				"Degree Score", 'cm_degree_'+name+'.png')


	C = nx.betweenness_centrality(G)
	plt.figure(4) # Betweenness centrality 
	generate_figure(G, range(N), [C[i] for i in range(N)], 
		"Betweenness Centrality Score", 'cm_between_'+name+'.png')

	plt.show()

	

def test_sbm(k, p, size_c, size_p):
	# test sbm method
	SBM = sbm(k, p, size_c, size_p)
	N = size_c + size_p
	A = nx.to_numpy_matrix(SBM)


	plt.figure(1) 
	colors = ['g' for i in range(size_c)] + ['b' for i in range(size_p)]
	nx.draw_networkx(SBM, node_color=colors, node_size=200, with_labels=False)
	plt.title("Graph generated using SBM")
	plt.axis('off')
	plt.savefig('plots/sbm_graph.png')

	plt.matshow(A)
	plt.title("Adjacency matrix of SBM")
	plt.axis('off')
	plt.savefig("plots/SBM_adjacency.png")

	plt.show()



def test_coefficients(T, p, size_c, size_p):
	# Test different core-periphery coefficients on SBM 

	iters = 5 # Number of iterations over which to average for each k
	k_range = np.arange(1,2,0.05) 
	f_suffix = '_sc='+str(size_c)+'_sp='+str(size_p)+'its='+str(iters)+'.png'
	
	h_coeffs = [] # Holme coefficients
	d_coeffs = [] # Degree coefficients

	for k in k_range:
		avg_h_coeff, avg_d_coeff = 0, 0
		for i in range(iters):
			G = sbm(k,p,size_c,size_p)
			avg_h_coeff += holme_coefficient(G,T) 
			avg_d_coeff += degree_coefficient(G,T)

		h_coeffs.append(avg_h_coeff / float(iters))
		d_coeffs.append(avg_d_coeff / float(iters))

	plt.figure(1)
	plt.plot(k_range, h_coeffs, 'go--')
	plt.ylabel('Value')
	plt.xlabel('k')
	plt.title('Holme Coefficient')
	plt.savefig('plots/h_coeff'+f_suffix)

	plt.figure(2)
	plt.plot(k_range, d_coeffs, 'go--')
	plt.ylabel('Value')
	plt.xlabel('k')
	plt.title('Naive Degree Coefficient')
	plt.savefig('plots/d_coeff'+f_suffix)

	plt.show()





# Run tests

# Block model parameters
p = 0.25
size_c, size_p = 10, 30


test_sbm(1.8, p, size_c, size_p)


# for k in [1.3, 1.5, 1.8, 2]:
# 	core_score_vs_index(k, p, size_c, size_p)


#G1 = nx.karate_club_graph()

# Colormap tests with SBM

n_c, n_p = 10, 30

p_strong = 0.2
k_strong = 2.2
SBM_strong = sbm(k_strong, p_strong, n_c, n_p)
#colormap_test(SBM_strong, 'sbm_strong')

p_weak = 0.25
k_weak = 1.3 
SBM_weak = sbm(k_weak, p_weak, n_c, n_p)
#colormap_test(SBM_weak, 'sbm_weak')


# Colormap tests
data = nx.read_edgelist("graphs/train.txt", nodetype=int, comments='%')

G = nx.Graph() 
G.add_nodes_from(range(len(data.nodes())))
G.add_edges_from([(e[0]-1,e[1]-1) for e in data.edges()])
colormap_test(G, 'train')


# Test coefficients
# test_coefficients(50, 0.25, 15, 15)











