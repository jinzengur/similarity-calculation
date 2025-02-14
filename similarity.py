import numpy as np
import torch
import collections
import surfgraph.chemical_environment as surfgraph_chem_env
import networkx as nx
import ase
from ase.io import read
from itertools import permutations

ad_elements=['C', 'O', 'N', 'H']
surf_elements=['Pt','Sn','Ru']
grid = [1,1,0]
radius = 2
dist = torch.nn.PairwiseDistance(eps=1e-10)
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
cutoff = 1.1
prefactor = 10
torch.set_default_dtype(torch.float32)

def get_graphs(atoms,r=radius):
    """

    Args:
        atoms (ase.Atoms): The ase.Atoms object to operate on

        r (float,optional): SURFGRAPH graph generation radius, default is 2

    Returns:
        list: [Full-graph and [ego-graphs]]
    """
    nl = ase.neighborlist.NeighborList(ase.neighborlist.natural_cutoffs(atoms, cutoff), self_interaction=False,
                      bothways=True, skin=0.25)
    nl.update(atoms)
    adsorbate_atoms = [atom.index for atom in atoms if atom.symbol in ad_elements]

    full, chem_env = surfgraph_chem_env.process_atoms(atoms, nl, adsorbate_atoms=adsorbate_atoms,grid=grid,radius=r)  ## get the full graph and the chem_env for all the adsorbates found

    return (full, chem_env)

def get_node_name(graph,ele_name):
    """

    get the node name of a node that represent an atom with
    the input element.

    Args:
        graph (nx.graph): an ego graph of interest
        ele_name (str): element name. E.g. Pt
    Returns:
        node_str (str): 
    """
    node_str=''
    for n in graph.nodes():
        at_name = n.split(':')[0]

        if at_name == ele_name:
            node_str = n
            break
            
    return node_str

def check_bond_type(node1_str,node2_str,powr):
    """

    compute the edge weight of the edge connecting the given two nodes
    e_ij = (1/z_i + 1/z_j) ^ powr
    z_i, z_j: atomic indexes in periodic table of node1, node2

    Args:
        node1_str (str): node1 name
        node2_str (str): node2 name
        powr (int): power in the edge weight assignment formula
    Returns:
        dist (float): edge weight
    """
     

    node1_ele = node1_str.split(':')[0]
    node2_ele = node2_str.split(':')[0]
    
    node1_atom = ase.Atom(node1_ele)
    node2_atom = ase.Atom(node2_ele)
    

    dist = np.sum([1/atom.number for atom in [node1_atom,node2_atom]]) ** powr
        
        
    return dist

def bfs(graph, root, adj_mat=False):
    """

    breadth first search

    Args:
        graph (nx.graph): graph of interest
        root (str): root node name
    Returns:
        visited (list(str)): list of node names in bfs sequence
        visited_edge_type: list of edge weights in bfs sequence
    """

    visited, queue = list(), collections.deque([root])
    visited.append(root)
    visited_edge_type = []

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()
        # print(str(vertex) + " ", end="\n")

        # If not visited, mark it as visited, and
        # enqueue it
        # print('vertex:',vertex)
        for neighbour in graph.neighbors(vertex):
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
                edge_type = check_bond_type(vertex,neighbour,-1)
                visited_edge_type.append(edge_type)

    return visited,visited_edge_type

def dfs(graph, start, visited=None, visited_edges_type=None):
    """

    depth first search

    Args:
        graph (nx.graph): graph of interest
        start (str): root node name
    Returns:
        visited (list(str)): list of node names in dfs sequence
        visited_edge_type: list of edge weights in dfs sequence
    """

    if visited is None:
        visited = list()
    visited.append(start)

    if visited_edges_type is None:
        visited_edges_type = list()

    # print(start)
    for next in graph.neighbors(start):
        if next not in visited:
            dist = check_bond_type(start,next,-1)
            visited_edges_type.append(dist)
            dfs(graph, next, visited,visited_edges_type)

    return visited,visited_edges_type
    
def get_curved_weight(ego_arr,nodes_str_arr,edge_weight):
    """

    modifiy the edge weight according to the formula:
    e_ij_new = (1/2) ^ r * e_ij, for r > 0
    e_ij_new = -10 * e_ij, for r = 0
    r is the number of edges between a node in the edge of interest to a node in the central adsorbate
    along the shortest path.

    Args:
        ego_arr (list(nx.graph)): list of ego graphs at different atomic shells in various atomic radius
        nodes_str_arr (list(str)): list of names of the node of interest and a neighboring connected node
        edge_weight (float): un-modified edge weight
    Returns:
        weight (float): modified edge weight 
    """

    node1_ele = nodes_str_arr[0].split(':')[0]
    node2_ele = nodes_str_arr[1].split(':')[0]

    intersect = set([node1_ele,node2_ele]).intersection(set(ad_elements)) # check if this edge is an ad-ad/ad-metal bond
    # print('nodes',nodes_str_arr[0],nodes_str_arr[1])
    weight = edge_weight

    for (ego_idx,ego) in enumerate(ego_arr):
        
        if ego.has_edge(nodes_str_arr[0],nodes_str_arr[1]):
            
            if (len(list(intersect)) >= 1) and (ego_idx != 0): # ad-metal or ad-ad bond not belong to the active site
                
                weight = edge_weight * 0.5**(ego_idx//2)
                
                break
            elif len(list(intersect)) == 0: # all metal-metal bond
                weight = edge_weight * 0.5 ** (ego_idx//2)
                break
            elif (len(list(intersect)) >= 1) and (ego_idx == 0): # ad-ad or ad-metal bond belong to the active sites
                
                weight = prefactor * edge_weight
                
                break
            
    return weight


def get_subgraph_ri_from_graph(graph,ri):
    """

    get the ego graph at the given atomic shell radius from a large edgo graph

    Args:
        graph (nx.graph): a large graph (r >= 2)
        ri (float): atomic shell radius
    Returns:
        ego_ego_ref (nx.graph): ego graph at the given atomic shell radius
    """

    center_ad_nodes = [x for x,y in graph.nodes(data=True) if y['central_ads']==True]
    ego_ego_ref = nx.ego_graph(graph, center_ad_nodes[0], radius=(ri*2)+1, distance="dist")
    ego_ego_ref = nx.Graph(nx.subgraph(graph, list(ego_ego_ref.nodes())))
    
    return ego_ego_ref

def get_adj_mat(ego_ref,powr,start_atom_ele='Pt'):
    """

    get the weighted adjacency matrix of an ego graph.
    The edge weight is assigned in get_curved_weight

    Args:
        full (nx.graph): the full graph that the given ego graph belongs to
        ego_ref (nx.graph): the ego graph of interest.
        atoms (ase.Atoms): the ase.Atoms object of the configuration of interest
        powr: power in the edge weight assignment formula
        start_atom_ele (optional, str): element of the root node for the bfs search
                                        default is 'Pt', but can changed to any element
                                        that presents in the graph. 
                                        Using the surface atom element is prefered.
    Returns:
        adj_mat (np.array): the weighted adjacency matrix of the given ego_graph.
    """

    node_str = get_node_name(ego_ref,start_atom_ele)
    
    graph_node_idx,graph_edge_types_arr = bfs(ego_ref,node_str)
    adj_mat = torch.zeros((len(graph_node_idx),len(graph_node_idx)))

    # radius_arr = [0,0.5,1,1.5,2]
    d_r = 0.5
    radius_arr = np.arange(0, radius+d_r, d_r)

    ego_arr = []
    
    ego_arr = [get_subgraph_ri_from_graph(ego_ref,r_i) for r_i in radius_arr]
    
    
    for (idx,node) in enumerate(graph_node_idx):
        neighbors = ego_ref.neighbors(node)
        degree = ego_ref.degree(node)
        
        for neighbor in neighbors:
            edge_weight = check_bond_type(node,neighbor,powr)
            edge_weight = get_curved_weight(ego_arr,[node,neighbor],edge_weight)
            idx_col = graph_node_idx.index(neighbor)
            adj_mat[idx,idx_col] = edge_weight

    return adj_mat

def get_graph_spectra(struc_fname,powr,huckel=True,start_atom_ele='Pt'):
    """

    get the eigenvalues of the adjacency matrix of each ego graph in the given configuration

    Args:
        struc_fname (str): path to the configuration (POSCAR/CONTCAR) of interest
        powr: power in the edge weight assignment formula
        huckel (optional, bool): make the non-zero edge weight negative. Default is TRUE.
        start_atom_ele (optional, str): element of the root node for the bfs search
                                        default is 'Pt', but can changed to any element
                                        that presents in the graph. 
                                        Using the surface atom element is prefered.
    Returns:
        eigvals_arr (torch.tensor): each row of the torch.tensor has 
                                    all the eigenvalues of a ego graph
                                    of the configuration of interest.
    """

    try:
        atoms = read(struc_fname)
    except:
        print('struc_fname:',struc_fname)
        print('problem with ase reading structure')
        raise ValueError
    
    full, egos = get_graphs(atoms,r=radius)
    num_eigvals_per_ego_arr = [egos[i].number_of_nodes() for i in range(len(egos))]
    len_eigvals = min(num_eigvals_per_ego_arr)
    eigvals_arr = torch.zeros((len(egos),len_eigvals))
    for (idx_ego,ego) in enumerate(egos):
        
        mat = get_adj_mat(ego,powr,start_atom_ele=start_atom_ele)
        if huckel:
            dimless_huckel = mat * -1
            eigvals_np = np.linalg.eigvalsh(dimless_huckel.numpy())
            eigvals = torch.tensor(eigvals_np)

            eigvals_arr[idx_ego] = eigvals[:len_eigvals]
        else:
            eigvals_rev_np = np.linalg.eigvalsh(mat.numpy())
            eigvals_rev = torch.tensor(eigvals_rev_np)

            eigvals, idx = torch.sort(eigvals_rev,descending=True)
            eigvals_arr[idx_ego] = eigvals[:len_eigvals]
    
    return eigvals_arr

def get_compare_groups(eigvals_t1,eigvals_t2):
    """

    get all the pairings of eigenvalue vector1 and vector2 with all permutations of elements in vector2

    Args:
        eigvals_t1 (torch.tensor): eigenvalue vector1
        eigvals_t2 (torch.tensor): eigenvalue vector2
    Returns:
        group_t (torch.tensor): pairings of vector1 and all permutations of vector2.
    """

    groups = []
    # n_groups = math.perm(eigvals_t1.size()[0],eigvals_t1.size()[0])
    eigvals_t1_arr = eigvals_t1.numpy()
    eigvals_t2_arr = eigvals_t2.numpy()

    permut = permutations(eigvals_t1_arr, eigvals_t2.size()[0])
    for comb in permut:
        zipped = zip(comb, eigvals_t2_arr)
        groups.append(list(zipped))
    groups=np.array(groups)
    groups_t = torch.tensor(groups)
    return groups_t

def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    """

    Normalized eucledian distance in pytorch.

    Cases:
        1. For comparison of two vecs directly make sure vecs are of size [B] e.g. when using nes as a loss function.
            in this case each number is not considered a representation but a number and B is the entire vector to
            compare x1 and x2.
        2. For comparison of two batch of representation of size 1D (e.g. scores) make sure it's of shape [B, 1].
            In this case each number *is* the representation of the example. Thus a collection of reps
            [B, 1] is mapped to a rep of size [B]. Note usually D does decrease since reps are not of size 1
            (see case 3)
        3. For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
            activations of size D. In the case when D=1 then we have [B, 1] -> [B, 1]. If you meant x1, x2 [D, 1] to be
            two vectors of size D to be compare feed them with shape [D].
            This one is also good for computing the NED for two batches of values. e.g. if you have a tensor of size
            [B, k] and the row is a batch and each entry is the y value for that batch. If a batch is a task then
            it is computing the NED for that task, which is good because that batch has it's own unique scale that
            we are trying to normalize and by doing it per task you are normalizing it as you'd expect (i.e. per task).

    Note: you cannot use this to compare two single numbers NED(x,y) is undefined because a single number does not have
    a variance. Variance[x] = undefined.

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
    https://github.com/brando90/Normalized-Euclidean-Distance-and-Similarity
    """
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps)) ### NEED to test
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size() == torch.Size([x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
        # [B, 1] -> [B]
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5

def ned_max_mag(x1:torch.Tensor,x2:torch.Tensor,dim=1) -> torch.Tensor:
    if len(x1.size()) == 1:
        mag_1 = torch.sqrt(torch.sum(x1**2))
        mag_2 = torch.sqrt(torch.sum(x2**2))
        mag_max = torch.max(mag_1,mag_2)
        x1_n = x1/mag_max
        x2_n = x2/mag_max
        sim_arr = dist(x1_n,x2_n)
    elif x1.size() == torch.Size([x1.size(0), 1]):
        mag_1 = torch.abs(x1)
        mag_2 = torch.abs(x2)
        mag_max = torch.max(mag_1,mag_2)
        x1_n = x1/mag_max
        x2_n = x2/mag_max
        sim_arr = dist(x1_n,x2_n)
    else:
        mag_1 = torch.sqrt(torch.sum(x1**2,1))
        mag_2 = torch.sqrt(torch.sum(x2**2,1))
        mag_1 = torch.reshape(mag_1,(mag_1.size()[0],1))
        mag_2 = torch.reshape(mag_2,(mag_2.size()[0],1))

        comp = torch.cat((mag_1,mag_2),1)
        mag_max = torch.max(comp,dim=1).values
        x1_n = torch.div(x1.T,mag_max).T
        x2_n = torch.div(x2.T,mag_max).T
        sim_arr = dist(x1_n,x2_n)
    
    return sim_arr


def compare_eigval_diff(compare_pair,powr=-1,num_eigvals=5,huckel=True,start_atom_ele='Pt',normalized='ned'):
    """

    similarity between 2 configuration is calculated by 
    1. getting the eigenvals of the adj mat/huckel mat for all ego graphs obtained
       from the 2 configuration subjected to compare.
    2. perform total number of  N_i X N_j Normalized Euclidean distance (NED) calculations 
       between top 5 smallest eigenvals of every ego graph, where N_i is the number 
       of egographs in the i configuration, and the same goes for N_j.
    3. average all the NED as the final similarity between two configuration,
       assuming all comparison has the same weight.

    Args:
        compare_pair (list(str)): path strings of configuration1 (POSCAR/CONTCAR) and configuration2
        powr (optional,float): power in the edge weight assignment. Default is -1
        num_eigenvals (optional,int): number of eigenvalues of a ego graph to use. Default is 5
        huckel (optional,bool): make the non-zero values in the adjacency matrix negative. Default is TRUE
        start_atom_ele (optional, str): element of the root node for the bfs search
                                        default is 'Pt', but can changed to any element
                                        that presents in the graph. 
                                        Using the surface atom element is prefered.
        normalized (optional, str): normalized distance calculate method. Default is 'ned'.
                                    options are: 'mmned': vectors are first normalized by the maximum magnitude among all elements,
                                                          and then standard euclidean distance is computed.
                                                 'cosine': cosine similarity between measure.
                                                 
    Returns:
        similarity (float): similarity score between two configurations
    """
    eigvals_arr0 = get_graph_spectra(compare_pair[0],powr,huckel=huckel,start_atom_ele=start_atom_ele)
    eigvals_arr1 = get_graph_spectra(compare_pair[1],powr,huckel=huckel,start_atom_ele=start_atom_ele)

    eigvals_arr0_top3 = eigvals_arr0[:,:num_eigvals]
    eigvals_arr1_top3 = eigvals_arr1[:,:num_eigvals]

    compare_groups = torch.tensor([])
    
    if eigvals_arr0_top3.size()[0] != eigvals_arr1_top3.size()[0]:
        return np.nan
    else:
        pass
    
    compare_groups = get_compare_groups(eigvals_arr0_top3,eigvals_arr1_top3)
    
    similarity_maxpooled = torch.zeros(compare_groups.size()[0])
    for (idx,group) in enumerate(compare_groups):
        if normalized:
            if normalized == 'ned':
                similarity_arr = ned_torch(group[:,0,:],group[:,1,:])
            elif normalized == 'mmned':
                similarity_arr = ned_max_mag(group[:,0,:],group[:,1,:])
            elif normalized == 'cosine':
                similarity_arr = cos_sim(group[:,0,:],group[:,1,:])
        else:
            print('Warning: no normalization performed to compute the distance between eigenval vectors')
            similarity_arr = dist(group[:,0,:],group[:,1,:])
        similarity_avg = torch.mean(similarity_arr)
        similarity_maxpooled[idx] = similarity_avg
    
    similarity = float(torch.min(similarity_maxpooled))
    if normalized == 'cosine':
        return similarity
    else:
        return -similarity