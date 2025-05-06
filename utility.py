import numpy as np
import glob
from surfgraph.helpers import draw_atomic_graphs
import networkx as nx
import ase
from ase.io import read
from ase.visualize import view
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import shutil
import re
from surfgraph.chemical_environment import process_atoms
from surfgraph.chemical_environment import compare_chem_envs
from surfgraph.site_detection import generate_normals_original as generate_normals
from surfgraph.site_detection import generate_site_type
from surfgraph.site_detection import generate_site_graphs
from numpy.linalg import norm
import ase
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize import view
from ase.io import read
from ase.constraints import FixAtoms,constrained_indices

ad_elements=['C', 'O', 'N', 'H']
surf_elements=['Pt','Sn','Ru']
grid = [1,1,0]
radius = 2
cutoff = 1.1
prefactor = 10

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = list(mcolors.CSS4_COLORS.values())
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["figure.dpi"] = 300
plt.rcParams['axes.linewidth'] = 5
fontsize=35
mk_size = 100
tk_width = 5
figsize = (9,6)

def draw_ego_graph(full,ego_arr,atoms):
    """
    draw the ego graphs
    Args:
        full (nx.graph): full graph
        ego_arr (list(nx.graphs)): ego graphs
        atoms (ase.Atoms): ase.Atoms object
    """
    labels = [None]*len(ego_arr)
    for index, graph in enumerate(ego_arr):
        labels[index] = {node:str(len([edge for edge in full[node] if edge.split(":")[0] not in ad_elements])) for node in graph.nodes()}
        # print(labels[index])
    draw_atomic_graphs(ego_arr, atoms=atoms, labels=labels)

def get_unique_node_arr(node_arr):
    """
    get the non-duplicate nodes names
    Args:
        node_arr (list)
    Returns:
        node_name_arr (list)
    """

    node_name_arr=[]
    node_id_unique=[]
    for n in node_arr:
        at_id = n.split('[')[0]
        if at_id in node_id_unique:
            continue
        else:
            node_name_arr.append(n)
            node_id_unique.append(at_id)
    return node_name_arr

def get_node_name_arr(graph,ele_name):
    """
    get the node names that match with the given element name in the given ego graph 
    Args:
        graph (nx.graph): an ego graph
        ele_name (str): element name. E.g. Pt
    Returns:
        node_str_arr (list): list of node names
    """
    node_str_arr=[]
    for n in graph.nodes():
        at_name = n.split(':')[0]
        if at_name == ele_name:
            node_str_arr.append(n)
    
    return node_str_arr

def make_dataframe(data_arr,col_name_arr,csv_path=False):
    """
    create a pd.dataframe from list
    Args:
        data_arr (list): a list with number of columns >= 1
        col_name_arr (list(str)): a list of column names to be assigned in the dataframe
    Returns:
        df (pd.dataframe): a dataframe 
    """
    temp_dict = {}
    for (idx,col) in enumerate(col_name_arr):
        temp_dict[col] = data_arr[idx]
    
    df = pd.DataFrame(data=temp_dict)

    if csv_path:
        df.to_csv(csv_path)
    return df

def ploting(results_df,interval=False):
    """
    plot the similarity profile

    Args:
        results_df (pandas.dataframe): dataframe including the similarity scores of configuration pairs
        interval (optional, list or bool): plot the similarity scores profile within a range. i.e. [score_min, score_max].
                                           default is False, meaning no interval and plot everything. 
    """

    plt.figure(figsize=figsize,linewidth=20)
    ax = plt.axes()
    x_rot = 25
    if interval:
        x_rot = 0
    else:
        pass
    ax.tick_params(axis='x',which='major',direction='out',length=10,width=tk_width,color='black',pad=2,labelsize=fontsize
                   ,labelcolor='blue',labelrotation=x_rot)
    ax.tick_params(axis='y',which='major',direction='out',length=10,width=tk_width,color='black',pad=2,labelsize=fontsize
                   ,labelcolor='blue',labelrotation=0)
    sim_val_arr = np.array([])
    labels = np.array([])
    if interval:
        df_temp = results_df.loc[(results_df['similarity']>interval[0]) & (results_df['similarity']<interval[1])]
        sim_val_arr = df_temp['similarity'].to_numpy()
        labels = df_temp['cluster'].to_numpy()
    else:
        sim_val_arr = results_df['similarity'].to_numpy()
        labels = results_df['cluster'].to_numpy()
    
    c_arr = [colors[int(-i)+22] for i in labels]
    plt.scatter(x=np.arange(len(sim_val_arr)),y=sim_val_arr,c=c_arr,s=mk_size)
    plt.ylabel('Similarity score',fontsize=fontsize,weight='bold')
    plt.xlabel('Configuration comparison',fontsize=fontsize,weight='bold')
    # plt.xticks(np.arange(0,len(sim_val_arr),50),fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(alpha=0.1,lw=tk_width)
    plt.show()

    return

def cluster_detect(csv_path):
    """
    Use DBSCAN to detect and label the similarity scores clusters

    Args:
        csv_path (str): path to the csv file containing the similarity scores of configuration pairs of interest.
    Returns:
        new_results_df (pandas.dataframe): dataframe that has the label of each similarity score
                                           according to the cluster it belongs.  
    """
    
    results_df = pd.read_csv(csv_path).dropna()
    similarity = results_df['similarity'].to_numpy()

    range_sim_all = max(similarity) - min(similarity)
    eps_limit_all = range_sim_all/500
    min_n = 2
    if len(similarity) >= 1000:
        min_n = 5
        eps_limit_all = range_sim_all/500
    else:
        eps_limit_all = range_sim_all/100
    
    dbscan_model_all = DBSCAN(eps=eps_limit_all,min_samples=min_n)
    clustering = dbscan_model_all.fit(similarity.reshape(-1,1))
    cluster_labels = clustering.labels_
    
    new_results_df = pd.DataFrame(results_df)
    new_results_df['cluster'] = cluster_labels
    
    return new_results_df

def get_ini_conf_path_from_outcar_path(contcar_path):
    """
    get the path string of the un-relaxed configuration from the relaxed configuration path string

    Args:
        contcar_path (str): relaxed configuration (typically the CONTCAR file) path string.
    Returns:
        conf_path (str): un-relaxed configuration (typically the POSCAR file as a vasp input) path string.
    """
    
    path_arr = contcar_path.split('/')
    dir_path = '/'.join(path_arr[:-3])
    conf_path = dir_path+'/unique_dir/' + path_arr[-2]

    return conf_path

def view_structure(conf_arr,unrelaxed=False):
    """
    use ase to visualize the structures

    Args:
        conf_arr (list(str)): list of path strings of the relaxed configurations of interest.
        unrelaxed: (optional, bool): view the un-relaxed configurations corresponding to the relaxed configurations.
                                     Default is FALSE, meaning visualize the relaxed configurations of interest.
    """

    movie = []
    for conf in conf_arr:
        if unrelaxed:
            conf = get_ini_conf_path_from_outcar_path(conf)
        else:
            pass
        atoms = read(conf)
        movie.append(atoms)
    
    view(movie)
    return

def get_conf_info(struc_fname,start_atom_ele='N'):
    # struc_fname = 'raw_files/4NO-cont-pos-out/unique_dir/NO4_POSCAR_0236_poscars'
    atoms = read(struc_fname)
    full, egos = sim.get_graphs(atoms)
    node_arr = get_unique_node_arr((np.concatenate(([get_node_name_arr(ego,start_atom_ele) for ego in egos]))))
    node_arr = sorted(node_arr)
    conf_type_dict={1:'top',2:'bridge',3:'hollow'}
    conf_ego_arr=[]
    for node in node_arr:
        for ego in egos:
            try:
                neighbors = ego.neighbors(node)
                at_ele_arr = [at_str.split(':')[0] for at_str in neighbors]
                surf_at_arr = [at for at in at_ele_arr if at in surf_elements]
                connectivity = len(surf_at_arr)
                conf_ego_arr.append(conf_type_dict[connectivity])
                break
            except:
                continue
    conf_info_str = '-'.join(conf_ego_arr)
    return conf_info_str

def make_unique_dir(unique_conf_arr,dir_name):
    """
    make a dir that contains the un-relaxed configurations of the given non-similar relaxed configurations
    
    Args:
        unique_conf_arr (list): list containing the path strings of the non-similar relaxed configurations
    Returns:
        unique_dir_conf_path (list): list containing the path strings of the un-relaxed configurations
                                     of the given non-similar relaxed configurations.
    """
    
    samp_conf_path = unique_conf_arr[0]
    path_info_arr = samp_conf_path.split('/')
    path = '/'.join(path_info_arr[:-3])
    print(path)
    
    unique_dir_path = path + f'/{dir_name}/'

    init_conf_path_arr = glob.glob('/'.join(path_info_arr[:-3])+'/unique_dir/*')

    unique_conf_info_arr = [unique_conf.split('/') 
                            for unique_conf in unique_conf_arr]

    ### use initial condition for unique_unique configurations
    unique_init_conf_path_arr = []
    for init_conf_path in init_conf_path_arr:
        conf_name = init_conf_path.split('/')[-1]
        for unique_conf_info in unique_conf_info_arr:
            if conf_name in unique_conf_info:
                unique_init_conf_path_arr.append(init_conf_path)
        else:
            pass
    unique_conf_arr = unique_init_conf_path_arr

    if os.path.isdir(unique_dir_path):
        shutil.rmtree(unique_dir_path)
        os.mkdir(unique_dir_path)
    else:
        os.mkdir(unique_dir_path)
    
    for conf in unique_conf_arr:
        
        name = conf.split('/')[-1] 
        shutil.copyfile(conf, f'{unique_dir_path}/{name}')

    unique_dir_conf_path = glob.glob(unique_dir_path+'/*') 
    return unique_dir_conf_path

def get_energy(outcar_path):
    """
    get the energy value from the vasp output OUTCAR file

    Args:
        outcar_path (str): path string of the OUTCAR file of interest.
    Returns:
        energy_val (float): energy value reported in the OUTCAR file.
    """
    energy_file_path = os.path.dirname(outcar_path)+'/energies.txt'
    with open(energy_file_path,'r') as file:
        contents = file.readlines()
    energy_info = ''
    for line in contents:
        if 'energy  without entropy' in line:
            energy_info = line
        
    energy_val_str = re.findall(r"[-+]?(?:\d*\.*\d+)",energy_info)[0]
    energy_val = float(energy_val_str) 
    return energy_val

def match_outcar_with_poscar(outcar_path,conf_path_arr):
    """
    find the corresponding relaxed configuration CONTCAR path from the given OUTCAR path

    Args:
        outcar_path (str): OUTCAR path string.
        conf_path_arr (list(str)): list of the CONTCAR path strings
    Returns:
        conf_path_str: path string of the directory containing the CONTCAR 
    """

    outcar_dir = os.path.dirname(outcar_path).split('/')[-1]
    
    conf_path_info_arr = [conf_path.split('/') for conf_path in conf_path_arr]

    for conf_path_info in conf_path_info_arr:
        
        if outcar_dir in conf_path_info:
            
            break
    
    conf_path_str = '/'.join(conf_path_info)
    return conf_path_str

def get_energys(conf_path_arr,plot=False):
    """
    get the energy value array from the given configuration path string array

    Args:
        conf_path_arr (list(str)): array of path strings of the relaxed configurations of interest.
        plot (optional, bool): plot the energy value profile and distribution. Default is False, meaning no plots.
    Returns:
        energy_df_sorted (pandas.dataframe): dataframe containing the relaxed configurations of interest and the 
                                             corresponding energy values.
    """

    outcars = [glob.glob(os.path.dirname(conf_path)+'/energies.txt')[0] for conf_path in conf_path_arr]

    energy_arr = []
    conf_info_arr = []

    for outcar in outcars:
        energy_val = get_energy(outcar)
        conf_info = match_outcar_with_poscar(outcar,conf_path_arr)
        conf_info_arr.append(conf_info)
        energy_arr.append(energy_val)

    dict = {'conf_name':conf_info_arr,'energy':energy_arr}
    energy_df = pd.DataFrame(dict)
    min_energy = energy_df['energy'].min()
    print(energy_df[energy_df['energy'] == min_energy]['conf_name'],min_energy)

    energy_df_sorted = energy_df.sort_values('energy')

    if plot:
        plt.figure()
        plt.scatter(x=np.arange(len(energy_arr)),y=np.sort(energy_arr),label='sorted')
        plt.xticks(np.arange(len(energy_arr)))
        plt.xlabel('Index')
        plt.ylabel('Energy (eV)')
        plt.grid()
        plt.legend()
    
        plt.figure()
        plt.hist(energy_arr,bins=10)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Probability density')
        plt.legend()
        plt.grid()
    
    return energy_df_sorted

def get_conf_path_arr_from_csv(csv_path):
    """
    get the array of configurations from the similarity csv file

    Args:
        csv_path (str): path string to the similarity csv file.
    Return:
        conf_all_arr (np.array(str)): array of path strings of the configurations in the similarity csv file. 
    """

    df = pd.read_csv(csv_path).dropna()
    conf1_arr = df['config1'].to_numpy()
    conf2_arr = df['config2'].to_numpy()

    conf_all_arr = np.concatenate((conf1_arr,conf2_arr))
    conf_all_arr = np.unique(conf_all_arr)
    
    return conf_all_arr

def get_unique_evolution(csv_path,e_cutoff=0.3,n_cluster=1,plot=False,show_struc=False):
    """
    applying Evo-Sim to get the non-similar relaxed configurations

    Args:
        csv_path (str): path string of the similarity csv file.
        e_cutoff (optional, float): energy cutoff to use in the evolutionary algorithm.
                                    Default is 0.3 eV.
        n_cluster (optional, int): number of topmost clusters to be used in the EvoSim. 
                                   Configuration pairs within these clusters are considered highly similar.
                                   Default is 1, as the first topmost cluster cluster always contains configuration pairs
                                   sharing same binding sites.
        plot (optional, bool): plot the energy profile and energy distribution.
        show_struc (optional, bool): use ase to visualize the non-similar relaxed configurations
    Returns:
          conf_arr_left (numpy.array(str)): array of path strings of the identified non-similar relaxed configurations.
    """
    
    conf_path_arr = get_conf_path_arr_from_csv(csv_path)
    energy_df = get_energys(conf_path_arr,plot=plot)
    energy_min = energy_df['energy'].min()
    energy_df_reduced = energy_df[energy_df['energy'] <= energy_min+e_cutoff]
    print(energy_df['conf_name'].to_numpy(),len(energy_df['conf_name'].to_numpy()))

    new_results_df = cluster_detect(csv_path)

    cluster_idx_arr_all = get_sorted_cluster_idx_arr(new_results_df)
    print(cluster_idx_arr_all)
    cluster_label_check = [cluster_idx_arr_all[-(i+1)] for i in range(n_cluster)]

    conf_arr = energy_df_reduced['conf_name'].to_numpy()
    
    print(f'confs left after evolutionary with {e_cutoff}eV cutoff')
    print(conf_arr,len(conf_arr))
    conf_arr_left = []
    conf_arr_not_want = []
    for (idx,conf_i) in enumerate(tqdm(conf_arr[:-1])): # enumerate

        if conf_i in conf_arr_not_want:
            continue
        else:
            pass

        conf_arr_left.append(conf_i)

        for conf_j in conf_arr[idx+1:]:

            if conf_j in conf_arr_not_want:
                continue
            else:
                pass
            
            info_df = new_results_df.loc[(new_results_df['config1'].isin([conf_i,conf_j])) 
                                         & (new_results_df['config2'].isin([conf_i,conf_j]))]
            try:
                cluster = info_df['cluster'].iloc[0]
            except:
                cluster = None
            if cluster in cluster_label_check:
                conf_arr_not_want.append(conf_j)
            else:
                conf_arr_left.append(conf_j)
    
    conf_arr_left = [i for i in np.unique(conf_arr_left) if i not in conf_arr_not_want]         

    if show_struc:
        view_structure(conf_arr_left)

    return conf_arr_left

def get_sorted_cluster_idx_arr(results_df):
    """
    sort the cluster label in ascending order according the similarity score magnitude

    Args:
        results_df (pandas.dataframe): dataframe with similarity scores labeled according to the clustering information.
    Returns:
        cluster_idx_arr_all (numpy.array): numpy array with cluster labels in ascending order according the similarity score magnitude.
    """

    sim_score_arr = []
    cluster_idx_arr = []
    cluster_arr = np.unique(results_df['cluster'].to_numpy())
    cluster_n = len(cluster_arr)
    for i in range(0,cluster_n):
        cluster_i = cluster_arr[i]
        sim_val_arr = results_df[results_df['cluster'] == cluster_i]['similarity'].to_numpy()
        sim_val_avg = np.average(sim_val_arr)
        
        sim_score_arr.append(sim_val_avg)
        cluster_idx_arr.append(cluster_i)
    
    sim_cluster_df = pd.DataFrame(list(zip(sim_score_arr, cluster_idx_arr)),columns = ['similarity', 'cluster_idx'])
    sim_cluster_df_sorted = sim_cluster_df.sort_values('similarity')
    cluster_idx_arr_all = sim_cluster_df_sorted['cluster_idx'].to_numpy()

    return cluster_idx_arr_all

def get_coverage_from_csv_path(csv_path):

    f_name = csv_path.split('/')[-1] 
    info = f_name.split('_')
    coverage_str = info[1]
    coverage = int(coverage_str[0])

    return coverage

def ploting_cluster_info(results_df,fname):

    colors = list(mcolors.CSS4_COLORS.values())
    plt.figure()
    sim_val_arr = results_df['similarity'].to_numpy()
    labels = results_df['cluster'].to_numpy()
    for i in tqdm(range(0,len(sim_val_arr))):
        
        sim_val = sim_val_arr[i]
        # x = np.arange(len(sim_val))
        color_idx = int(labels[i])
        plt.scatter(i,sim_val,c=colors[color_idx])

    plt.ylabel('similarity')
    plt.xlabel('index')
    plt.title(fname)
    plt.grid()

    return

def get_sorted_cluster_idx_arr(results_df):

    sim_score_arr = []
    cluster_idx_arr = []
    cluster_arr = np.unique(results_df['cluster'].to_numpy())
    cluster_n = len(cluster_arr)
    for i in range(0,cluster_n):
        cluster_i = cluster_arr[i]
        sim_val_arr = results_df[results_df['cluster'] == cluster_i]['similarity'].to_numpy()
        sim_val_avg = np.average(sim_val_arr)
        
        sim_score_arr.append(sim_val_avg)
        cluster_idx_arr.append(cluster_i)
    
    sim_cluster_df = pd.DataFrame(list(zip(sim_score_arr, cluster_idx_arr)),columns = ['similarity', 'cluster_idx'])
    sim_cluster_df_sorted = sim_cluster_df.sort_values('similarity')
    cluster_idx_arr_all = sim_cluster_df_sorted['cluster_idx'].to_numpy()

    return cluster_idx_arr_all

def cluster_detect(csv_path):
    
    results_df = pd.read_csv(csv_path).dropna()
    similarity = results_df['similarity'].to_numpy()

    range_sim_all = max(similarity) - min(similarity)
    eps_limit_all = range_sim_all/500
    min_n = 2
    if len(similarity) >= 1000:
        min_n = 5
        eps_limit_all = range_sim_all/500
    else:
        eps_limit_all = range_sim_all/100
    
    dbscan_model_all = DBSCAN(eps=eps_limit_all,min_samples=min_n)
    clustering = dbscan_model_all.fit(similarity.reshape(-1,1))
    cluster_labels = clustering.labels_
    
    new_results_df = pd.DataFrame(results_df)
    new_results_df['cluster'] = cluster_labels
    
    return new_results_df

def get_reduced_df(csv_path,n_cluster=1):

    results_df = cluster_detect(csv_path)

    cluster_idx_arr_all = get_sorted_cluster_idx_arr(results_df)
    print(cluster_idx_arr_all)
    cluster_label_check = [cluster_idx_arr_all[-(i+1)] for i in range(n_cluster)]

    reduced_df = results_df[results_df['cluster'].isin(cluster_label_check)]
    
    return reduced_df

def get_ad_from_ego0(ego0,metal_nodes_arr):

    ad = ''

    try:
        neighbor_nodes = ego0.neighbors(metal_nodes_arr[0])
        for node in neighbor_nodes:
            if node.split(':')[0] in ad_elements:
                ad = node
    except:
        node_list = [node for node in ego0.nodes]
        ad = node_list[0]

    ad_ele = ad.split(':')[0]
    
    return ad_ele

def count_neighbor_metal_of_ad(ego0,ad_nodes_arr):
    
    num_metal_node_arr = []
    result_arr = []
    for ad_node in ad_nodes_arr:
        neighbor_nodes = ego0.neighbors(ad_node)
        num_neighbor_metal_nodes = len([node for node in neighbor_nodes if node.split(':')[0] in surf_elements])
        if num_neighbor_metal_nodes != 0:
            ad_ele = ad_node.split(':')[0]
            site_type = ''
            if num_neighbor_metal_nodes == 1:
                site_type = 'Top'
            elif num_neighbor_metal_nodes == 2:
                site_type = 'Bridge'
            elif num_neighbor_metal_nodes == 3:
                site_type = 'Hollow'
            else:
                site_type = 'Abnormal'
            result_arr.append(f'{ad_ele}-{site_type}')
    
    return result_arr

def get_site_info_from_graph(ego_0,full):

    metal_nodes_ego_0 = [node for node in ego_0.nodes if node.split(':')[0] in surf_elements]
    ad_nodes_ego_0 = [node for node in ego_0.nodes if node.split(':')[0] in ad_elements]

    ad_ele_id_dict = {'N':10,'C':20,'O':30}

    site_info = 0

    if len(metal_nodes_ego_0) == 0: # deteched
        site_kind = 'Deteched'
        ad_ele = get_ad_from_ego0(ego_0,metal_nodes_ego_0)
        
        site_info = f'{ad_ele}-{site_kind}'

    else:
        result_arr = count_neighbor_metal_of_ad(ego_0,ad_nodes_ego_0)
        site_info = '!'.join(result_arr)
    
    return site_info

def get_site_info(struc_fname):

    atoms = read(struc_fname)
    full, egos = sim.get_graphs(atoms)

    site_info_arr = []

    for (idx_ego,ego) in enumerate(egos):
        ego_r0 = sim.get_subgraph_ri_from_graph(ego,0)
        
        site_info = get_site_info_from_graph(ego_r0,full)

        site_info_arr.append(site_info)
    
    site_info_str = '|'.join(site_info_arr)
    return site_info_str

def get_bind_conf_from_graph(ego0):
    
    heavy_ad_atom = ['C','N','O']
    ad_nodes_ego_0 = [node for node in ego0.nodes if node.split(':')[0] in heavy_ad_atom]

    all_metal_nodes_arr = []
    for ad_node in ad_nodes_ego_0:
        neighbor_nodes = ego0.neighbors(ad_node)
        for node in neighbor_nodes:
            if node.split(':')[0] in surf_elements:
                all_metal_nodes_arr.append(node)
    meta_nodes_arr = np.unique(all_metal_nodes_arr)
    bind_conf_info = ''
    if len(meta_nodes_arr) == 1:
        bind_conf_info = 'pi'
    elif len(meta_nodes_arr) == 2:
        bind_conf_info = 'sigma'
    else:
        bind_conf_info = 'bibilabu'
    
    return bind_conf_info

def get_bind_conf_info(struc_fname):
    
    atoms = read(struc_fname)
    bind_conf_arr = []
    full, ego0_arr = sim.get_graphs(atoms,r=0)
    for (idx_ego,ego0) in enumerate(ego0_arr):
        bind_conf = get_bind_conf_from_graph(ego0)
        bind_conf_arr.append(bind_conf)
    bind_conf_info_str = '|'.join(bind_conf_arr)
    return bind_conf_info_str

def add_bind_conf_info_to_dft_energy_csv(csv_or_df):

    if type(csv_or_df) == str:
        df = pd.read_csv(csv_or_df)
    elif isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df
    else:
        raise ValueError('Input should be either the path to energy csv or the energy dataframe')

    conf_path_arr = df['conf_name'].to_numpy()
    
    bind_conf_info_arr = [get_bind_conf_info(conf_path) for conf_path in tqdm(conf_path_arr)]
    new_df = pd.DataFrame(df)
    new_df['bind_conf'] = bind_conf_info_arr

    return new_df

def add_conf_info_to_dft_energy_csv(csv_or_df):
    
    if type(csv_or_df) == str:
        df = pd.read_csv(csv_or_df)
    elif isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df
    else:
        raise ValueError('Input should be either the path to energy csv or the energy dataframe')

    conf_path_arr = df['conf_name'].to_numpy()
    
    site_info_arr = [get_site_info(conf_path) for conf_path in tqdm(conf_path_arr)]
    new_df = pd.DataFrame(df)
    new_df['site_info'] = site_info_arr

    return new_df

def get_contcar_from_isomor_outcar(isomor_outcar_dir_path):
    """
    
    get the CONTCAR in the configuration directory (vasp calculation directory) from the isomorphism checking outputed path string

    Args:
        isomor_outcar_dir_path (list(str)): list of path strings of configurations in the isomorphism checking output directory

    Returns:
         contcar_dirs_isomor (str): list of path strings of configurations in the configuration directories
         contcar_outcar_dict (dict): dictionary contains the path strings of configurations in the configuration directories and corresponding isomorphism checking output directory.
    """

    isomor_outcar_arr = glob.glob(f'{isomor_outcar_dir_path}/*')

    if len(isomor_outcar_arr)==0:
        raise ValueError('Nothing was found, check the file path!')
    
    isomor_outcar_name_arr = [isomor_outcar_path.split('/')[-1] 
                              for isomor_outcar_path in isomor_outcar_arr]
    
    isomor_outcar_dir_path_arr = isomor_outcar_dir_path.split('/')
    contcars_path = '/'.join(isomor_outcar_dir_path_arr[:-1]) + '/*POSCAR*/CONTCAR'
    contcars_path_arr = glob.glob(f'{contcars_path}')

    contcar_dirs_isomor = []
    contcar_outcar_dict = {}
    for idx_c,contcar_path in enumerate(contcars_path_arr):
        contcar_dir_name = contcar_path.split('/')[-2]
        for idx_o,isomor_outcar_name in enumerate(isomor_outcar_name_arr):
            if contcar_dir_name in isomor_outcar_name:
                contcar_dirs_isomor.append(contcar_path)
                contcar_outcar_dict[isomor_outcar_arr[idx_o]] = contcars_path_arr[idx_c]

    return contcar_dirs_isomor,contcar_outcar_dict

mult = 1.1
skin = 0.25
radius = 2
grid = [2,2,0]
def make_conf(atoms_filename,ad_file_path,min_dist=2,no_adsorb='',coordination="1,2,3"):
    movie = []
    atoms = read(atoms_filename)
    ads = read(ad_file_path)
##########################################################################
    nl = NeighborList(natural_cutoffs(atoms, mult), self_interaction=False,  bothways=True, skin=skin)
    nl.update(atoms)

    adsorbate_atoms = [index for index, atom in enumerate(atoms) if atom.symbol in ad_elements]

    normals, mask = generate_normals(atoms,  surface_normal=0.5, adsorbate_atoms=adsorbate_atoms, normalize_final=True)   ### make sure to manually set the normals for 2-D materials, all atoms should have a normal pointing up, as all atoms are surface atoms
    # normals, mask = np.ones((len(atoms), 3)) * (0, 0, 1), list(range(len(atoms)))
    constrained = constrained_indices(atoms)
    mask = [index for index in mask if index not in constrained]
    #for index in mask:
    #    atoms[index].tag = 1
    # print(mask)

    atoms.set_velocities(normals/10)

    all_sites = []

    full_graph, envs = process_atoms(atoms, nl=nl, adsorbate_atoms=adsorbate_atoms, radius=radius, grid=grid) ### here the default radii as well as grid are considered, these can also be added as args.

    center = atoms.get_center_of_mass()

    for coord in [int(x) for x in coordination.split(",")]:
        found_count = 0
        found_sites = generate_site_type(atoms, mask, normals, coordination=coord, unallowed_elements=no_adsorb)

        for site in found_sites:
            all_sites.append(site)
        
        unique_sites = generate_site_graphs(atoms, full_graph, nl, found_sites, adsorbate_atoms=adsorbate_atoms, radius=radius)
        
        for index, sites in enumerate(unique_sites):
            new = atoms.copy()
            best_site = sites[0]
            
            for site in sites[1:]:
                if norm(site.position - center) < norm(best_site.position - center):
                    best_site = site
            #print(best_site.adsorb(new, ads, adsorbate_atoms),args.min_dist)
            ### this check is to ensure, that sites really close are not populated
            if best_site.adsorb(new, ads, adsorbate_atoms) >= min_dist:
                # print(find_ele(new,'C'))
                found_count += 1
                ### if hydrogen bonds exist then use this loop to populate these structures
                #H_bond_movie = orient_H_bond(new)
                H_bond_movie = []
                #print(H_bond_movie[:])
                if len(H_bond_movie) > 0:
                    for i in H_bond_movie:
                        movie.append(i)
                else:
                    movie.append(new)
                # all_unique.append(site)
    return movie