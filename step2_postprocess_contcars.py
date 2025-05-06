import numpy as np
import os
import sys
import glob
import shutil
import argparse
import ase
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize import view
from ase.io import read
from tqdm import tqdm
from collections import Counter
import re
from surfgraph.chemical_environment import unique_chem_envs
from surfgraph.chemical_environment import process_atoms
from datetime import datetime
from joblib import Parallel, delayed
import similarity as sim
import utility

ad_elements = ['C', 'O', 'N', 'H']
surf_elements=['Pt','Sn','Ru']
overlap_cutoff = 1.5 # cutoff distance in A
radii_multiplier = 1.1
grid_str = '2,2,0'
graph_r = 2
pwd = os.getcwd()

def check_deteched(contcar_path):
    atoms = read(contcar_path)
    # print(comp1_config1)
    full, egos = sim.get_graphs(atoms,r=1)

    for ego in egos:
        nodes_ele_arr = [n.split(':')[0] for n in ego.nodes()]
        intersect = set(nodes_ele_arr).intersection(set(surf_elements))

        if len(list(intersect)) == 0:
            return contcar_path
    
    return

def get_energys(conf_path_arr):

    a_conf_path = conf_path_arr[0]
    a_conf_path_arr = a_conf_path.split('/')

    outcars = [glob.glob(os.path.dirname(conf_path)+'/energies.txt')[0] for conf_path in conf_path_arr]

    # poscars = [glob.glob('/'.join(a_conf_path_arr[:-3])+f'/unique_dir/{conf_path.split('/')[-2]}')[0] 
    #            for conf_path in conf_path_arr]
    # print(outcars)

    energy_arr = []
    conf_info_arr = []

    for outcar in outcars:
        energy_val = get_energy(outcar)
        conf_info = match_outcar_with_poscar(outcar,conf_path_arr)
        # print(outcar,conf_info)
        conf_info_arr.append(conf_info)
        energy_arr.append(energy_val)

    dict = {'conf_name':conf_info_arr,'energy':energy_arr}
    energy_df = pd.DataFrame(dict)
    min_energy = energy_df['energy'].min()
    print(energy_df[energy_df['energy'] == min_energy]['conf_name'],min_energy)

    energy_df_sorted = energy_df.sort_values('energy')
    
    return energy_df_sorted

def adsorbate_formulas_check(ego_graphs, atoms):
    # Jiatong Gui 25oct24
    formulas = []
    for ego in ego_graphs:
        adsorbate_atoms = []
        for node in ego.nodes():
            match = re.search(r':(\d+)', node)
            if match:
                atom_index = int(match.group(1))
                if atoms[atom_index].symbol in ad_elements:
                    adsorbate_atoms.append(atoms[atom_index].symbol)
        formula_count = Counter(adsorbate_atoms)
        formula = ''.join([f"{element}_{count}|" for element, count in formula_count.items()])
        formulas.append(formula)
    for adsorbate in formulas:
        adsorbate_info_raw = adsorbate.split('|')
        adsorbate_info = [i for i in adsorbate_info_raw if i not in ['\n','',' ']]
        ad_formula_dict = {}
        for i in adsorbate_info:
            info_raw = i.split('_')
            info = [i for i in info_raw if i not in ['\n','',' ']]
            ele_name = info[0]
            ele_num = info[1]
            ad_formula_dict[ele_name] = ele_num
        # if adsorbate != "C_1|O_1" or adsorbate != "C_1|O_1" or adsorbate != "HO" or adsorbate != "OH":
        if len(set(adsorbate_info)-set(['C_1','O_1'])) != 0 or len(set(adsorbate_info)-set(['H_1','O_1'])) != 0:
            if 'H' in ad_formula_dict.keys() and 'O' in ad_formula_dict.keys():
                h_count = ad_formula_dict['H']
                o_count = ad_formula_dict['O']
                if h_count == o_count:
                    continue
                else:
                    return False
            else:
                pass
        else:
            pass
    return True

def make_graph(atoms):

    nl = NeighborList(natural_cutoffs(atoms, radii_multiplier), self_interaction=False,
                      bothways=True, skin=0.25)
    nl.update(atoms)
    adsorbate_atoms = [atom.index for atom in atoms if atom.symbol in ad_elements]
    full, chem_env = process_atoms(atoms, nl, adsorbate_atoms=adsorbate_atoms,
                                   radius=graph_r, grid=[int(grid) for grid in grid_str.split(",")],clean_graph=None)  ## get the full graph and the chem_env for all the adsorbates found
    
    return chem_env

def get_unique_conf(confs_path='all_sites/*',energy=False):
    all_atoms = []
    filenames = []
    energies = []
    file_path_arr = glob.glob(confs_path)
    print('convert conf file to atom obj')
    for filename in tqdm(file_path_arr):
        try:
            all_atoms.append(read(filename))
            filenames.append(filename)
        except Exception as e:
            print("{} failed to read".format(filename))
    if energy:
        print('get energies')
        for filename in tqdm(file_path_arr):
            energies.append(utility.get_energy(filename))
    else:
        print('no energy information')
        for filename in tqdm(file_path_arr):
            energies.append(float('inf'))

    chem_envs = []
    
    try:
        chem_envs = Parallel(n_jobs=-2)(delayed(make_graph)(i)
                                    for i in tqdm(all_atoms))
    except:
        chem_envs = [make_graph(i) for i in tqdm(all_atoms)]
    unique, groups = unique_chem_envs(chem_envs, list(zip(energies, filenames)))
        
    ###### This next condition, finds the unique configs amongst the OUTCARS/ atoms object provided and arranges them according to ascending order of energies

    print("Outputting the lowest energy unique configurations")
    groups = [sorted(group) for group in groups]
    for group in sorted(groups):
        if group[0][0] == float('inf'):
            print("{}: Unknown energy, {} duplicates".format(group[0][1], len(group) - 1))
        else:
            print("{}: {} eV, {} duplicates".format(group[0][1], group[0][0], len(group) - 1))
        
        if os.path.isdir('unique_dir'):
            file_d = group[0][1].split('/')[-2]
            file_a = group[0][1].split('/')[-1]
            shutil.copyfile('./'+group[0][1],'./unique_dir/{}_{}'.format(file_a,file_d))
        else:
            os.makedirs('./unique_dir')
            file_d = group[0][1].split('/')[-2]
            file_a = group[0][1].split('/')[-1]
            shutil.copyfile(group[0][1],'./unique_dir/{}_{}'.format(file_a,file_d))
        for duplicate in group[1:]:
            if duplicate[0] == float('inf'):
                print("-> {}: Unknown energy".format(duplicate[1]))
            else:
                print("-> {}: {} eV".format(duplicate[1], duplicate[0]))

def filter_abnormal_adsorbate(filepath):
    atoms = read(filepath)
    full, egos = sim.get_graphs(atoms, r=0)
    check = adsorbate_formulas_check(egos, atoms)
    if check:
        return
    else:
        return filepath


parser = argparse.ArgumentParser(description="process the DFT results. 1st Do isomorphism, 2nd remove deteched configurations.")
parser.add_argument('-contcar_dir',type=str,help='Path to the contcar directories where DFT calculations were finished')

args = parser.parse_args()

### get into the contcar folder and do isomorphism on all acquired CONTCARs
os.chdir(f'{args.contcar_dir}')

print('extract energy information to txt files')
outcar_arr = glob.glob('*POSCAR*/OUTCAR')
for outcar in tqdm(outcar_arr):
    conf_dir_path = os.path.dirname(outcar)
    with open(outcar,'r') as file:
        contents = file.readlines()
    energy_info1 = ''
    energy_info2 = ''
    for line in contents:
        if 'free  energy   TOTEN' in line:
            energy_info1 = line
        elif 'energy  without entropy' in line:
            energy_info2 = line
    with open(f'{conf_dir_path}/energies.txt','w') as file:
        file.write(energy_info1)
        file.write(energy_info2)

print('doing isomorphism on CONTCARs using analyze_chem_env')

unique_dir_name = 'unique_dir/' 
if os.path.isdir(unique_dir_name):
    shutil.rmtree(unique_dir_name)
else:
    pass
get_unique_conf(confs_path='*POSCAR*/CONTCAR',energy=True)

### Remove the configurations where adsorbates are deteched from the surface.
### Check the deteched configurations by counting surface atoms (metal atoms)
### in the ego graph with r=1. 
### If any metal atom exist in the ego graphs, the configuration is good,
### otherwise, the configuration is a deteched configuration, and it will be removed.

os.chdir(pwd)
print('Remove the deteched adsorbates configurations')
isomorph_contcar_path_arr,contcar_outcar_dict = utility.get_contcar_from_isomor_outcar(f'{args.contcar_dir}/unique_dir')

deteched_conf_arr_i = [check_deteched(contcar_path) 
                     for contcar_path in tqdm(isomorph_contcar_path_arr)]

deteched_conf_arr = [i for i in deteched_conf_arr_i if i is not None]
print('deteched CONTCARs:',deteched_conf_arr)

deteched_outcar_arr = [i for i in contcar_outcar_dict 
                       if contcar_outcar_dict[i] in deteched_conf_arr]
print('deteched OUTCARs:',deteched_outcar_arr)

for outcar_path in deteched_outcar_arr:
    os.remove(outcar_path)

abnormal_ad_conf_arr_i = [filter_abnormal_adsorbate(contcar_path)
                          for contcar_path in tqdm(glob.glob(f'{args.contcar_dir}/unique_dir/*'))]
abnormal_ad_conf_arr = [i for i in abnormal_ad_conf_arr_i if i is not None]
print('abnormal CONTCARs:',abnormal_ad_conf_arr)

for contcar_path in abnormal_ad_conf_arr:
    os.remove(contcar_path)

print('Done')
