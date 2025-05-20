import numpy as np
import os
import sys
import glob
import subprocess
import shutil
import argparse
import ase
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize import view
from ase.io import read
from ase.constraints import FixAtoms,constrained_indices
from surfgraph.chemical_environment import unique_chem_envs
from surfgraph.chemical_environment import process_atoms
from joblib import Parallel, delayed
import similarity as sim
import utility
from tqdm import tqdm

ad_elements = ['C', 'O', 'N', 'H']
overlap_cutoff = 1.5 # cutoff distance in A
radii_multiplier = 1.1
grid_str = '2,2,0'
graph_r = 2

def making_dir(dir_path):

    if os.path.isdir(dir_path):
        print(f'WARNING! ARE YOU SURE ABOUT OVERWRITING {dir_path}?')
        print(f'WARNING! ARE YOU SURE ABOUT OVERWRITING {dir_path}?')
        print(f'WARNING! ARE YOU SURE ABOUT OVERWRITING {dir_path}?')
        judge = input("input yes|no: ")
        if judge == 'yes':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            raise ValueError('stopped')
    else:
        os.mkdir(dir_path)
    
    return

def check_overlap(conf_path):

    atoms = read(conf_path)

    nat_cutoff = ase.neighborlist.natural_cutoffs(atoms)
    cutoff_dict = {atoms.symbols[i]:nat_cutoff[i] for i in range(len(atoms.symbols))}
    # print(cutoff_dict)

    nl = ase.neighborlist.NeighborList(nat_cutoff,self_interaction=False,
                                                  bothways=True, skin=0.25)
    nl.update(atoms)

    distances = atoms.get_all_distances(mic=True)

    for index, atom in enumerate(atoms):
        neighbors, offsets = nl.get_neighbors(index)
        for neighbor in neighbors:
            dist = distances[index][neighbor]
            bond_dist = cutoff_dict[atoms.symbols[index]]+cutoff_dict[atoms.symbols[neighbor]]
            if dist < overlap_cutoff:
                if (atoms.symbols[index] in ad_elements) and (atoms.symbols[neighbor] in ad_elements):
                    # continue
                    if set([atoms.symbols[index],atoms.symbols[neighbor]]) == set(['O','O']):
                        return conf_path
                    else:
                        continue
                else:
                    # print(conf_path)
                    # print(atoms.symbols[index],atoms.symbols[neighbor])
                    # print(atoms[index],atoms[neighbor])
                    return conf_path
    return

def check_overlap_oxides(conf_path):
    
    num_o_oxide = 7
    atoms = read(conf_path)
    adsorbate_atoms_all = [index for index, atom in enumerate(atoms[:-2]) if atom.symbol in ad_elements]
    adsorbate_atoms_all = [index for (idx, index) in enumerate(adsorbate_atoms_all) if idx >= num_o_oxide]
    if len(adsorbate_atoms_all) == 0:
        return
    
    index_to_check = range(len(atoms)-2, len(atoms))
    for index in index_to_check:
        dists = atoms.get_distances(index, adsorbate_atoms_all, mic=True)
        dist = dists.min()
        if dist >= 2:
            pass
        else:
            return conf_path
    
    return

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
            print(filename)
            energies.append(get_energy(filename))
    else:
        print('no energy information')
        for filename in tqdm(file_path_arr):
            energies.append(float('inf'))

    chem_envs = []
    
    # num_cores = multiprocessing.cpu_count() - 1
    # print('num_cores:',num_cores)
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

def do_atom(poscar_path, dist,fix_pos=True):

    poscar_path = os.getcwd() + '/' + poscar_path
    file_l= poscar_path
    x = read(file_l)
    # print(x)
    del x.constraints
    
    c = FixAtoms(indices=[atom.index for atom in x if (atom.position[2] < dist)])
    # print(c)
    x.set_constraint(c)
    file_s = poscar_path#+'_fix'+'.POSCAR'
    x.write(file_s,format='vasp')

parser = argparse.ArgumentParser(description="getting the initial conditions")
parser.add_argument('-ad_file',type=str,help='Path to the adsorbate structure file')
parser.add_argument('-conf_dir',type=str,help='Path to the folder including all configuration files that need to add the adsorbate')
parser.add_argument("--stepped", action="store_true",
                    help="enable a specific feature")

### First construct the new coverage folder based on the input adsorbate and configurations
### then copy vasp calculation files and vasp calc folders construction file
z_coor_constrain = 12.7 # for Pt553
pwd = os.getcwd()
args = parser.parse_args()

if args.stepped:
    print('Have you eyeballed the surface for the z coordinate of the last layer and obtained the z-coor cutoff?')
    judge = input("input yes|no: ")
    if judge == 'yes':
        pass
    else:
        raise ValueError('Get the z-coor cutoff')
else:
    pass

conf_path_info_arr = args.conf_dir.split('/')

mother_dir = conf_path_info_arr[0]

conf_dir = '/'.join(conf_path_info_arr[:-3])

coverage_info = conf_path_info_arr[1].split('-')[:-1]

ad_name = args.ad_file.split('/')[-1].split('.')[0]

new_coverage_info = coverage_info
new_coverage_info.append(f'1{ad_name}')

new_coverage_info.append('out')

new_conf_dir_name = '-'.join(new_coverage_info)

if conf_dir:
    new_conf_dir = f'{conf_dir}/{new_conf_dir_name}/'
else:
    new_conf_dir = f'{conf_path_info_arr[0]}/{new_conf_dir_name}/'

print('New conf dir:',new_conf_dir)
print('Do you agree with this name?')
judge = input("input yes|no: ")
if judge == 'yes':
    pass
else:
    raise ValueError('get the correct file name')

all_site_dir = f'{new_conf_dir}all_sites/'
contcar_dir = f'{new_conf_dir}contcar/'

making_dir(new_conf_dir)
making_dir(all_site_dir)
making_dir(contcar_dir)

### adsorb a new adsorbate onto the surface in the input surface configurations
coordinations = "1,2,3"
no_adsorb=''
min_dist = 2
if ad_name == 'OH' and args.stepped:
    coordinations = "1"

elif ad_name == 'OH':
    coordinations = "1"
    
elif ad_name in ['CO','NO'] and args.stepped:
    coordinations = "1,2"
    no_adsorb = "Sn"
    
elif ad_name in ['CO','NO']:
    coordinations = "1,2,3"
    no_adsorb = "Sn"
    
elif args.stepped:
    coordinations = "1,2"
else:
    pass

# gen_site_process = subprocess.Popen(gen_site_arg_str,shell=True)
# gen_site_process.wait()
print('coordinations:',coordinations)
movie_raw = Parallel(n_jobs=-2)(delayed(utility.make_conf)(i,args.ad_file,no_adsorb=no_adsorb,min_dist=min_dist,coordination=coordinations)
                                       for i in tqdm(glob.glob(args.conf_dir)))
movie = [j for sub in movie_raw for j in sub]
for index, atoms in enumerate(movie):
    atoms.write("./{}/{:05}.{}".format(all_site_dir, index, 'POSCAR'))

### Change working dir into the new conf dir to do isomorph on
### init configurations, and get contcar folder ready.
print('doing isomorphism checking')
new_conf_dir_abs = os.path.abspath(f'{new_conf_dir}')
os.chdir(f'{new_conf_dir}')

get_unique_conf()

os.chdir(pwd)

### making contcar dirs for DFT calculations
print(f'making dirs in {new_conf_dir_name}contcar/ for DFT calculations')
os.chdir(f'{contcar_dir}')

unique_conf_arr = glob.glob(f'../unique_dir/*')

for conf_path in tqdm(unique_conf_arr):
    fname = os.path.basename(conf_path)
    destination_dir = f'{fname}/'
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = f'{destination_dir}/POSCAR'
    shutil.copy2(conf_path, destination_path)

print('Check POSCAR in contcar/ for atom overlap.')
poscars_all_arr = glob.glob(f'*POSCAR*/POSCAR')
wrong_poscars_arr = [check_overlap(path) for path in tqdm(poscars_all_arr)]
wrong_conf_arr = [os.path.dirname(i) for i in wrong_poscars_arr if i is not None]
print('wrong_conf_arr:',wrong_conf_arr)
for dir_path in np.unique(wrong_conf_arr):
    shutil.rmtree(dir_path)
os.chdir(pwd)

### for stepped surfaces, unconstrain all atoms
### except the bottom-most layer Metal atoms.
### the z coordinate cutoff need to eyeball!!!
os.chdir(mother_dir)
if args.stepped:
    print('constrain the bottom most layer')
    poscar_path_arr = glob.glob(f"{new_conf_dir_name}/contcar/*POSCAR*/POSCAR")
    for poscar_path in poscar_path_arr:
        do_atom(poscar_path,z_coor_constrain)
else:
    pass

print('Done!')
