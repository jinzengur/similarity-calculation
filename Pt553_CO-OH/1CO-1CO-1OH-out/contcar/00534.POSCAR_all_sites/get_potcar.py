import os
import shutil
import argparse
import glob

parser = argparse.ArgumentParser(
    description='command lines for creating POTCAR')

parser.add_argument('-potcar',type=str,help='path to the POTCAR database folder')
parser.add_argument('-conf_dir',type=str,help='path to the conf folder')
args = parser.parse_args()

print(args.conf_dir)
a_conf = glob.glob(args.conf_dir+'/*')[0]

with open(a_conf,'r') as file:
    conf_contents = file.readlines()

at_info_str = conf_contents[0]
at_info = [i for i in at_info_str.split(' ') if i not in ['',' ','\n']]

potcar_contents = []
for at in at_info:
    print(at)
    potcar_at_path = args.potcar+f'/{at}/POTCAR'
    with open(potcar_at_path,'r') as file:
        potcar_at_contents = file.readlines()
    potcar_contents.extend(potcar_at_contents)

vasp_f_path = os.path.dirname(__file__)
with open(f'{vasp_f_path}/POTCAR','w') as file:
    file.writelines(potcar_contents)