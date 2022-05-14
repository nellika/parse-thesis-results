import os
import numpy as np
import pandas as pd

experiment_set = ['1','2','3', '4']
detectors = ['fast','brief','orb','sift']
batch_sizes = ['512','1024']
experiment_name = 'EXP_2'
coloring = 'dotted'
exp_id_length = 4

directory = f'{experiment_name}/batch_size_{coloring}'

objects=['arm','base','gripper']
models=['200000']

lines = []
count = 0
not_available = []
columns = ['experiment', 'nerf_model', 'pose','dector','batch_size','object', 'metric']
iterations = np.arange(0, 720, 20)
columns.extend(iterations)
full = []

for exps in experiment_set:
    for d in detectors:
        for b in batch_sizes:
            for obj in objects:
                for model in models:
                    path = f'{directory}/{exps}/{d}/{b}/{obj}/'
                    if os.path.exists(path):
                        for filename in os.listdir(path):
                            f = os.path.join(path, filename)
                            log_file = open(f,'r')
                            log_content = log_file.readlines()
                            rotation = log_content[43].replace('[','').replace(']','').replace(' ','').replace('\n','').split(',')
                            translation = log_content[44].replace('[','').replace(']','').replace(' ','').replace('\n','').split(',')
                            loss = log_content[46].replace('[','').replace(']','').replace(' ','').replace('\n','').split(',')
                            exp_start = filename.find('regions')
                            exp_id = f'{exps}_{d}_{obj}_{b}'
                            pose = filename[exp_start+8:exp_start+9]
                            lines.append([*[exp_id,model,pose,d,b,obj],*rotation])
                            lines.append([*[exp_id,model,pose,d,b,obj],*translation])
                            lines.append([*[exp_id,model,pose,d,b,obj,'loss'],*loss])
                            
                            count+=1
                    else: not_available.append(path)

df = pd.DataFrame(lines, columns=columns)
df.to_pickle(f"{experiment_name}/{experiment_name}_batch_size_{coloring}_logs5.pkl")
df.to_pickle(f"{experiment_name}/{experiment_name}_batch_size_{coloring}_logs4.pkl", protocol=4)
print(f'{count} logs parsed, not available: {len(not_available)}\n{not_available}')