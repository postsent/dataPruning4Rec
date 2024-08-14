"""
run commands one by one
"""
import subprocess
from pathlib import Path
import pickle

coreset_mode = ['random', 'entropy', 'aum', 'mean_aum', 'post_aum', 'post_aum_rnn','el2n', 'forgetting'] 
coreset_ratio = [0.6, 0.7, 0.8]
models = ['CORE', 'SRGNN', 'GRU4Rec']
dataset = 'diginetica'
stratified = False
coverage = False
folder = f'./saved_samples/{dataset}'

def init(dict):
    for r in coreset_ratio:
        dict[r] = {}
        for m in models:
            dict[r][m] = 0
    return dict
for mode in coreset_mode:
    fname = f'{folder}/{dataset}-{mode}.pickle'
    my_file = Path(fname)
    assert my_file.is_file() == False, 'file already exist'
    # init
    results = {}
    results = init(results)
    with open(fname, 'wb') as handle:
        pickle.dump(results, handle)
    for ratio in coreset_ratio:
        if mode == 'random':
            f = f'core_random__{ratio}_unbalanced.npy'
        else:
            if stratified:
                f = f"core_stratified_{mode}_{ratio}_kmeans.npy"
            elif coverage:
                f = f"core_{mode}__{ratio}_kmeans.npy"
            else:
                f = f"core_coreset_{mode}_{ratio}_unbalanced.npy"
        
        # TODO: if exist ignore
        for model in models:
            a = subprocess.call(["python", "main_train.py", "-m", model, "-d", dataset, "--subset_file", f, "--coreset_mode", mode, "--coreset_ratio", str(ratio), "--is_run_scripts"])
            print(a)