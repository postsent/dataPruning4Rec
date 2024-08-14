import torch
import numpy as np
from recbole.utils import ModelType, set_color
from recbole.sampler import KGSampler
from recbole.data.dataloader import *

from recbole.data.utils import load_split_dataloaders, create_samplers, get_dataloader, save_split_dataloaders
import random
from tqdm import tqdm
import torch.nn.functional as F

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import get_model
import pickle
import os

def my_subsample(interaction, subset_file):
    """subsample interaction based on session_id 
    train_dataset.inter_feat['session_id'].shape | 634088 
    train_dataset.inter_feat['item_id_list'].shape | 634088, 69
    train_dataset.inter_feat['item_id'].shape | 634088                  # label
    train_dataset.inter_feat['item_length'].shape | 634088
    train_dataset.inter_feat['graph_idx'].shape | 634088
    """ 
    ori_len = interaction.length       
    if type(subset_file) == str:
        if subset_file == 'random':
            perc = 0.1
            lst = [i for i in range(ori_len)]
            random.shuffle(lst)
            train_session_id_subset = torch.tensor(lst[:int(ori_len*perc)])
        else:
            train_session_id_subset =  torch.from_numpy(np.load(subset_file)) 
    else: 
        train_session_id_subset = np.array(subset_file)
        
    for key in interaction.columns:
        interaction[key] = interaction[key][train_session_id_subset]
    
    interaction.length = train_session_id_subset.shape[0]
    logger = getLogger()
    logger.info('\n')
    logger.info(f"Load subsets index from: {subset_file}")
    logger.info(f"Load {round(interaction.length/ori_len*100, 2)} % #train_sessions, {ori_len}->{interaction.length} \n")

def data_preparation(config, dataset, subset_file=None, valid_loader=False):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
         # [INFO]: subsample train_dataset, subsample after dataset.build for recbole GNN & normal recbole package
        if type(subset_file) == list or subset_file is not None:
            my_subsample(train_dataset.inter_feat, subset_file)
        
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "evaluation")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "evaluation")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )
    if valid_loader:
        valid_data = get_dataloader(config, "train")(
                config, valid_dataset, train_sampler, shuffle=config["shuffle"]
        )
        return train_data, valid_data

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data

def load_model(model_file, model, device):
    assert os.path.exists(model_file), 'Saved model File not exist'
    checkpoint_file = model_file 
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    print(f"Loading model [{type(model).__name__}] from: {checkpoint_file}")
    return model 

def load_ensemble_pretrained_models(args):
    """recbole & recbole GNN"""
    expert_list = []
    is_load_data = False
    
    for model_name in tqdm(args.ensembles, total=len(args.ensembles), ncols=100):
        config = Config(model=model_name, dataset=args.dataset, config_file_list=[args.config_overall_pth])
        if not is_load_data: # all base models use sequential dataset
            dataset = create_dataset(config)
            train_data, _, _ = data_preparation(config, dataset)
            is_load_data = True
        
        arch = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        
        saved_pth = f'{args.td_folder}/{model_name}-{args.dataset}.pth' # TODO efficient ensemble
        if model_name == 'CORE':
            saved_pth = f'{args.td_folder}/{model_name}.pth' 
        logger = getLogger()
        logger.info(f'Load model from: {saved_pth}')

        model = load_model(saved_pth, arch, config["device"])
        expert_list.append(model)

    return expert_list


def metrics(args, outputs, targets): # TODO
    """recall@20"""
    output_softmax = F.softmax(outputs, dim=1)
    correctness = []
    top_values, top_indices = torch.topk(output_softmax, k=args.metrics_topk)
    for top_idx, tar in zip(top_indices.detach().cpu().numpy(), targets.detach().cpu().numpy()):
        correctness.append(1 if np.isin(tar, top_idx) else 0)
    return correctness

def get_metric(args, ensemble, model, iter_data, config):
    """hit"""
    # 0 in np.array([i['item_id'].tolist() for i in iter_data]).flatten()[0] : False, target start from 1
    correctness = []
    for batch_idx, interaction in enumerate(iter_data):
        if batch_idx == 0:
            assert interaction['session_id'].tolist()[:5] == [1,2,3,4,5], 'train data shuffled when get session emb'
        inter = interaction.to(config["device"])
        targets = inter[model.POS_ITEM_ID]
        
        outputs = ensemble.calculate_expert_logits(model, inter)
        correctness += metrics(args, outputs, targets)
        
    return correctness

def overlap_ratio_matrix(lists):
    """jaccard coefficient 
    intersection / union of correctly predicted
    """
    n = len(lists)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):  # Only iterate over j >= i to avoid duplicate calculations
            if i == j:
                matrix[i][j] = 100  # Same list, full overlap
            else:
                count_overlap = 0
                total_correct = 0
                for k, y_pred in enumerate(lists[i]):
                    y_pred_other = lists[j][k]
                    if y_pred == 1:
                        total_correct += 1
                        if y_pred_other == 1:
                            count_overlap += 1
                
                overlap_ratio = count_overlap / total_correct
                matrix[i][j] = round(overlap_ratio*100, 2) # Set both directions
    return matrix

def get_one_acc(model, data, k=20):
    eval_fn = None
    if type(model).__name__ == 'ensembleMoE':
        eval_fn = model.calculate_all_expert_logits
    else:
        pass
    for batch_idx, interaction in enumerate(tqdm(data, total=len(data), ncols=100, desc=set_color(f"Evaluate [{type(model).__name__}]: ", "pink"))):
        if batch_idx == 0:
            assert interaction['session_id'].tolist()[:5] == [1,2,3,4,5], 'shuffled'
        inter = interaction.to(model.device)
        targets = inter[model.POS_ITEM_ID]
        
        preds = F.softmax(eval_fn(model, inter), dim=1)
        top_values, top_indices = torch.topk(preds, k=k, dim=1)
        is_in_top20 = torch.eq(top_indices, targets.unsqueeze(1)).any(dim=1)
        corr_idx = torch.where(is_in_top20==True)


def get_model_correctness(args, ensemble, config, data):
    
    model_list = ensemble.experts
    correctness = []
    for model in model_list:
        iter_data = tqdm(data, total=len(data), ncols=100, desc=set_color(f"Evaluate [{type(model).__name__}]: ", "pink"))
        c = get_metric(args, ensemble, model, iter_data, config)
        correctness.append(c)
        recall = round(sum(c)/len(c), 2)
        print(f'[{model}]: hit@{args.metrics_topk} = {recall}')
    return correctness
 
def vis_overlap(args, ensemble, config, data, is_save=False):
    """overlap prediction ratio, corr matrix"""
    correctness = get_model_correctness(args, ensemble, config, data)
    if is_save:
        dict = {}
        for i, m in enumerate(args.ensembles):
            dict[m] = correctness[i] # (np.where(np.array(correctness[i])==1)[0]+1).tolist() # sessionID starts from 1
        pth = 'saved/moe/models_pred.npy'
        with open(pth, 'wb') as handle:
            pickle.dump(dict, handle)
        print('\n' + f'[INFO]: save model prediction at: {pth}, #session={len(correctness[0])}' + '\n')
        return
    overlap_matrix = overlap_ratio_matrix(correctness)
    print(overlap_matrix)

def get_hard_samples(ensemble, config, data):
    """
    all pred correct - easy
    some predt correct - medium
    all pred wrong - hard
    """
    ensemble_correct = get_model_correctness(ensemble, config, data)

    n_ensemble = len(ensemble.experts)
    data_size = len(ensemble_correct[0])
    i = 0

    easy_idxs = []
    hard_idxs = []
    medium_idxs = []

    while i < data_size:
        n_correct = 0
        for model in ensemble_correct:
            n_correct += model[i]
        if n_correct == n_ensemble:
            easy_idxs.append(i)
        elif n_correct == 0:
            hard_idxs.append(i)
        else:
            medium_idxs.append(i)
        i += 1

    def _r(input):
        return round(len(input)/data_size, 2)
    print(f'easy, medium, hard idx ratio: {_r(easy_idxs)}, {_r(medium_idxs)}, {_r(hard_idxs)}')
    np.save(f'saved/moe/easy_idxs.npy', easy_idxs) 
    np.save(f'saved/moe/medium_idxs.npy', medium_idxs) 
    np.save(f'saved/moe/hard_idxs.npy', hard_idxs) 
    np.save(f'saved/moe/non_hard_idxs.npy', np.concatenate([easy_idxs, medium_idxs])) 
    
    return easy_idxs, medium_idxs, hard_idxs

def get_balance_samples(config, data):
    iter_data = tqdm(data, total=len(data), ncols=100, desc=set_color(f"Iterate all data: ", "pink"))
    for batch_idx, interaction in enumerate(iter_data):
        pass
    return 

    # all item appear at least once

def load_pretrain_emb(model, emb_pth):
    "load pretrained model emb from one to another"
    pretrain_emb = torch.load(emb_pth, map_location=model.device)["state_dict"]["item_embedding.weight"]
    model.item_embedding.weight.data.copy_(pretrain_emb) 
    model.item_embedding.weight.requires_grad = True
    logger = getLogger()
    logger.info(f'Load pretrained embedding from: {emb_pth}')

import matplotlib.pyplot as plt
def plot_acc_calibration(idx_test, output, labels, n_bins, title):
    output = torch.softmax(output, dim=1)
    pred_label = torch.max(output[idx_test], 1)[1]
    p_value = torch.max(output[idx_test], 1)[0]
    ground_truth = labels[idx_test]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='dodgerblue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='lightcoral', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + title +'.png', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()


"""
# a = np.array(lists[i])
            # b = np.array(lists[j])

            # correct_a_idx = np.where(a==1)[0]
            # a_val = a[correct_a_idx]
            # b_val = b[correct_a_idx]
            # a_val 
"""