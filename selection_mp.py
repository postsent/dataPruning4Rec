from distutils.log import error
import random
import pickle
import numpy as np
import matplotlib

from collections import defaultdict
from tqdm import tqdm

from samplings.core.data import CoresetSelection
from samplings.utils_sample import plot_score_distribution, get_partition, add_all_items_once, cluster_wise_sampling, min_list_cover, get_categories

import os
import torch

from recbole.utils import set_color, init_seed
import queue
import copy

from sklearn.metrics.pairwise import pairwise_distances

matplotlib.use('Agg')

def select_coreset(trainset, targets, args, data_score=None, is_return=False):
   
    # prepare variable
    item_sessionId = defaultdict(list)
    flattened_list = []
    
    for sessionId, s in enumerate(tqdm(trainset, total=len(trainset), ncols=100, desc=set_color(f"Store item_sessionId  ", "pink"))):
        flattened_list += s
        for item in set(s):
            item_sessionId[item].append(sessionId)
    uniq_train_items = list(set(flattened_list))
    
    # set seed
    init_seed(seed=args.seed, reproducibility=True)

    total_num = len(targets)
    # total samples to be selected for coreset
    coreset_num = int(args.coreset_ratio * total_num)

    # load data scores from training 100% data
    if args.coreset_mode != 'random' and data_score is None:
        with open(args.score_file, 'rb') as f:
            data_score = pickle.load(f)

    # set descending=True if higher data score implies higher importance
    if any(k in args.coreset_key for k in ['entropy', 'forgetting', 'el2n']):
        print(f"Using descending order for coreset key [{args.coreset_key}]")
        args.data_score_descending = 1

    score_index = None
    coreset_index = []
    if not os.path.isfile(args.item_file):
        coreset_index = min_list_cover(trainset)
        np.save(args.item_file, np.array(coreset_index))
    else:
        coreset_index = np.load(args.item_file).tolist()

    sets = [set(trainset[i]) for i in coreset_index]
    assert len(set(e for s in sets for e in s)) == len(uniq_train_items), f"not all unique items added: {len(set(e for s in sets for e in s))} vs {len(uniq_train_items)}"

    print(f'[INFO]: Added all items once, #sessions={len(coreset_index)}, %train={round(len(coreset_index)/total_num*100, 2)}')

    assert len(coreset_index) < coreset_num, 'add all item #sessions > expected total'
    # reduce ratio since prepend all items
    
    coreset_ratio = args.coreset_ratio - len(coreset_index) / total_num # w.r.t to all data seletion
    filtered_data_ratio = (args.coreset_ratio - len(coreset_index) / total_num) * (total_num / (total_num-len(coreset_index)))  # w.r.t prepend data selection
    print(f'[INFO]: Ratio reduce {args.coreset_ratio} -> {coreset_ratio}')
    
    categories = None
    remain_idxs = None
    if args.n_clusters and args.class_balanced: # category-wise sampling
        if args.partition == 'kmeans':
            categories, cluster_centers = get_categories(args.dataset, args.category_file, args.td_folder, args.n_clusters)
        elif args.partition == 'random':
            categories = get_partition(total_num, N=args.n_clusters, rand=True)
        elif args.partition == 'temporal':
            categories = get_partition(total_num, N=args.n_clusters, rand=False)
        unique, counts = np.unique(categories, return_counts=True)
        # print(f'[INFO]: {args.partition} category distribution: ', np.asarray((unique, counts)).T)
        categories = np.array(categories)
        
        categories[np.array(coreset_index)] = -1 # already selected, ignore 
    else:
        prepend = np.ones(total_num)
        prepend[np.array(coreset_index)] = -1
        remain_idxs = np.where(prepend==1)[0]
    ############################################################
    ######################## baselines #########################
    ############################################################

    if args.coreset_mode in ['random']:  # random sampling, baseline
        
        if args.class_balanced:
            print('[INFO]: Class balance mode.')
            coreset_index += cluster_wise_sampling(args.coreset_mode, categories, args.n_clusters, coreset_ratio)
        else:
            print('[INFO]: Not class balance mode.')
            coreset_index += CoresetSelection.random_selection(remain_idxs=remain_idxs,num=int(coreset_ratio * total_num))

    elif args.coreset_mode == 'coreset':  # for baseline methods other than CCS
        n_category = getattr(args, 'n_clusters', 0)
        print("Selecting from %s samples" % total_num)
        coreset_index += CoresetSelection.score_monotonic_selection(score=data_score[args.coreset_key], key=args.coreset_key,
                                                                   ratio=coreset_ratio,
                                                                   descending=(args.data_score_descending == 1),
                                                                   class_balanced=args.class_balanced, cat=(categories, n_category), remain_idxs=remain_idxs)
    elif args.coreset_mode == 'stratified':  

        if not args.class_balanced:
            coreset_index += CoresetSelection.stratified_sampling(data_score[args.coreset_key], int(coreset_ratio * total_num), stratas=args.stratas, remain_idxs=remain_idxs)
        else:
            # remove extreme value in scores e.g. inf, otherwise, graph propagate fails
            # mis_num = int(args.mis_ratio * total_num)
            # # sort index here
            # mis_key = args.coreset_key
            # mis_descending = True
            # if 'aum' in mis_key:
            #     mis_descending = False

            # data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=mis_key, # args.mis_key if args.mis_key else 'el2n'
            #                                                         mis_num=mis_num, mis_descending=mis_descending, 
            #                                                         coreset_key=args.coreset_key)

            # categories = categories[score_index] 
            # remind_idx_ = np.where(categories!=-1)[0]
            # score_re = data_score[args.coreset_key][remind_idx_]
            # sorted_idx = score_re.argsort(descending=mis_descending)

            # hard_index = sorted_idx[:mis_num]
            # print(f'mislabel_mask key: {mis_key}')
            # print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][remind_idx_][hard_index][:15]}')
            # print(f'Prune {hard_index.shape[0]} samples.')

            # categories[remind_idx_[hard_index]] = -1
            # easy_index = sorted_idx[mis_num:]
            # score_index = remind_idx_[easy_index]
            
            filtered_data_ratio = (args.coreset_ratio - len(coreset_index) / total_num) * (total_num / (total_num-len(coreset_index)-mis_num))  # w.r.t prepend data selection

            assert categories.shape[0] == data_score[args.coreset_key].shape[0], 'match'

            for label in tqdm(range(args.n_clusters), desc=f"Select {args.coreset_key}-{filtered_data_ratio} subsets "): 
                # get sample for the label
                sample_idxs_by_label = np.where(categories==label)[0]
                target_coreset_num = int(filtered_data_ratio * sample_idxs_by_label.shape[0])

                data_score_by_label = data_score[args.coreset_key][sample_idxs_by_label]
                coreset_index_by_label = CoresetSelection.stratified_sampling(data_score_by_label, target_coreset_num, args.stratas)
                
                selected_idx = sample_idxs_by_label[np.array(coreset_index_by_label)]
                coreset_index += selected_idx.tolist() # score_index[selected_idx]
                
                # assert all([targets[idx] == label for idx in score_index[sample_idxs_by_label[coreset_index_by_label]]])
                assert len(coreset_index_by_label) == int(target_coreset_num)
    elif args.coreset_mode == 'moderate':
        assert args.embed_file
        features = np.load(args.embed_file)
        map_idxs = np.where(categories!=-1)[0]

        selected_idxs = CoresetSelection.moderate_selection(ratio=filtered_data_ratio, features=features[map_idxs], targets_list=categories[map_idxs])
        coreset_index += map_idxs[selected_idxs].tolist()
    elif args.coreset_mode == 'prototype':
        """https://github.com/naotoo1/Beyond-Neural-Scaling"""
        assert args.embed_file
        features = np.load(args.embed_file)
        num_classes = args.n_clusters
        for i in tqdm(range(num_classes), desc='Select hard sample per cluster based cos-sim'):
            c = cluster_centers[i].reshape(1, -1)
            map_idxs = np.where(categories==i)[0]
            dist = pairwise_distances(c, features[categories==i], metric='cosine')[0]
            r = round(filtered_data_ratio * features[categories==i].shape[0])
            selected_idxs = np.argsort(-dist)[:r]# large to small
            coreset_index += map_idxs[selected_idxs].tolist() 
    elif args.coreset_mode == 'votek':
        pass
    else:
        raise ValueError

    ##########################################################################################

    coreset_index = np.array(coreset_index)

    assert len(np.unique(coreset_index)) - len(coreset_index) == 0, "Duplicates sessions are selected"
    assert np.issubdtype(coreset_index.dtype, np.integer), "index type needs to be int"
    assert coreset_index.max() <= total_num - 1, f'selected idx over range: {coreset_index.max()} vs {total_num - 1}'

    # num is less since overlap with prepended item session (item appears at least once)
    if len(coreset_index) < coreset_num:
        all_idxs = [i for i in range(0, len(targets))]
        k = int(coreset_num - len(coreset_index))
        
        extra = np.array([])
        extra_sample_set = list(set(all_idxs).difference(set(coreset_index)))
        m = 'Random'
        if args.coreset_mode in ['random', 'moderate', 'prototype','stratified']:  # add random samples if not reach ratio
            extra = np.array(random.sample(extra_sample_set, k))
            
        elif args.coreset_mode in ['coreset']:
            m = f'Highest scores [{args.coreset_key}]'
            extra_idx = torch.tensor(extra_sample_set)
            score = data_score[args.coreset_key]
            high_score_idxs = score[extra_idx].argsort(descending=(args.data_score_descending == 1))
            high_score_idxs = high_score_idxs[:k]
            extra = extra_idx[high_score_idxs]
            # high_score_idxs_non_overlapp = np.array([x for x in high_score_idxs if x not in coreset_index]) # order persistent
            # extra = high_score_idxs_non_overlapp[:k]
        else:
            raise NotImplementedError

        n_before = coreset_index.shape[0]
        coreset_index = np.hstack((coreset_index, extra))
        print(f"[INFO]: Added extra {k} samples by [{m}]: {round(n_before/total_num*100, 2)}%->{round(coreset_index.shape[0]/total_num*100, 2)}%")
        
    elif len(coreset_index) > coreset_num:    
        k = int(len(coreset_index) - coreset_num)
        thres = 100
        if len(coreset_index) - coreset_num <= thres:
            print(f'[INFO]: Extra {k} samples')
            coreset_index = coreset_index[:-k]
        else:
            raise ValueError(f' selected idx > expected: {len(coreset_index)} vs {coreset_num}')
        print("Removed extra %s samples" %  k)
    else:
        print('[INFO]: The coreset size is as expected, no addition or removal are applied')

    if is_return:
        return np.array(coreset_index)

    balance = args.partition if args.n_clusters else 'unbalanced'

    score_pth = f'{args.saved_path}/core_{args.coreset_mode}_{args.coreset_key}_{args.coreset_ratio}_{balance}.npy' 
    np.save(score_pth, np.array(coreset_index))

    print(f"Pruned {total_num} samples in original train set to {coreset_index.shape[0]}, {round(coreset_index.shape[0]/total_num*100, 2)}%")
    print(f'Saved coreset index at: {score_pth}')

