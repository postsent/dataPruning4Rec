import torch
import numpy as np
import torch.nn as nn
import pickle
from logging import getLogger
from tqdm import tqdm

from recbole.utils import set_color
import torch.nn.functional as F
from utils_seq import load_model

def save_topk_logits(folder, data, model, k):
    model.eval()
    gating_ratio = None
    res = []
    iter_data = tqdm(data, total=len(data), ncols=100, desc=set_color(f"[INFO] Get logits top[{k}]", "pink"))
    for batch_idx, interaction in enumerate(iter_data):
        if batch_idx == 0:
            assert interaction['session_id'].tolist()[:5] == [1,2,3,4,5], 'train data shuffled when get session emb'
        interaction = interaction.to(model.device)
        with torch.no_grad():
            preds, ratio = model.calculate_all_expert_logits(interaction, gating_ratio=True)
        if gating_ratio is None:
            gating_ratio = ratio
        else: 
            gating_ratio = torch.cat((gating_ratio, ratio), dim=0)
        top_values, top_indices = torch.topk(preds, k=k)
        for i in range(top_values.size(0)):
            res.append((top_values[i].tolist(), top_indices[i].tolist()))
    pth = f'{folder}/moe_top.npy'
    np.save(pth, np.array(res))
    print(f'Saved to: {pth}')
    mean = torch.round(torch.mean(gating_ratio, dim=0), decimals=6) 
    var = torch.round(torch.var(gating_ratio, dim=0), decimals=6) 
    print(set_color(f'Gating gating_ratio mean, var: {mean.cpu().numpy()}, {var.cpu().numpy()}', "yellow"))

def post_training_metrics_el2n(data_importance, model, train_data, key='post_el2n', topk=100):
    
    model.eval()
    data_importance[key] = []
    
    count = [1 if i in key else 0 for i in ['mean', 'moe', 'post', 'global'] ]
    assert sum(count) == 1, "multiple keywords in key"
    
    if 'mean' in key or 'moe' in key:
        print(set_color(f"\n[INFO]: Score model [Ensemble] \n", "pink"))
        gating_ratio = None
    elif 'global' in key:
        print(set_color(f"[global.w]: {nn.Softmax(0)(model.w)}", "yellow"))
    else:
        eval_model = model.experts[0]
        assert type(eval_model).__name__ == 'CORE', 'forward eval model'
        if 'rnn' in key:
            eval_model = model.experts[1]
            assert type(eval_model).__name__ == 'GRU4Rec', 'forward eval model'
        elif 'narm' in key:
            eval_model = model.experts[2]
            assert type(eval_model).__name__ == 'NARM', 'forward eval model'
        # elif 'gnn' in key:
        #     eval_model = model.experts[2]
        #     assert type(eval_model).__name__ == 'SRGNN', 'forward eval model'
        print(set_color(f"\n[INFO]: Score model [{type(eval_model).__name__}] \n", "pink"))
    
    iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"[INFO] Get difficulty score [{key}]   ", "pink"))
    for batch_idx, interaction in enumerate(iter_data):
        if batch_idx == 0:
            assert interaction['session_id'].tolist()[:5] == [1,2,3,4,5], 'train data shuffled when get session emb'
        interaction = interaction.to(model.device)
        with torch.no_grad():
            if 'mean' in key:
                preds = model.get_expert_logits_mean(interaction)
            elif 'moe' in key:
                preds, ratio = model.calculate_all_expert_logits(interaction, gating_ratio=True)
                if gating_ratio is None:
                    gating_ratio = ratio
                else: 
                    gating_ratio = torch.cat((gating_ratio, ratio), dim=0)
            elif 'post' in key:
                preds = nn.Softmax(dim=1)(model.calculate_expert_logits(eval_model, interaction))
            elif 'global' in key:
                preds = model.calculate_all_expert_logits(interaction)
            else:
                raise NotImplementedError
            
        targets = interaction[model.POS_ITEM_ID]
        score = None
        
        if 'entropy' in key:
            entropy = -1 * preds * torch.log(preds + 1e-10)
            score = torch.sum(entropy, dim=1)
        elif 'el2n' in key:
            l2_loss = torch.nn.MSELoss(reduction='none')
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1])
            score = torch.sqrt(l2_loss(preds, targets_onehot).sum(dim=1))
        elif 'aum' in key:
            batch_idx = torch.arange(preds.shape[0])
            target_prob = preds[batch_idx, targets]
            preds[batch_idx, targets] = 0
            other_highest_prob = torch.max(preds, dim=1)[0]
            score = target_prob - other_highest_prob
        else:
            raise NotImplementedError
        data_importance[key] += score.detach().cpu().tolist()
    if 'moe' in key:
        logger = getLogger()
        print('ratio.shape: ', gating_ratio.shape)
        mean = torch.round(torch.mean(gating_ratio, dim=0), decimals=6) 
        var = torch.round(torch.var(gating_ratio, dim=0), decimals=6) 
        logger.info(set_color(f'Gating gating_ratio mean, var: {mean.cpu().numpy()}, {var.cpu().numpy()}', "yellow"))
        
    data_importance[key] = torch.tensor(data_importance[key])
 
def post_training_metrics(data_importance, model, train_data, device):
    """
    Calculate loss and entropy based on pretrained model
    """
    model.eval()
    data_importance['entropy'] = []
    data_importance['loss'] = []
    data_importance['confidence'] = []

    iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Evaluate   ", "pink"))

    for batch_idx, interaction in enumerate(iter_data):
        if batch_idx == 0:
            assert interaction['session_id'].tolist()[:5] == [1,2,3,4,5], 'train data shuffled when get session emb'
        
        interaction = interaction.to(device)
        # CORE model
        item_seq = interaction[model.ITEM_SEQ]
        seq_output = model.forward(item_seq)
        all_item_emb = model.item_embedding.weight
        all_item_emb = F.normalize(all_item_emb, dim=-1) # Robust Distance Measuring (RDM)
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / model.temperature
        
        targets = interaction[model.POS_ITEM_ID]
        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)

        prob = nn.Softmax(dim=1)(logits)
    
        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1)

        confidence = prob[torch.arange(0, logits.shape[0]).to(device), targets] # confidence on target item

        data_importance['entropy'] += entropy.detach().cpu().tolist()
        data_importance['loss'] += loss.detach().cpu().tolist()
        data_importance['confidence'] += confidence.detach().cpu().tolist()

    data_importance['entropy'] = torch.tensor(data_importance['entropy'])
    data_importance['loss'] = torch.tensor(data_importance['loss'])
    data_importance['confidence'] = torch.tensor(data_importance['confidence'])

    expected_len = train_data.dataset.inter_feat['item_id'].shape[0]
    assert expected_len == data_importance['entropy'].shape[0] == data_importance['loss'].shape[0] == data_importance['confidence'].shape[0], 'shape'

def training_dynamics_metrics(td_log, data_importance, max_epoch=5):
    """Calculate td metrics
    'forgetting': None, 
    'el2n': None, 
    'variance': None, 
    'entropy': None, 
    """

    def map_idx(td_, key_, idxs_):
        """map source array given target array index"""
        return np.array(td_[key_])[idxs_]

    # map based on shuffle idx
    i = 0
    while i < len(td_log):
        idxs = np.array(td_log[i]['idxs']) - 1 # TODO: substracted by 1 since the stored idx=session_id, starts from 1
        # update all score list based on shuffle idx
        for key, v in td_log[i].items():
            if key == 'idxs' or td_log[i][key] == []: continue
            td_log[i][key] = map_idx(td_log[i], key, idxs)
        i += 1

    def get_moving_avg(td_log_, key):
        dict_key = key
        if key == 'variance':
            dict_key = 'confidence'
        elif key == 'forgetting':
            dict_key = 'correctness'
        res = np.array([s[dict_key] for s in td_log_])
        
        if key == 'variance':
            res = np.exp(res)
            res = np.std(res, axis=0)
        elif key in ['forgetting']:
            last_correct = np.zeros(len(td_log_[0]['idxs']))
            tmp = np.zeros(len(td_log_[0]['idxs']))
            for cur in res:
                tmp += np.logical_and(last_correct == 1, cur == 0)
                last_correct = cur
            res = tmp
        elif key in ['el2n', 'aum']:
            res = np.sum(res, axis=0)
        else:
            raise ValueError
        return torch.from_numpy(res)        

    data_importance['variance'] = get_moving_avg(td_log, 'variance')
    data_importance['el2n'] = get_moving_avg(td_log, 'el2n')
    data_importance['forgetting'] = get_moving_avg(td_log, 'forgetting')
    data_importance['aum'] = get_moving_avg(td_log, 'aum')

def feature_maps(model, dataloader, device, layer=4):
    """Extract feature map"""
    # model.eval()
    # features = []
    # for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
    #     inputs, targets = inputs.to(device), targets.to(device)

    #     feats = model.feature_map(inputs, layer=layer)
    #     features.append(feats.detach().cpu().numpy())

    # features = np.concatenate(features)
    # return features
    pass # TODO

def append_scores(args, model, train_data, key, moe_pth=''):
    score_file = f'saved/td-{args.dataset}/scores-{args.dataset}-CORE.pickle'
    with open(score_file, 'rb') as f:
        data_importance = pickle.load(f)
    if 'moe' in key or 'global' in key:
        # load pretrained
        model = load_model(moe_pth, model, args.device)

    print(f'[INFO]: Append score: [{key}]')
    post_training_metrics_el2n(data_importance, model, train_data, key=key)
    logger = getLogger()
    logger.info(f'Saving data score at {score_file}')
    with open(score_file, 'wb') as handle:
        pickle.dump(data_importance, handle)
        
def get_scores(args, model, train_data):
    data_importance = {}
    with open(args.td_file, 'rb') as f:
        pickled_data = pickle.load(f)
        training_dynamics = pickled_data['training_dynamics']

    # metrics
    data_importance['targets'] = train_data.dataset.inter_feat['item_id']
    post_training_metrics(data_importance, model, train_data, model.device)

    assert [i for i in range(train_data.dataset.inter_feat['item_id'].shape[0])] == [i - 1 for i in sorted(training_dynamics[0]['idxs'])], 'save idxs not match'
    training_dynamics_metrics(training_dynamics, data_importance)

    logger = getLogger()
    score_file = f'saved/td-{args.dataset}/scores-{args.dataset}-{args.model}.pickle'
    logger.info(f'Saving data score at {score_file}')
    with open(score_file, 'wb') as handle:
        pickle.dump(data_importance, handle)

if __name__ == "__main__":


    if args.feature:
        # trainset = SeqDataset(train_data)
        # print(len(trainset))
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

        # print(f'Saving train features at {train_feature_path}')
        # features = feature_maps(model, trainloader, device)
        # print(features.shape)
        # np.save(train_feature_path, features)
        # print(f'Saving test features at {test_feature_path}')
        # testset = IndexDataset(testset)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        # features = feature_maps(model, testloader, device)
        # np.save(test_feature_path, features)
        pass # TODO


# if 'entropy_20' in key:
#             preds, _ = torch.topk(preds, k=topk)
#             entropy = -1 * preds * torch.log(preds + 1e-10)
#             score = torch.sum(entropy, dim=1)