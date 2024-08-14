import os
import pickle
import numpy as np
import copy
import torch
import numpy as np
import torch.nn.functional as F

from recbole.trainer import Trainer
from tqdm import tqdm
from recbole.utils import set_color, get_gpu_usage
import torch.cuda.amp as amp
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

# for eval sampling
from get_importance_score import post_training_metrics_el2n
from samplings.selection_mp import select_coreset
from samplings.utils_sample import convert_seq_format
from utils_seq import data_preparation
from recbole.utils import get_trainer
from logging import getLogger

from recbole.utils import get_model
from recbole.config import Config
from recbole.data.dataset import SequentialDataset
from recbole.data import create_dataset
import torch.optim as optim

class TrainingDynamicsLogger(object):
    """
    Helper class for saving training dynamics for each iteration.
    Maintain a list containing output probability for each sample.
    """
    def __init__(self, epoch, dataset, td_file):
        
        self.log = []
        log_template = {
            'idxs' : [],
            'correctness': [], 
            'el2n': [], 
            'entropy': [], 
            'confidence': [],
            'aum' : []
        }
        self.filepath = td_file
        self.data_name = dataset
        for _ in range(epoch):
            self.log.append(copy.deepcopy(log_template))
        
    def log_dict(self, epoch, dict):
        
        self.log[epoch]['correctness'] += dict['correctness']
        self.log[epoch]['el2n'] += dict['el2n']
        self.log[epoch]['confidence'] += dict['confidence']
        self.log[epoch]['entropy'] += dict['entropy']
        self.log[epoch]['aum'] += dict['aum']

    def log_idx(self, epoch, idxs):
        self.log[epoch]['idxs'] = idxs

    def save_training_dynamics(self):
        pickled_data = {
            'data_name': self.data_name,
            'training_dynamics': self.log
        }

        with open(self.filepath, 'wb') as handle:
            pickle.dump(pickled_data, handle)
        
    def reinit(self):
        self.training_dynamics = []

def save_td(out, targets, k=20):
    """record, 
    indexing https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
    https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
    https://github.com/haizhongzheng/Coverage-centric-coreset-selection/blob/b37f166ace760e17f4d5baae17a828bdb88c6667/generate_importance_score.py#L89
    """
    outputs = out.clone().detach()
    output_softmax = F.softmax(outputs, dim=1)

    correctness = 0
    confidence = 0 # target prob
    el2n = 0
    
    # el2n
    l2_loss = torch.nn.MSELoss(reduction='none')
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1])
    el2n = torch.sqrt(l2_loss(output_softmax, targets_onehot).sum(dim=1)).tolist()

    # for confidence, aply softmax, for variance, apply exp later
    confidence = output_softmax.gather(1, (targets).view(-1,1)).reshape(-1).tolist()

    # correct
    correctness = []
    top_values, top_indices = torch.topk(output_softmax, k=k)
    for top_idx, tar in zip(top_indices.detach().cpu().numpy(), targets.detach().cpu().numpy()):
        correctness.append(1 if np.isin(tar, top_idx) else 0)

    # entropy
    entropy = -1 * output_softmax * torch.log(output_softmax + 1e-10)
    entropy = torch.sum(entropy, dim=1).tolist()

    
    # accumulated_margin
    batch_idx = torch.arange(output_softmax.shape[0])
    target_prob = output_softmax[batch_idx, targets]
    output_softmax[batch_idx, targets] = 0
    other_highest_prob = torch.max(output_softmax, dim=1)[0]
    aum = (target_prob - other_highest_prob).tolist()
    
    return {
        'correctness': correctness, 
        'el2n': el2n, 
        'confidence': confidence, 
        'entropy': entropy, 
        'aum': aum
    }

from recbole.model.sequential_recommender.core import CORE
class CORETD(CORE):
    def __init__(self, config, dataset):
        super(CORETD, self).__init__(config, dataset)
        self.logits = None

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = (
            torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        )
        self.logits = logits
        loss = self.loss_fct(logits, pos_items)
        return loss


class CustomTrainer(Trainer):
    """
    new trainer so that can keep track of the training dynamic
    """
    def __init__(self, config, model, args, test_data, m='GRU4Rec'):
        
        super(CustomTrainer, self).__init__(config, model)
        self.log_train = getLogger()
        if args.model == 'ensembleMoE' and args.is_eval_sample: # TODO separate eval moedl and ensemble; no duplicate attribute name as this
            
            self.saved_moe_pth = f'saved/saved_moe/{args.dataset}/{args.exp_name}'
            self.coreset_args = self._get_coreset_args(args)
            self.eval_model = m
            self.raw_dataset = SequentialDataset(config).build()[0] # first
    
            c = Config(model=m, dataset=args.dataset, config_file_list=[args.config_overall_pth])
            c["shuffle"] = False # set train no shuffle since get score
            c["train_batch_size"] = c["train_batch_size"]*2
            dataset = create_dataset(c)
            train_data_eval_score, _, _ = data_preparation(c, dataset)
            self.train_data_eval_score = train_data_eval_score
            self.test_data_c = test_data
        self.args = args
        

    def _get_coreset_args(self, args):
        class coreset_config:
            def __init__(self, key, ratio):
                self.n_clusters = 0 # not cluster wise
                self.coreset_key = key  # margin for digi & amazon_m2, entropy for yoochoose post_
                self.coreset_ratio = ratio
                self.coreset_mode = 'coreset'
                self.class_balanced = False
                self.data_score_descending = 0 # TODO aum
                self.partition = ''
                self.mis_key = ''
                self.seed = 0
                self.item_file = args.item_file
                self.category_file = ""
                self.mis_ratio = 0.0
        key = 'moe_aum'
        ratio = 0.6
        if args.dataset == 'yoochoose':
            key = 'moe_entropy'
            ratio = 0.3
        return coreset_config(key, ratio)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        if self.args.td:
            shuffle_session_idx = []
        
        for batch_idx, interaction in enumerate(iter_data):                

            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)
            
            ################################# EDIT #################################
            if self.args.td:
                shuffle_session_idx += interaction['session_id'].tolist()
                self.args.TD_logger.log_dict(epoch_idx, save_td(self.model.logits, interaction[self.model.POS_ITEM_ID]))
            #################################

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            scaler.scale(loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
                
        # record shuffle index and training dynamic
        if self.args.td:
            self.args.TD_logger.log_idx(epoch_idx, shuffle_session_idx)
            self.args.TD_logger.save_training_dynamics()

        # eval sampled data
        if self.args.model == 'ensembleMoE' and self.args.is_eval_sample: 
            self.save_one_model(epoch_idx)

            self.log_train.info(set_color(f"--------------------------------------------------", "pink"))
            self.log_train.info(set_color(f"Evaluate retrain model on selected sample [{self.coreset_args.coreset_key}]", "pink"))
            self.eval_sample()
            self.log_train.info(set_color(f"--------------------------------------------------", "pink"))
            
        return total_loss

    def save_one_model(self, epoch):
        state = {
            "config": self.config,
            "epoch": epoch,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
        }
        pth = f'{self.saved_moe_pth}/moe-{epoch}.pth'
        self.log_train.info(set_color(f"[INFO]: Saved ensemble at: {pth} \n", "pink"))
        os.makedirs(self.saved_moe_pth, exist_ok=True)    
        torch.save(state, pth)
        test_result = self.evaluate(self.test_data_c, load_best_model=True, model_file=pth, show_progress=True)
        self.log_train.info(set_color(f'test result ep[{epoch}]', 'yellow') + f': {test_result}')

    def eval_sample(self):
        # TODO: convert seq format        
        data_score = {}
        post_training_metrics_el2n(data_score, self.model, self.train_data_eval_score, key=self.coreset_args.coreset_key)

        subset_file = select_coreset(trainset=convert_seq_format(self.raw_dataset), targets=self.raw_dataset['item_id'].tolist(), args=self.coreset_args, data_score=data_score, is_return=True)
        
        self._eval_model_on_samples(subset_file)

    def _eval_model_on_samples(self, subset_file, is_many=True, k=20):    
        c_pth = 'config/overall.yaml'
        if self.args.dataset == 'amazon_m2':
            c_pth = 'config/amazon_m2.yaml'  
            k = 100
        models = [self.eval_model]
        if is_many:
            models = ['GRU4Rec', 'CORE']
        self.log_train.info(set_color(f"Load eval model for sampled data [{models}]", "pink"))
        avg_recall = []
        avg_mrr = []
        for m in models:
            c = Config(model=m, dataset=self.args.dataset, config_file_list=[c_pth])
            assert c['epochs'] >= 80, 'retrain GRU needs high epochs'
            dataset = create_dataset(c)
            
            train_data, valid_data, test_data = data_preparation(c, dataset, subset_file)
            model = get_model(c['model'])(c, train_data.dataset).to(c['device'])
            eval_trainer = get_trainer(c['MODEL_TYPE'], c['model'])(c, model)
            eval_trainer.fit(train_data, valid_data, saved=True, show_progress=c['show_progress'])
            
            # avg_recall.append(eval_trainer.best_valid_result[f'recall@{k}'])
            # avg_mrr.append(eval_trainer.best_valid_result[f'mrr@{k}'])
            test_result = eval_trainer.evaluate(test_data, load_best_model=True, show_progress=c['show_progress'])
            avg_recall.append(test_result[f'recall@{k}'])
            avg_mrr.append(test_result[f'mrr@{k}'])
            self.log_train.info(set_color(f"Test results: {test_result}", "pink"))
        if is_many:
            avg_recall = round(sum(avg_recall) / len(avg_recall), 3)
            avg_mrr = round(sum(avg_mrr) / len(avg_mrr), 3)
            self.log_train.info(set_color(f"Avg best test results: Recall@{k}:{avg_recall}, MRR@{k}:{avg_mrr}", "pink"))
            # self.log_train.info(set_color(f"Avg best valid results: Recall@{k}:{avg_recall}, MRR@{k}:{avg_mrr}", "pink"))
