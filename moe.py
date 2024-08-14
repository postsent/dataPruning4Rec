"""
https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
also: https://gist.github.com/ruvnet/0928768dd1e4af8816e31dde0a0205d5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from moe.utils_moe import get_seq_feat, get_ori_seq, get_item_freq, intra_distance_loss

from recbole.model.layers import TransformerEncoder
from torch.distributions.normal import Normal

class MLP(nn.Module):
    """1-layer Feed-Forward networks
    reference: https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    batchnorm1d not helpful
    """

    def __init__(self, input_size, output_size, hidden_size, n_layer=1, dropout=0.0):
        super(MLP, self).__init__()
   
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size)) # TODO hidden size
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layer):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):  
        for layer in self.layers:
            x = layer(x)
        return x

class EnsembleMoE(SequentialRecommender):
    """TODO"""
    def __init__(self, config, dataset, expert_list, n_experts, ename, bal_coef=1, noisy_gating=False, k=2):
        
        super(EnsembleMoE, self).__init__(config, dataset)
        
        self.softmax = nn.Softmax(1)
        self.device = config['device']
        
        self.loss_fct = nn.CrossEntropyLoss() 

        self.experts = expert_list 
        self.experts_name = ename
        
        self.bal_coef = bal_coef

        for e in expert_list: # fine tune embed model
            e.eval()
            for param in e.parameters():
                param.requires_grad = False
        assert type(self.experts[0]).__name__ == 'CORE', 'First model default to CORE model to produce seq representation'
        self.n_classes, self.emb_size = self.experts[0].item_embedding.weight.shape

        # data features
        seq_data = get_ori_seq(config)
        self.item_freq = get_item_freq(seq_data)
        self.feat_size = 3 + 4 * len(expert_list) # 3 basic, 4 dist related
        """ emb + feat """
        self.process_emb = MLP(input_size=self.emb_size, output_size=self.emb_size, hidden_size=64, n_layer=3) 
        self.process_feat = MLP(input_size=self.feat_size, output_size=self.feat_size, hidden_size=64, n_layer=3) 

        self.noisy_gating = noisy_gating
        if not noisy_gating:
            self.router = MLP(input_size=self.emb_size+self.feat_size, output_size=n_experts, hidden_size=64, n_layer=3) 
        else:
            # weighted avg by k experts
            self.num_experts = n_experts
            self.k = k
            self.w_gate = MLP(input_size=self.emb_size+self.feat_size, output_size=n_experts, hidden_size=64, n_layer=3) 
            self.w_noise = MLP(input_size=self.emb_size+self.feat_size, output_size=n_experts, hidden_size=64, n_layer=3) 
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))

            self.softplus = nn.Softplus()
            assert(self.k <= self.num_experts)


    def calculate_expert_logits(self, expert, interaction):
        """Expert forward in eval mode """
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        expert.eval()
        name = type(expert).__name__ 
        if name == 'CORE':
            seq_output = expert.forward(item_seq)
            all_item_emb = expert.item_embedding.weight
            all_item_emb = F.normalize(all_item_emb, dim=-1) # Robust Distance Measuring (RDM)
            logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / expert.temperature
            self.seq_output = seq_output
        elif name in ['SRGNN', 'SASRec', 'GRU4Rec', 'LightSANs', 'NARM']: # 'GCSAN'
            expert.eval()
            seq_output = expert.forward(item_seq, item_seq_len)
            test_item_emb = expert.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        elif name == 'BERT4Rec':
            expert.eval()
            masked_item_seq = interaction[self.MASK_ITEM_SEQ]
            masked_index = interaction[self.MASK_INDEX]

            seq_output = expert.forward(masked_item_seq)
            test_item_emb = expert.item_embedding.weight[: expert.n_items] 
            
            pred_index_map = expert.multi_hot_embed(masked_index, masked_item_seq.size(-1))           # [B*mask_len max_len]
            pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)    # [B mask_len] -> [B mask_len max_len] multi hot ; # [B mask_len max_len]
            # [B mask_len max_len] * [B max_len H] -> [B mask_len H] ; only calculate loss for masked position
            seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]
            test_item_emb = expert.item_embedding.weight[: expert.n_items]  # [item_num H]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        else:
            raise ValueError(f'No such model: {name} in ensemble pool')
        return logits

    def get_heuristics(self, interaction, expert_logits):
        
        feat = get_seq_feat(interaction, self).to(self.device)
        for logits in expert_logits:
            feat = torch.hstack((feat, self._get_heuristics(logits, interaction[self.POS_ITEM_ID])))
        return feat

    def _get_heuristics(self, logits, targets):
        """
        entropy, confidence, el2n, aum
        """
        prob = nn.Softmax(dim=1)(logits) # .clone().detach()
        # entropy
        entropy = -1 * prob * torch.log(prob + 1e-10)
        # confidence
        batch_idx = torch.arange(prob.shape[0])
        target_prob = prob[batch_idx, targets]
        # el2n
        l2_loss = torch.nn.MSELoss(reduction='none')
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=prob.shape[1])
        el2n = torch.sqrt(l2_loss(prob, targets_onehot).sum(dim=1))
        # aum
        prob[batch_idx, targets] = 0
        other_highest_prob = torch.max(prob, dim=1)[0]
        aum = (target_prob - other_highest_prob)

        res = entropy.sum(1).view(-1, 1)
        res = torch.hstack((res, target_prob.view(-1, 1), aum.view(-1, 1), el2n.view(-1, 1)))

        return res

    def calculate_all_expert_logits(self, interaction, gating_ratio=False, is_bal=False):
        """weighted sum of expert model output
        bal_loss: The backpropagation of this loss encourages all experts to be approximately equally used across a batch.
        """

        self.seq_output = None # updated in calculate_expert_logits[0]
        all_expert_logits = []
        for i, expert in enumerate(self.experts):
            expert_logits = self.calculate_expert_logits(expert, interaction) # [NOTE] no need for activation
            all_expert_logits.append(expert_logits)

        feat = self.get_heuristics(interaction, all_expert_logits)           
        router_input = torch.hstack((self.process_emb(self.seq_output), self.process_feat(feat)))
        if self.noisy_gating:
            gates, load = self.noisy_top_k_gating(router_input, self.training)
        else:
            gates = self.softmax(self.router(router_input))
            load = self._gates_to_load(gates)

        logits = None
        for i, expert in enumerate(self.experts):
            expert_logit = self.softmax(all_expert_logits[i])
            weighted_output = expert_logit * gates[:, i].unsqueeze(1) # Extract & apply gating scores, unsqueeze [50 -> [50, 1] to match model output shape to squeeze back for output
            
            if logits is None:
                logits = weighted_output.squeeze(1) 
            else:
                logits += weighted_output.squeeze(1)

        # for get importance score
        if gating_ratio:
            return logits, gates
        
        # calculate importance loss
        importance = gates.sum(0)
        bal_loss = self.cv_squared(importance) + self.cv_squared(load)
        
        if is_bal:
            return logits, bal_loss     
        return logits
        
    def calculate_loss(self, interaction):
        pos_items = interaction[self.POS_ITEM_ID]
        logits, bal_loss  = self.calculate_all_expert_logits(interaction, is_bal=True) 
        loss = self.loss_fct(logits, pos_items)
        return loss + self.bal_coef * bal_loss

    def full_sort_predict(self, interaction):
        return self.calculate_all_expert_logits(interaction)

    def predict(self, interaction):
        pass

    def get_expert_logits_mean(self, interaction):

        res = None
        for i, expert in enumerate(self.experts):
            expert_logits = self.calculate_expert_logits(expert, interaction) 
            if res is None:
                res = self.softmax(expert_logits)
            else:
                res += self.softmax(expert_logits)
        res = res / len(self.experts)
        return res
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob