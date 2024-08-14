import argparse
import os
from logging import getLogger
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_local_time

from moe import EnsembleMoE, GRU4RecKD, GlobalEnsemble
from utils_seq import load_ensemble_pretrained_models, vis_overlap, get_hard_samples, load_model, load_pretrain_emb
from samplings.utils_sample import FowardTrainner, init_coreset_logger, save_test_results, tune_sampling
from utils_trainer import CustomTrainer, TrainingDynamicsLogger, CORETD
from get_importance_score import get_scores, append_scores, save_topk_logits

def run_single_model(args):

    if args.model in ['SRGNN','GCSAN'] and not args.recbole:
        from recbole_gnn.config import Config
        from recbole_gnn.utils import create_dataset, data_preparation, get_model
    else:
        from recbole.config import Config
        from recbole.data import create_dataset
        from utils_seq import data_preparation
        from recbole.utils import get_model

    # configurations initialization
    config_overall_pth = 'config/overall.yaml'
    
    if args.model == 'ensembleMoE': # TODO: moe in amazon m2 case
        config_overall_pth = 'config/overall_moe.yaml'   
    if args.dataset == 'amazon_m2':
        config_overall_pth = 'config/amazon_m2.yaml'     
    
    args.config_overall_pth = config_overall_pth

    if args.model == 'ensembleMoE':
        config = Config(model=EnsembleMoE, dataset=args.dataset, config_file_list=[config_overall_pth, f'config/EnsembleMoE.yaml'])
    elif args.model == 'global':
        config = Config(model=GlobalEnsemble, dataset=args.dataset, config_file_list=[config_overall_pth, f'config/EnsembleMoE.yaml'])
    elif args.model == 'kd':
        config = Config(model='GRU4Rec', dataset=args.dataset, config_file_list=[config_overall_pth])
    else:
        config = Config(model=args.model, dataset=args.dataset, config_file_list=[config_overall_pth])

    # different config cases
    if args.model in ['SRGNN','GCSAN'] and args.dataset == 'amazon_m2':
        config['train_batch_size'] = 100
    elif args.model == 'ensembleMoE':
        config['epochs'] = 3
    elif args.model == 'global':
        config['epochs'] = 1
    elif args.model == 'SASRec' and args.dataset == 'amazon_m2': # https://github.com/RUCAIBox/RecBole/issues/1357
        config['MAX_ITEM_LIST_LENGTH'] = 201 
    
    # update config
    if args.embed_size:
        config['embedding_size'] = args.embed_size
    if args.epoch:
        config['epochs'] = args.epoch


    args.device = config['device']
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    
    if args.coreset_mode:
        init_coreset_logger(config, args.coreset_mode, args.coreset_ratio)
    else:
        init_logger(config)

    logger = getLogger()
    logger.info(args)
    logger.info(config)
    
    if args.td:
        logger.info(f'=== Save training dynamic to: {args.td_file} ===')
        TD_logger = TrainingDynamicsLogger(config['epochs'], args.dataset, args.td_file)
        args.TD_logger = TD_logger
        
    # sample dataset at the build() level, recbole gnn diff recbole
    dataset = create_dataset(config) if not args.model in ['SRGNN','GCSAN'] or args.recbole else create_dataset(config, args.subset_file)
    logger.info(dataset)

    if args.get_samples or args.is_vis or args.save_pred or args.get_session_emb or args.get_score or args.append_score or args.save_topk_logits:
        config["shuffle"] = False # set train no shuffle
        
        train_data, valid_data, test_data = data_preparation(config, dataset, args.subset_file)
        if args.get_session_emb:
            model = get_model(config["model"])(config, train_data.dataset).to(config['device'])
            trainer = FowardTrainner(config, model) 
            trainer.save_session_emb(train_data, args)
            return
        if args.get_score:
            model = get_model(config["model"])(config, train_data.dataset).to(config['device'])
            model = load_model(f'{args.td_folder}/{args.model}.pth', model, config["device"])
            get_scores(args, model, train_data)
            return 

        expert_list = load_ensemble_pretrained_models(args)
        model = EnsembleMoE(config, train_data.dataset, expert_list, args.n_ensembles, args.ensembles, noisy_gating=args.noisy_gating).to(config['device'])

        if args.append_score:
            moe_pth = ''
            if 'moe' in args.append_score:
                moe_pth = f'{args.td_folder}/moe-{args.dataset}.pth'
            elif 'global' in args.append_score:
                moe_pth = f'{args.td_folder}/global.pth'
                model = GlobalEnsemble(config, train_data.dataset, expert_list, args.n_ensembles, args.ensembles).to(config['device'])
            append_scores(args, model, train_data, args.append_score, moe_pth=moe_pth)
            return
        if args.save_topk_logits:
            moe_pth = f'{args.td_folder}/moe-{args.dataset}.pth'
            model = load_model(moe_pth, model, args.device)
            save_topk_logits(args.td_folder, train_data, model, k=args.save_topk_logits)
            return
        assert args.model == 'ensembleMoE'
        if args.get_samples:
            """easy, medium, hard idx ratio: 0.44, 0.22, 0.34"""
            get_hard_samples(model, config, train_data)
            return 
    
        if args.is_vis or args.save_pred:
            """ ['CORE', 'SASRec', 'SRGNN', 'GRU4Rec', 'LightSANs']
            [[100, 90.75, 88.16, 92.19, 89.62],
            [91.71, 100, 88.57, 92.29, 90.42],
            [91.54, 91.01, 100, 92.42, 89.69],
            [88.09, 87.26, 85.04, 100, 85.78],
            [92.62, 92.47, 89.27, 92.78, 100]]
            """
            vis_overlap(args, model, config, train_data, is_save=args.save_pred)
            return 

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset, args.subset_file)

    if args.is_eval_base_models:
        
        for model_name in args.ensembles:
            config = Config(model=model_name, dataset=args.dataset, config_file_list=[args.config_overall_pth])
            model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
            saved_pth = f'{args.td_folder}/{model_name}.pth'
            test_result = trainer.evaluate(test_data, load_best_model=True, model_file=saved_pth, show_progress=config['show_progress'])
            logger.info(set_color('test result', 'yellow') + f': {test_result}')
        return 

    # model loading and initialization
    if args.model == 'ensembleMoE': 
        expert_list = load_ensemble_pretrained_models(args)
        model = EnsembleMoE(config, train_data.dataset, expert_list, args.n_ensembles, args.ensembles, bal_coef=args.coef, noisy_gating=args.noisy_gating).to(config['device'])
    elif args.model == 'global': 
        expert_list = load_ensemble_pretrained_models(args)
        model = GlobalEnsemble(config, train_data.dataset, expert_list, args.n_ensembles, args.ensembles).to(config['device'])
    elif args.model == 'kd':
        model = GRU4RecKD(config, train_data.dataset, args.topk_logit_pth).to(config["device"])
    elif args.model == 'CORE' and args.td:
        model = CORETD(config, train_data.dataset).to(config['device'])
    else:
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    if args.pretrain_emb:
        load_pretrain_emb(model, emb_pth=f'{args.td_folder}/{args.pretrain_emb}.pth')

    logger.info(model)

    # trainer loading and initialization
    if args.td or args.model == 'ensembleMoE':
        trainer = CustomTrainer(config, model, args, test_data)
    else:
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # update trainer
    if args.exp_name:
        saved_model_file = '{}-{}-{}-{}.pth'.format(trainer.config['model'], args.dataset, args.exp_name, get_local_time())
        trainer.saved_model_file = os.path.join(trainer.checkpoint_dir, saved_model_file)

    if args.eval_moe:
        test_result = trainer.evaluate(test_data, load_best_model=True, model_file=args.moe_pth, show_progress=config['show_progress'])
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
        return 
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress'] # save must be true to save best model 
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    if args.is_run_scripts:
        folder = f'./saved_samples/{args.dataset}'
        save_test_results(test_result, folder, args.dataset, args.coreset_mode, args.coreset_ratio, args.model, args.exp_name)
    if args.tune_sampling:
        folder = f'./tune_sampling/{args.dataset}'
        tune_sampling(test_result, best_valid_result, folder, args.dataset, args.coreset_mode, args.exp_name)
        
    if args.hyperparam_file != '':
        with open(args.hyperparam_file, "a") as f:
            f.write(f"{test_result}")
            f.write('\n')
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CORE', help='model') # COREmoe, SRGNN, ensembleMoE
    parser.add_argument('--dataset', '-d', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    
    parser.add_argument("--subset_file", type=str,help="session id subset list") # random is 10% for sanity check
    
    parser.add_argument('--is_vis', action="store_true", help="visualize")
    parser.add_argument('--save_pred', action="store_true", help="save all base model predictions")
    parser.add_argument('--metrics_topk', type=int, default=10, help="hit@k")
    parser.add_argument('--get_samples', action="store_true", help="get overlap samples")

    parser.add_argument('--get_session_emb', action="store_true", help="get_session_emb")

    parser.add_argument('--recbole', action="store_true", help="recbole api")
    
    # retrain with sampling
    parser.add_argument('--is_run_scripts', action="store_true", help="auto")
    parser.add_argument('--coreset_mode', default='', type=str) # choices=['random', 'el2n', 'forgetting', 'variance', 'entropy', 'post_el2n', 'aum', 'post_aum']
    parser.add_argument('--coreset_ratio', type=float)

    # td
    parser.add_argument('--td', action="store_true", help="training dynamics")
    parser.add_argument('--get_score', action="store_true", help="get_score")
    parser.add_argument('--append_score', type=str, default='', help='append no moving avg score')

    parser.add_argument('--hyperparam_file', type=str, default='', help='')
    parser.add_argument('--is_eval_base_models', action="store_true", help="")
    parser.add_argument('--is_eval_sample', action="store_true", help="")
    
    parser.add_argument('--exp_name', type=str, default='', help='')
    parser.add_argument('--coef', type=float, help="hit@k")

    parser.add_argument('--tune_sampling', action="store_true", help="")
    parser.add_argument('--save_topk_logits', type=int, help="")
    parser.add_argument('--embed_size', type=int, help="")
    parser.add_argument('--pretrain_emb', type=str, default='', help="")
    parser.add_argument('--epoch', type=int, help="")
    # parser.add_argument('--noisy_gating', action="store_true", help="")
    parser.add_argument('--eval_moe', action="store_true", help="")

    args, _ = parser.parse_known_args()

    # args.ensembles = ['CORE'] # sanity check
    # args.ensembles = ['CORE', 'SASRec', 'SRGNN', 'GRU4Rec', 'LightSANs'] # first model produce the embedding
    args.ensembles = ['CORE', 'GRU4Rec', 'NARM']
    
    args.n_ensembles = len(args.ensembles)
    args.td_folder = f'saved/td-{args.dataset}'
    if args.get_session_emb:
        assert args.model == 'CORE'
        args.embed_file = f'{args.td_folder}/sessionEmbedding-{args.model}-{args.dataset}.npy'
        args.model_file = f'{args.td_folder}/{args.model}.pth'
    if args.subset_file and args.subset_file != 'random':
        args.subset_file = f'./saved_samples/{args.dataset}/{args.subset_file}'
    
    args.td_file = f'{args.td_folder}/td-{args.dataset}-{args.model}.pickle'
    
    if args.td:
        os.makedirs(args.td_folder, exist_ok=True)
    if args.model == 'ensembleMoE': 
        args.item_file = f'{args.td_folder}/items-{args.dataset}.npy' # all item added once

    args.noisy_gating = True 
    args.topk_logit_pth = f'{args.td_folder}/moe_top.npy'
    args.moe_pth = f'{args.td_folder}/moe-{args.dataset}.pth'
    run_single_model(args)