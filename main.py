import os
import random
from itertools import product

import numpy as np
from Model.MHRec import MHRec
from arg_parser import parse_args, load_yaml_config
from utils import setup_seed, gpu, get_local_time, topk_sample
import torch
import logging
import dataload
from torch.utils.data import DataLoader
from train_and_evaluate import train_and_evaluate

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parse_args()
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, f"{args.Model}_{args.data_path}").replace("\\", "/") + ".log"

    log_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%a %d %b %Y %H:%M:%S'

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info('============Arguments==============')
    for arg, value in vars(args).items():
        logging.info('%s: %s', arg, value)
    logging.info('local time：%s', get_local_time())
    setup_seed(args.seed)
    device = gpu()
    batch_size = args.batch_size
    num_workers = args.num_workers
    dim_E = args.dim_E
    epochs = args.num_epoch
    feature_embedding = args.feature_embed
    model_name = args.Model
    config = load_yaml_config(model_name)
    reg_weight = args.reg_weight
    learning_rate = args.learning_rate
    dropout = args.dropout
    n_layers = args.n_layers
    ii_topk = args.ii_topk
    uu_topk = args.uu_topk
    ssl_temp = args.ssl_temp
    ssl_alpha = args.ssl_alpha
    # MHRec
    h_layers = args.h_layers
    num_hypernodes = args.num_hypernodes
    beta1 = args.beta1
    beta2 = args.beta2

    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)
    train_dataset = dataload.TrainingDataset(num_user, num_item, user_item_dict, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    if args.Model in ["MHRec"]:
        dir_str = './Data/' + args.data_path
        visual_file_path = os.path.join(dir_str, 'hyperedges_visual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk))
        textual_file_path = os.path.join(dir_str, 'hyperedges_textual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk))

        hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
        hyperedges_textual = np.load(visual_file_path, allow_pickle=True).tolist()
        diffusion_hyperedges_visual = dataload.HyperDiffusionData(num_user, num_item, hyperedges_visual)
        diffusion_hyperedges_textual = dataload.HyperDiffusionData(num_user, num_item, hyperedges_textual)
        diffusionLoader_visual = DataLoader(diffusion_hyperedges_visual, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        diffusionLoader_textual = DataLoader(diffusion_hyperedges_textual, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # ----------------------------------------------

    hyper_ls = []
    for param in config['hyper_parameters']:
        hyper_ls.append(config[param])
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    best_performance = None
    best_params = None
    best_metrics = None

    for idx, hyper_param_combo in enumerate(combinators):
        hyper_param_dict = dict(zip(config['hyper_parameters'], hyper_param_combo))

        logging.info('========={}/{}: Parameters:{}========='.format(
            idx + 1, total_loops, hyper_param_dict))

        for key, value in hyper_param_dict.items():
            setattr(args, key, value)

        model_constructors = {
            'MHRec': lambda: MHRec(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight, args.ii_topk, args.uu_topk, args.num_hypernodes, args.n_layers,
                                   args.h_layers, args.ssl_temp, args.ssl_alpha, args.beta1, args.beta2, device),
        }
        model = model_constructors.get(model_name, lambda: None)()
        model.to(device)
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter requires_grad: {param.requires_grad}")
            # print(f"Parameter data:\n{param.data}")
            print("=" * 30)

        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.learning_rate}])
        if args.Model in ["MHRec"]:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs,
                                                      diffusionLoader_visual=diffusionLoader_visual,
                                                      diffusionLoader_textual=diffusionLoader_textual)
        else:
            current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs)

        current_best_recall = current_best_metrics[20]['recall']
        if best_performance is None or current_best_recall > best_performance:
            best_performance = current_best_recall
            best_params = hyper_param_dict.copy()
            best_metrics = current_best_metrics

    logging.info("Best performance: {:.5f}".format(best_performance))
    logging.info("Best parameters: {}".format(best_params))

    logging.info("Best metrics:")
    for k, metrics in best_metrics.items():
        metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
        logging.info(f"{k}: {' | '.join(metrics_strs)}")
