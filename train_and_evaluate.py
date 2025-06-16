
import random
import time
import logging
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
from utils import EarlyStopping, gene_metrics
import scipy.sparse as sp
from arg_parser import parse_args
from collections import defaultdict

args = parse_args()
topk = args.topk


def train(model, train_loader, optimizer, diffusionLoader_visual=None, diffusionLoader_textual=None):
    model.train()
    sum_loss = 0.0

    if args.Model in ["MHRec"]:
        epDiLoss_image, epDiLoss_text = 0, 0
        denoise_opt_image = torch.optim.Adam(model.denoise_model_image.parameters(), lr=args.learning_rate,
                                             weight_decay=0)
        denoise_opt_text = torch.optim.Adam(model.denoise_model_text.parameters(), lr=args.learning_rate,
                                            weight_decay=0)
        logging.info('Start to visual hyperedges diffusion')
        for i, batch in enumerate(diffusionLoader_visual):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = model.getItemEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()
            uEmbeds_visual = model.getUserEmbeds_visual().detach()
            image_feats = model.getImageFeats().detach()

            combined_node_embeds = torch.cat([uEmbeds, iEmbeds], dim=0)  # [num_users + num_items, embedding_dim]
            combined_visual_embeds = torch.cat([uEmbeds_visual, image_feats],
                                               dim=0)  # [num_users + num_items, embedding_dim]

            denoise_opt_image.zero_grad()
            diff_loss_image = model.image_diffusion_model.training_losses(model.denoise_model_image, batch_item,
                                                                          combined_node_embeds,
                                                                          combined_visual_embeds)
            # loss_image = diff_loss_image.mean() + gc_loss_image.mean() * model.e_loss
            loss_image = diff_loss_image.mean()
            epDiLoss_image += loss_image.item()
            loss_image.backward()
            denoise_opt_image.step()

            logging.info('Diffusion Step %d/%d; Diffusion Loss %.6f' % (
                i, diffusionLoader_visual.dataset.__len__() // args.batch_size, loss_image.item()))

        logging.info('Start to textual hyperedges diffusion')
        for i, batch in enumerate(diffusionLoader_textual):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = model.getItemEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()
            uEmbeds_textual = model.getUserEmbeds_textual().detach()
            textual_feats = model.getTextFeats().detach()

            combined_node_embeds = torch.cat([uEmbeds, iEmbeds], dim=0)  # [num_users + num_items, embedding_dim]
            combined_textual_embeds = torch.cat([uEmbeds_textual, textual_feats],
                                                dim=0)  # [num_users + num_items, embedding_dim]

            denoise_opt_text.zero_grad()
            diff_loss_text = model.text_diffusion_model.training_losses(model.denoise_model_text, batch_item,
                                                                        combined_node_embeds,
                                                                        combined_textual_embeds)
            # loss_text = diff_loss_text.mean() + gc_loss_text.mean() * model.e_loss
            loss_text = diff_loss_text.mean()
            epDiLoss_text += loss_text.item()
            loss_text.backward()
            denoise_opt_text.step()

            logging.info('Diffusion Step %d/%d; Diffusion Loss %.6f' % (
                i, diffusionLoader_textual.dataset.__len__() // args.batch_size, loss_text.item()))

        logging.info('')
        logging.info('Start to re-build hypergraph matrix')

        with torch.no_grad():
            sampling_noise = False
            sampling_steps = 5

            rows_visual = []
            cols_visual = []
            data_visual = []
            hyperedge_counter = 0

            for _, batch in enumerate(diffusionLoader_visual):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # visual
                denoised_batch_visual = model.image_diffusion_model.p_sample(
                    model.denoise_model_image, batch_item,
                    sampling_steps, sampling_noise
                )
                _, indices_visual = torch.topk(denoised_batch_visual, k=model.num_hypernodes)

                batch_size = batch_index.size(0)

                hyperedge_indices = np.arange(hyperedge_counter, hyperedge_counter + batch_size)
                hyperedge_counter += batch_size

                nodes = indices_visual.cpu().numpy().reshape(-1)
                hyperedges = np.repeat(hyperedge_indices, model.num_hypernodes)

                data = np.ones_like(nodes, dtype=np.float32)

                rows_visual.append(nodes)
                cols_visual.append(hyperedges)
                data_visual.append(data)

            rows_visual = np.concatenate(rows_visual)
            cols_visual = np.concatenate(cols_visual)
            data_visual = np.concatenate(data_visual)

            num_nodes = model.num_user + model.num_item
            num_hyperedges_visual = hyperedge_counter

            H_visual = sp.coo_matrix(
                (data_visual, (rows_visual, cols_visual)),
                shape=(num_nodes, num_hyperedges_visual),
                dtype=np.float32
            )
            H_visual = H_visual.tocoo()
            indices = torch.from_numpy(np.vstack((H_visual.row, H_visual.col))).long()
            values = torch.from_numpy(H_visual.data).float()
            shape = H_visual.shape
            H_visual = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(model.device)

            del rows_visual, cols_visual, data_visual

            rows_textual = []
            cols_textual = []
            data_textual = []
            hyperedge_counter = 0

            for _, batch in enumerate(diffusionLoader_textual):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                # textual
                denoised_batch_textual = model.text_diffusion_model.p_sample(
                    model.denoise_model_text, batch_item,
                    sampling_steps, sampling_noise
                )
                _, indices_textual = torch.topk(denoised_batch_textual, k=model.num_hypernodes)

                batch_size = batch_index.size(0)
                hyperedge_indices = np.arange(hyperedge_counter, hyperedge_counter + batch_size)
                hyperedge_counter += batch_size

                nodes = indices_textual.cpu().numpy().reshape(-1)
                hyperedges = np.repeat(hyperedge_indices, model.num_hypernodes)

                data = np.ones_like(nodes, dtype=np.float32)

                rows_textual.append(nodes)
                cols_textual.append(hyperedges)
                data_textual.append(data)

            rows_textual = np.concatenate(rows_textual)
            cols_textual = np.concatenate(cols_textual)
            data_textual = np.concatenate(data_textual)

            num_hyperedges_textual = hyperedge_counter

            H_textual = sp.coo_matrix(
                (data_textual, (rows_textual, cols_textual)),
                shape=(num_nodes, num_hyperedges_textual),
                dtype=np.float32
            )
            H_textual = H_textual.tocoo()
            indices = torch.from_numpy(np.vstack((H_textual.row, H_textual.col))).long()
            values = torch.from_numpy(H_textual.data).float()
            shape = H_textual.shape
            H_textual = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(model.device)

            del rows_textual, cols_textual, data_textual

        logging.info('hypergraph matrix built!')

        for users, pos_items, neg_items in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            loss = model.loss(users, pos_items, neg_items, H_visual, H_textual)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
    return sum_loss


def evaluate(model, data, ranklist, topk):
    model.eval()
    with torch.no_grad():
        metrics = gene_metrics(data, ranklist, topk)
    return metrics


def train_and_evaluate(model, train_loader, val_data, test_data, optimizer, epochs, eval_dataloader=None,
                       diffusionLoader=None, test_diffusionLoader=None, train_loader_sec_hop=None,
                       test_loader_sec_hop=None, diffusionLoader_visual=None, diffusionLoader_textual=None,
                       user_homo_loader=None, visual_item_homo_loader=None, textual_item_homo_loader=None):
    model.train()
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(epochs):
        if args.Model in ["MHRec"]:
            loss = train(model, train_loader, optimizer, diffusionLoader_visual=diffusionLoader_visual,
                         diffusionLoader_textual=diffusionLoader_textual)
        else:
            loss = train(model, train_loader, optimizer)
        logging.info("Epoch {}, Loss: {:.5f}".format(epoch + 1, loss))

        model.eval()
        rank_list = model.gene_ranklist()
        val_metrics = evaluate(model, val_data, rank_list, topk)
        test_metrics = evaluate(model, test_data, rank_list, topk)

        logging.info('Validation Metrics:')
        for k, metrics in val_metrics.items():
            metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
            logging.info(f"{k}: {' | '.join(metrics_strs)}")

        logging.info('Test Metrics:')
        for k, metrics in test_metrics.items():
            metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
            logging.info(f"{k}: {' | '.join(metrics_strs)}")

        recall = test_metrics[max(topk)]['recall']
        early_stopping(recall, test_metrics)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_metrics = early_stopping.best_metrics
    logging.info('Best Test Metrics:')
    for k, metrics in best_metrics.items():
        metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
        logging.info(f"{k}: {' | '.join(metrics_strs)}")

    return early_stopping.best_metrics
