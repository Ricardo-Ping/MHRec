import os
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch
import random

import dataload
from arg_parser import parse_args

random.seed(42)


def topk_sample(k, user_graph_dict, num_user):
    user_graph_index = []
    count_num = 0
    tasike = [0] * k

    for i in range(num_user):
        if len(user_graph_dict[i][0]) < k:
            count_num += 1
            if len(user_graph_dict[i][0]) == 0:
                user_graph_index.append(tasike)
                continue
            user_graph_sample = user_graph_dict[i][0][:k]
            while len(user_graph_sample) < k:
                rand_index = np.random.randint(0, len(user_graph_sample))
                user_graph_sample.append(user_graph_sample[rand_index])
            user_graph_index.append(user_graph_sample)
            continue

        user_graph_sample = user_graph_dict[i][0][:k]
        user_graph_index.append(user_graph_sample)

    return user_graph_index


def get_knn_adj_mat(mm_embeddings, item_topk):
    context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    sim.fill_diagonal_(-float('inf'))
    _, knn_ind = torch.topk(sim, item_topk, dim=-1)

    return knn_ind


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_path
    print(f'Generating u-u matrix for {dataset} ...\n')
    dir_str = './Data/' + dataset
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)

    image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
    text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)

    print(f'uu_topk: {args.uu_topk}, ii_topk: {args.ii_topk}')

    adjusted_item_ids = train_data[:, 1] - num_user
    user_item_graph = sp.coo_matrix((np.ones(len(train_data)),
                                     (train_data[:, 0], adjusted_item_ids)),
                                    shape=(num_user, num_item), dtype=np.float32)

    user_graph_dict = np.load(os.path.join(dir_str, 'user_graph_dict.npy'),
                              allow_pickle=True).item()

    # [num_user, user_topk]
    user_user_k_graph = topk_sample(args.uu_topk, user_graph_dict, num_user)

    visual_adj_file = os.path.join(dir_str, 'ii_visual_{}.pt'.format(args.ii_topk))
    textual_adj_file = os.path.join(dir_str, 'ii_textual_{}.pt'.format(args.ii_topk))

    if os.path.exists(visual_adj_file) and os.path.exists(textual_adj_file):
        # [num_item, item_topk]
        item_item_k_visual_graph = torch.load(visual_adj_file)
        item_item_k_textual_graph = torch.load(textual_adj_file)
    else:
        image_graph = get_knn_adj_mat(image_embedding.weight.detach(), args.ii_topk)
        item_item_k_visual_graph = image_graph
        text_graph = get_knn_adj_mat(text_embedding.weight.detach(), args.ii_topk)
        item_item_k_textual_graph = text_graph
        del image_graph
        del text_graph
        torch.save(item_item_k_visual_graph, visual_adj_file)
        torch.save(item_item_k_textual_graph, textual_adj_file)

    visual_file_name = 'hyperedges_visual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk)
    textual_file_name = 'hyperedges_textual_u{}_i{}.npy'.format(args.uu_topk, args.ii_topk)

    visual_file_path = os.path.join(dir_str, visual_file_name)
    textual_file_path = os.path.join(dir_str, textual_file_name)

    if os.path.exists(visual_file_path) and os.path.exists(textual_file_path):

        hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
        hyperedges_textual = np.load(textual_file_path, allow_pickle=True).tolist()
    else:
        min_similar_users = 1
        max_similar_users = args.uu_topk

        min_similar_items = 1
        max_similar_items = args.ii_topk

        hyperedges_visual = set()
        hyperedges_textual = set()

        for u_i in train_data:
            u = u_i[0]
            i = u_i[1]
            adjusted_item_index = i - num_user

            num_similar_users = random.randint(min_similar_users, max_similar_users)
            similar_users = user_user_k_graph[u]
            similar_users = similar_users[:num_similar_users]

            num_similar_items = random.randint(min_similar_items, max_similar_items)
            similar_items_visual = item_item_k_visual_graph[adjusted_item_index]
            similar_items_visual = similar_items_visual[:num_similar_items]

            similar_items_textual = item_item_k_textual_graph[adjusted_item_index]
            similar_items_textual = similar_items_textual[:num_similar_items]

            hyperedge_visual = [u] + similar_users + [i] + (similar_items_visual + num_user).tolist()
            hyperedge_textual = [u] + similar_users + [i] + (similar_items_textual + num_user).tolist()

            hyperedge_visual = tuple(sorted(hyperedge_visual))
            hyperedge_textual = tuple(sorted(hyperedge_textual))

            hyperedges_visual.add(hyperedge_visual)
            hyperedges_textual.add(hyperedge_textual)

        hyperedges_visual = list(hyperedges_visual)
        hyperedges_textual = list(hyperedges_textual)

        hyperedges_visual_array = np.array(hyperedges_visual, dtype=object)
        hyperedges_textual_array = np.array(hyperedges_textual, dtype=object)

        np.save(visual_file_path, hyperedges_visual_array, allow_pickle=True)
        np.save(textual_file_path, hyperedges_textual_array, allow_pickle=True)

