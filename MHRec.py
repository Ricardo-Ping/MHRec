
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math
import scipy.sparse as sp
from arg_parser import parse_args
import os
import torch_sparse

args = parse_args()


class GCNLayer(nn.Module):
    def __init__(self, device):
        super(GCNLayer, self).__init__()
        self.device = device

    def forward(self, adj, embeds, flag=True):
        adj = adj.to(self.device)
        embeds = embeds.to(self.device)
        if flag:
            return torch.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, H, X):
        E = torch.sparse.mm(H.transpose(0, 1), X)

        H = H.coalesce()
        indices = H.indices()
        values = H.values()

        node_indices = indices[0]
        hyperedge_indices = indices[1]

        X_i = X[node_indices]
        E_j = E[hyperedge_indices]

        concat = torch.cat([X_i, E_j], dim=1)
        e = torch.matmul(concat, self.a).squeeze()
        e_exp = torch.exp(e)
        node_attention_sums = torch.zeros(X.size(0), device=X.device)
        node_attention_sums = node_attention_sums.index_add(0, node_indices, e_exp)
        node_attention_sums_nz = node_attention_sums[node_indices] + 1e-16
        alpha = e_exp / node_attention_sums_nz

        m = alpha.unsqueeze(-1) * E_j
        X_out = torch.zeros_like(X)
        X_out = X_out.index_add(0, node_indices, m)

        return X_out


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims  # [num_user + num_item, 1000]
        self.out_dims = out_dims  # [1000, num_user + num_item]
        self.time_emb_dim = emb_size  # 64
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # [num_user + num_item + 64, 1000]
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims  # [1000, num_user + num_item]
        # num_item + 64 >> 1000
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 1000 >> num_item
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                self.time_emb_dim // 2)).cuda()

        temp = timesteps[:, None].float() * freqs[None]

        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)

        # [batchsize, num_item + 64]
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)  # [batchsize, num_item]

        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)

            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start

        one_minus_alpha_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        return alpha_t + one_minus_alpha_t

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)
        model_log_variance = self.posterior_log_variance_clipped

        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                                x.shape) * model_output + self._extract_into_tensor(
            self.posterior_mean_coef2, t, x.shape) * x)

        return model_mean, model_log_variance

    def training_losses(self, model, x_start, itmEmbeds, model_feats):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)

        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        return diff_loss

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class MHRec(nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight, ii_topk,
                 uu_topk, num_hypernodes, n_layers, h_layers, ssl_temp, ssl_alpha, beta1, beta2, device):
        super(MHRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.device = device
        self.item_topk = ii_topk
        self.user_topk = uu_topk
        self.hyperedges_visual = None
        self.hyperedges_textual = None
        self.n_layers = n_layers
        self.h_layers = h_layers
        self.num_hypernodes = num_hypernodes
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = 5
        self.noise_scale = 0.1
        self.noise_min = 0.0001
        self.noise_max = 0.02
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha

        self.v_feat = v_feat.clone().detach().to(self.device)
        self.t_feat = t_feat.clone().detach().to(self.device)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.dim_E)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.dim_E)
        nn.init.xavier_uniform_(self.image_trs.weight)
        nn.init.xavier_uniform_(self.text_trs.weight)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_embedding.weight)
        self.user_visual_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_visual_embedding.weight)
        self.user_textual_embedding = nn.Embedding(self.num_user, self.dim_E)
        nn.init.xavier_normal_(self.user_textual_embedding.weight)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.gcnLayers = nn.Sequential(*[GCNLayer(self.device) for i in range(self.n_layers)])
        self.hypergraphLayersVisual = nn.Sequential(
            *[HypergraphAttentionLayer(self.dim_E, self.dim_E) for _ in range(self.h_layers)]
        )
        self.hypergraphLayersTextual = nn.Sequential(
            *[HypergraphAttentionLayer(self.dim_E, self.dim_E) for _ in range(self.h_layers)]
        )

        adjusted_item_ids = edge_index[:, 1] - self.num_user
        self.user_item_graph = sp.coo_matrix((np.ones(len(edge_index)),
                                              (edge_index[:, 0], adjusted_item_ids)),
                                             shape=(self.num_user, self.num_item), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)

        dataset = args.data_path
        self.dir_str = './Data/' + dataset

        self.pre_processing()

        dims = '[1000]'
        out_dims = eval(dims) + [num_user + num_item]  # [1000, num_user + num_item]
        in_dims = out_dims[::-1]  # [num_user + num_item, 1000]
        norm = False
        d_emb_size = 10
        self.denoise_model_image = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)
        self.denoise_model_text = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)

        self.image_diffusion_model = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps).to(
            self.device)
        self.text_diffusion_model = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps).to(
            self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        inter_M = self.user_item_graph
        inter_M_t = self.user_item_graph.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        rows_and_cols = np.array([row, col])
        i = torch.tensor(rows_and_cols, dtype=torch.long)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        return SparseL

    def construct_hypergraph(self):
        num_nodes = self.num_user + self.num_item

        num_visual_hyperedges = len(self.hyperedges_visual)
        num_textual_hyperedges = len(self.hyperedges_textual)

        rows_visual = []
        cols_visual = []
        data_visual = []

        for i, hyperedge in enumerate(self.hyperedges_visual):
            for node in hyperedge:
                rows_visual.append(node)
                cols_visual.append(i)
                data_visual.append(1)

        H_visual = sp.coo_matrix((data_visual, (rows_visual, cols_visual)),
                                 shape=(num_nodes, num_visual_hyperedges), dtype=np.float32)
        self.H_visual = H_visual  # (num_nodes, num_hyperedges)

        rows_textual = []
        cols_textual = []
        data_textual = []

        for i, hyperedge in enumerate(self.hyperedges_textual):
            for node in hyperedge:
                rows_textual.append(node)
                cols_textual.append(i)
                data_textual.append(1)

        H_textual = sp.coo_matrix((data_textual, (rows_textual, cols_textual)),
                                  shape=(num_nodes, num_textual_hyperedges), dtype=np.float32)
        self.H_textual = H_textual

    def generate_G_from_H(self, H):
        DV = np.array(H.sum(axis=1)).squeeze()
        DV_inv_sqrt = np.power(DV, -0.5, where=DV != 0)
        DV_inv_sqrt[DV == 0] = 0.

        DE = np.array(H.sum(axis=0)).squeeze()
        DE_inv = np.power(DE, -1.0, where=DE != 0)
        DE_inv[DE == 0] = 0.

        W = np.ones(H.shape[1])
        W = sp.diags(W)

        # D_v^{-1/2} * H
        Dv_inv_sqrt = sp.diags(DV_inv_sqrt)
        H_normalized = Dv_inv_sqrt @ H @ W

        # H_normalized * D_e^{-1}
        De_inv = sp.diags(DE_inv)
        H_normalized = H_normalized @ De_inv

        # G = H_normalized * H_normalized^T
        G = H_normalized @ H_normalized.transpose().tocsr()

        G = G.tocoo()
        indices = torch.from_numpy(np.vstack((G.row, G.col))).long()
        values = torch.from_numpy(G.data).float()
        shape = G.shape
        G = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(self.device)
        return G

    def pre_processing(self):
        self.user_graph_dict = np.load(os.path.join(self.dir_str, 'user_graph_dict.npy'),
                                       allow_pickle=True).item()

        # [num_user, user_topk]
        self.user_user_k_graph = self.topk_sample(self.user_topk)

        visual_adj_file = os.path.join(self.dir_str, 'ii_visual_{}.pt'.format(self.item_topk))
        textual_adj_file = os.path.join(self.dir_str, 'ii_textual_{}.pt'.format(self.item_topk))

        if os.path.exists(visual_adj_file) and os.path.exists(textual_adj_file):
            # [num_item, item_topk]
            self.item_item_k_visual_graph = torch.load(visual_adj_file)
            self.item_item_k_textual_graph = torch.load(textual_adj_file)
        else:
            image_graph = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.item_item_k_visual_graph = image_graph
            text_graph = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.item_item_k_textual_graph = text_graph
            del image_graph
            del text_graph
            torch.save(self.item_item_k_visual_graph, visual_adj_file)
            torch.save(self.item_item_k_textual_graph, textual_adj_file)

        visual_file_name = 'hyperedges_visual_u{}_i{}.npy'.format(self.user_topk, self.item_topk)
        textual_file_name = 'hyperedges_textual_u{}_i{}.npy'.format(self.user_topk, self.item_topk)

        visual_file_path = os.path.join(self.dir_str, visual_file_name)
        textual_file_path = os.path.join(self.dir_str, textual_file_name)

        if os.path.exists(visual_file_path) and os.path.exists(textual_file_path):
            self.hyperedges_visual = np.load(visual_file_path, allow_pickle=True).tolist()
            self.hyperedges_textual = np.load(textual_file_path, allow_pickle=True).tolist()
        else:
            self.hyperedges_visual = []  # [len(edge_index), 2 + self.user_topk + self.item_topk]
            self.hyperedges_textual = []

            for u_i in self.edge_index:
                u = u_i[0]
                i = u_i[1]
                adjusted_item_index = i - self.num_user

                similar_users = self.user_user_k_graph[u]

                similar_items_visual = self.item_item_k_visual_graph[adjusted_item_index]  # Tensor of indices
                similar_items_textual = self.item_item_k_textual_graph[adjusted_item_index]

                hyperedge_visual = [u] + similar_users + [i] + (similar_items_visual + self.num_user).tolist()
                hyperedge_textual = [u] + similar_users + [i] + (similar_items_textual + self.num_user).tolist()

                self.hyperedges_visual.append(hyperedge_visual)
                self.hyperedges_textual.append(hyperedge_textual)

            np.save(visual_file_path, self.hyperedges_visual)
            np.save(textual_file_path, self.hyperedges_textual)


    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        tasike = [0] * k

        for i in range(self.num_user):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                user_graph_index.append(user_graph_sample)
                continue

            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_index.append(user_graph_sample)

        return user_graph_index

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        sim.fill_diagonal_(-float('inf'))
        _, knn_ind = torch.topk(sim, self.item_topk, dim=-1)

        return knn_ind

    def getItemEmbeds(self):
        return self.item_embedding.weight

    def getUserEmbeds(self):
        return self.user_embedding.weight

    def getUserEmbeds_visual(self):
        return self.user_visual_embedding.weight

    def getUserEmbeds_textual(self):
        return self.user_textual_embedding.weight

    def getImageFeats(self):
        v_embedding = self.image_trs(self.image_embedding.weight)
        return v_embedding

    def getTextFeats(self):
        t_embedding = self.text_trs(self.text_embedding.weight)
        return t_embedding

    def convert_scipy_to_torch_sparse(self, coo_matrix):
        coo = coo_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(self.device)

    def forward(self):
        v_embedding = self.image_trs(self.image_embedding.weight)  # (num_item, dim_E)
        t_embedding = self.text_trs(self.text_embedding.weight)  # (num_item, dim_E)

        weight = self.softmax(self.modal_weight)

        user_v_embedding = self.user_visual_embedding.weight  # (num_user, dim_E)
        user_t_embedding = self.user_textual_embedding.weight  # (num_user, dim_E)

        embedsImageAdj = torch.cat([user_v_embedding, F.normalize(v_embedding)], dim=0)
        embedsImageAdjLst = [embedsImageAdj]
        for gcn in self.hypergraphLayersVisual:
            embedsImageAdj = gcn(self.G_visual, embedsImageAdjLst[-1])
            embedsImageAdj += embedsImageAdjLst[-1]
            embedsImageAdj = F.dropout(embedsImageAdj, 0.5)
            embedsImageAdjLst.append(embedsImageAdj)
        embedsImage = torch.mean(torch.stack(embedsImageAdjLst), dim=0)

        embedsImage_ = torch.cat([user_v_embedding, F.normalize(v_embedding)], dim=0)
        embedsImage_Lst = [embedsImage_]
        for gcn in self.gcnLayers:
            embedsImage_ = gcn(self.adj, embedsImage_Lst[-1])
            embedsImage_Lst.append(embedsImage_)
        # embedsImage_ = sum(embedsImage_Lst)
        embedsImage_ = torch.mean(torch.stack(embedsImage_Lst), dim=0)
        embedsImage += self.beta1 * embedsImage_

        embedsTextAdj = torch.cat([user_t_embedding, F.normalize(t_embedding)], dim=0)
        embedsTextAdjLst = [embedsTextAdj]
        for gcn in self.hypergraphLayersTextual:
            embedsTextAdj = gcn(self.G_textual, embedsTextAdjLst[-1])
            embedsTextAdj += embedsTextAdjLst[-1]
            embedsTextAdj = F.dropout(embedsTextAdj, 0.5)
            embedsTextAdjLst.append(embedsTextAdj)
        embedsText = torch.mean(torch.stack(embedsTextAdjLst), dim=0)

        embedsText_ = torch.cat([user_t_embedding, F.normalize(t_embedding)], dim=0)
        embedsText_Lst = [embedsText_]
        for gcn in self.gcnLayers:
            embedsText_ = gcn(self.adj, embedsText_Lst[-1])
            embedsText_Lst.append(embedsText_)
        # embedsText_ = sum(embedsText_Lst)
        embedsText_ = torch.mean(torch.stack(embedsText_Lst), dim=0)
        embedsText += self.beta1 * embedsText_

        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        embeds = torch.concat([self.user_embedding.weight, self.item_embedding.weight])
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(self.adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = torch.mean(torch.stack(embedsLst), dim=0)

        all_embs = embeds + self.beta2 * F.normalize(embedsModal)

        self.result = all_embs

        return all_embs[:self.num_user], all_embs[self.num_user:], embedsImage, embedsText, embeds

    def contrastLoss(self, embeds1, embeds2, nodes):
        embeds1 = F.normalize(embeds1, p=2)
        embeds2 = F.normalize(embeds2, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / self.ssl_temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / self.ssl_temp).sum(-1) + 1e-8
        return -torch.log(nume / deno).mean()

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        user_embeddings = user_emb[users]
        pos_item_embeddings = item_emb[pos_items]
        neg_item_embeddings = item_emb[neg_items]

        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
        user_embeddings = u_g[users]  # (batch_size, dim_E)
        pos_item_embeddings = i_g[pos_items]  # (batch_size, dim_E)
        neg_item_embeddings = i_g[neg_items]  # (batch_size, dim_E)

        user_initial_embeddings = torch.cat([
            self.user_embedding.weight[users],
            self.user_visual_embedding.weight[users],
            self.user_textual_embedding.weight[users]
        ], dim=1)  # (batch_size, dim_E * 3)

        pos_item_initial_embeddings = torch.cat([
            self.item_embedding.weight[pos_items],
            self.image_trs(self.image_embedding.weight[pos_items]),
            self.text_trs(self.text_embedding.weight[pos_items])
        ], dim=1)  # (batch_size, dim_E * 3)

        neg_item_initial_embeddings = torch.cat([
            self.item_embedding.weight[neg_items],
            self.image_trs(self.image_embedding.weight[neg_items]),
            self.text_trs(self.text_embedding.weight[neg_items])
        ], dim=1)  # (batch_size, dim_E * 3)

        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2) +
                torch.mean(user_initial_embeddings ** 2) + torch.mean(pos_item_initial_embeddings ** 2) + torch.mean(
            neg_item_initial_embeddings ** 2)
        )

        return reg_loss

    def loss(self, users, pos_items, neg_items, G_visual, G_textual):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        self.G_visual = G_visual
        self.G_textual = G_textual

        u_embeddings, i_embeddings, embeds_v, embeds_t, embeds_g = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, u_embeddings, i_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, u_embeddings, i_embeddings)

        ssl_loss_1 = self.contrastLoss(embeds_g[:self.num_user], embeds_t[:self.num_user], users) * self.ssl_alpha
        ssl_loss_2 = self.contrastLoss(embeds_g[self.num_user:], embeds_v[self.num_user:], pos_items) * self.ssl_alpha
        ssl_loss_3 = self.contrastLoss(embeds_g[:self.num_user], embeds_v[:self.num_user], users) * self.ssl_alpha
        ssl_loss_4 = self.contrastLoss(embeds_g[self.num_user:], embeds_t[self.num_user:], pos_items) * self.ssl_alpha
        ssl_loss = ssl_loss_3 + ssl_loss_4 + ssl_loss_1 + ssl_loss_2

        loss = bpr_loss + reg_loss + ssl_loss

        return loss

    def gene_ranklist(self, topk=50):
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu()

        all_index_of_rank_list = torch.LongTensor([])

        score_matrix = torch.matmul(user_tensor, item_tensor.t())

        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            score_matrix[row][col] = 1e-6

        _, index_of_rank_list_train = torch.topk(score_matrix, topk)
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

        return all_index_of_rank_list
