import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .loss import BPRLoss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


def _cfg(config, name, default=None):
    """Work with both RecBole Config and ordinary dictionaries."""
    try:
        result = config[name]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if result is None else result


class _SplitEmbedding(nn.Module):
    """Names are intentionally compatible with the supplied Trainer_ours."""

    def __init__(self, count, acc_dim, fair_dim):
        super().__init__()
        self.embedding_org = nn.Embedding(count, acc_dim)
        self.embedding_fair = nn.Embedding(count, fair_dim)

    def forward(self, ids):
        return torch.cat(
            [self.embedding_org(ids), self.embedding_fair(ids)], dim=-1
        )


class PAEF_LightGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config
        # Paper's 64-dimensional embedding refers to the final concatenated
        # representation. Split it between the two optimization branches.
        self.latent_dim = int(_cfg(config, "embedding_size", 64))
        if self.latent_dim < 2:
            raise ValueError("embedding_size must be at least 2")
        self.acc_dim = self.latent_dim // 2
        self.fair_dim = self.latent_dim - self.acc_dim
        self.n_layers = int(_cfg(config, "n_layers", 2))
        self.reg_weight = float(_cfg(config, "reg_weight", 0.0))
        self.require_pow = bool(_cfg(config, "require_pow", False))
        self.fair_weight = float(_cfg(config, "DRM_weight", 1.0))  # paper lambda
        self.tau = float(_cfg(config, "tau", 1.0))
        self.sample_size = int(_cfg(config, "sample_size", 200))
        self.fairness_chunk_size = int(_cfg(config, "fairness_user_chunk_size", 16))

        eval_topk = _cfg(config, "topk", [20])
        if isinstance(eval_topk, (list, tuple)):
            eval_topk = eval_topk[0] if eval_topk else 20
        self.topk_num = int(_cfg(config, "fairness_topk", eval_topk))

        if self.tau <= 0:
            raise ValueError("NeuralSort temperature tau must be positive")
        if self.n_items < 3:
            raise ValueError("PAEF requires at least two non-padding items")
        if self.sample_size < 2:
            raise ValueError("sample_size must be >= 2")
        if self.topk_num < 1:
            raise ValueError("fairness_topk must be >= 1")
        if self.fairness_chunk_size < 1:
            raise ValueError("fairness_user_chunk_size must be >= 1")

        self.sample_size = min(self.sample_size, self.n_items - 1)
        self.topk_num = min(self.topk_num, self.sample_size)

        # Combined representation has exactly `embedding_size` dimensions.
        # Both branches are propagated through the same LightGCN.
        self.user_embedding = _SplitEmbedding(self.n_users, self.acc_dim,
                                              self.fair_dim)
        self.item_embedding = _SplitEmbedding(self.n_items, self.acc_dim,
                                              self.fair_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self._user_interactions = inter_matrix.tocsr()
        self.register_buffer(
            "norm_adj_matrix", self.get_norm_adj_mat(inter_matrix),
            persistent=False,
        )
        self._build_provider_index(dataset)
        self.apply(xavier_uniform_initialization)

    def _build_provider_index(self, dataset):
        """Map valid item IDs to provider IDs, excluding padding ID zero."""
        features = dataset.get_item_feature()
        item_field = self.config["ITEM_ID_FIELD"]
        provider_field = self.config["PRODIVER_ID_FIELD"]  # repo's spelling
        ids = np.asarray(features[item_field].cpu().numpy(), dtype=np.int64)
        providers = np.asarray(features[provider_field].cpu().numpy(), dtype=np.int64)
        valid = ((ids > 0) & (ids < self.n_items) & (providers > 0))
        unique_providers = np.unique(providers[valid])
        if len(unique_providers) == 0:
            raise ValueError("No valid providers found in item features")
        mapping = np.full(self.n_items, -1, dtype=np.int64)
        mapping[ids[valid]] = np.searchsorted(unique_providers, providers[valid])
        counts = np.bincount(mapping[mapping >= 0],
                             minlength=len(unique_providers)).astype(np.float32)
        self.n_groups = int(len(unique_providers))
        self.register_buffer("item_provider_index", torch.from_numpy(mapping),
                             persistent=False)
        self.register_buffer("provider_item_counts", torch.from_numpy(counts),
                             persistent=False)

    def get_norm_adj_mat(self, interactions=None):
        """Symmetrically normalized LightGCN interaction adjacency."""
        if interactions is None:
            interactions = self._user_interactions.tocoo()
        interactions = interactions.tocoo()
        num_nodes = self.n_users + self.n_items
        rows = np.concatenate([interactions.row, interactions.col + self.n_users])
        cols = np.concatenate([interactions.col + self.n_users, interactions.row])
        adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                            shape=(num_nodes, num_nodes)).tocsr()
        
        adj.data[:] = 1.0
        degree = np.asarray(adj.sum(axis=1)).reshape(-1)
        d_inv_sqrt = np.power(np.maximum(degree, 1e-7), -0.5)
        norm = (sp.diags(d_inv_sqrt) @ adj @ sp.diags(d_inv_sqrt)).tocoo()
        indices = torch.from_numpy(np.stack([norm.row, norm.col])).long()
        values = torch.from_numpy(norm.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, norm.shape).coalesce()

    def get_ego_embeddings(self, detach_acc=False, detach_fair=False):
        u_acc = self.user_embedding.embedding_org.weight
        i_acc = self.item_embedding.embedding_org.weight
        u_fair = self.user_embedding.embedding_fair.weight
        i_fair = self.item_embedding.embedding_fair.weight
        if detach_acc:
            u_acc, i_acc = u_acc.detach(), i_acc.detach()
        if detach_fair:
            u_fair, i_fair = u_fair.detach(), i_fair.detach()
        return torch.cat(
            [torch.cat([u_acc, u_fair], dim=-1),
             torch.cat([i_acc, i_fair], dim=-1)], dim=0
        )

    def forward(self, detach_acc=False, detach_fair=False):
        embeddings = self.get_ego_embeddings(detach_acc, detach_fair)
        mean_embeddings = embeddings
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(self.norm_adj_matrix, embeddings)
            mean_embeddings = mean_embeddings + embeddings
        mean_embeddings = mean_embeddings / (self.n_layers + 1)
        return torch.split(mean_embeddings, [self.n_users, self.n_items], dim=0)

    def _invalidate_eval_cache(self):
        self.restore_user_e = None
        self.restore_item_e = None

    def calculate_outer_loss(self, interaction):
        """Leader (Eq. 19): BPR updates only accuracy embeddings."""
        self._invalidate_eval_cache()
        users = interaction[self.USER_ID]
        positives = interaction[self.ITEM_ID]
        negatives = interaction[self.NEG_ITEM_ID]
        user_all, item_all = self.forward(detach_fair=True)
        user_vecs = user_all[users]
        pos_scores = (user_vecs * item_all[positives]).sum(dim=-1)
        neg_scores = (user_vecs * item_all[negatives]).sum(dim=-1)
        rec_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(
            self.user_embedding.embedding_org(users),
            self.item_embedding.embedding_org(positives),
            self.item_embedding.embedding_org(negatives),
            require_pow=self.require_pow,
        )
        return rec_loss + self.reg_weight * reg_loss

    @staticmethod
    def detNeuralSort(scores, tau=1.0, k=None):
        """NeuralSort Eq. (8), shape [users, K, candidate_items]."""
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch_size, n_candidates]")
        if tau <= 0:
            raise ValueError("tau must be positive")
        _, n = scores.shape
        if k is None:
            k = n
        k = min(int(k), n)
        pairwise_abs = (scores.unsqueeze(-1) - scores.unsqueeze(-2)).abs()
        abs_sum = pairwise_abs.sum(dim=-1)
        scaling = n + 1 - 2 * (torch.arange(k, device=scores.device) + 1)
        logits = (scaling.to(scores.dtype)[None, :, None] * scores[:, None, :]
                  - abs_sum[:, None, :]) / tau
        return torch.softmax(logits, dim=-1)

    def _sample_candidates(self, user_ids):
        candidates = np.empty((len(user_ids), self.sample_size), dtype=np.int64)
        for row, user_id in enumerate(user_ids):
            start = self._user_interactions.indptr[user_id]
            end = self._user_interactions.indptr[user_id + 1]
            positives = np.unique(self._user_interactions.indices[start:end])
            positives = positives[(positives > 0) & (positives < self.n_items)]
            has_negatives = len(positives) < self.n_items - 1
            max_positives = self.sample_size - 1 if has_negatives else self.sample_size
            if len(positives) > max_positives:
                positives = np.random.choice(positives, size=max_positives,
                                             replace=False)
            remaining = self.sample_size - len(positives)
            if remaining == 0:
                candidates[row] = positives
                continue

            all_seen = self._user_interactions.indices[start:end]
            seen = set(int(x) for x in all_seen)
            picked = set(int(x) for x in positives)
            negatives = []
            for _ in range(10):
                if len(negatives) == remaining:
                    break
                draws = np.random.randint(1, self.n_items,
                                          size=max(16, 3 * (remaining - len(negatives))))
                for val in draws:
                    v = int(val)
                    if v not in seen and v not in picked:
                        picked.add(v)
                        negatives.append(v)
                        if len(negatives) == remaining:
                            break
            if len(negatives) < remaining:
                universe = np.arange(1, self.n_items)
                remaining_pool = np.setdiff1d(universe, np.fromiter(
                    seen.union(picked), dtype=np.int64), assume_unique=False)
                needed = remaining - len(negatives)
                if len(remaining_pool) < needed:
                    raise RuntimeError("Not enough distinct unseen candidate items")
                negatives.extend(np.random.choice(
                    remaining_pool, size=needed, replace=False).tolist())
            candidates[row] = np.concatenate(
                [positives, np.asarray(negatives, dtype=np.int64)]
            )
        return candidates

    def _fairness_loss(self, user_ids):
        """Eq. (13), using a mini-batch stochastic approximation to user sums."""
        user_all, item_all = self.forward(detach_acc=True)
        device = user_all.device
        k = self.topk_num
        rank_positions = torch.arange(2, k + 2, device=device, dtype=user_all.dtype)
        bias = 1.0 / torch.log2(rank_positions)
        provider_total = torch.zeros(self.n_groups, device=device, dtype=user_all.dtype)
        for start in range(0, len(user_ids), self.fairness_chunk_size):
            chunk_user_ids = user_ids[start: start + self.fairness_chunk_size]
            sample_np = self._sample_candidates(chunk_user_ids)
            sampled_items = torch.as_tensor(sample_np, device=device, dtype=torch.long)
            sampled_users = torch.as_tensor(chunk_user_ids, device=device, dtype=torch.long)
            scores = (user_all[sampled_users].unsqueeze(1) *
                      item_all[sampled_items]).sum(dim=-1)
            soft_perm = self.detNeuralSort(scores, tau=self.tau, k=k)
            item_exposure = (soft_perm * bias[None, :, None]).sum(dim=1)

            provider_idx = self.item_provider_index[sampled_items].reshape(-1)
            exposure = item_exposure.reshape(-1)
            valid = provider_idx >= 0
            # Avoid massive dense [providers x users*candidates] masks.
            provider_total = provider_total.scatter_add(
                0, provider_idx[valid], exposure[valid]
            )
        
        provider_exposure = provider_total / self.provider_item_counts.clamp_min(1.0)
        ideal = provider_exposure.mean()
        return ((provider_exposure - ideal) ** 2).sum()

    def calculate_inner_loss(self, interaction):
        """Follower (Eq. 20): lambda * squared provider exposure disparity."""
        self._invalidate_eval_cache()
        ids = interaction[self.USER_ID].detach().cpu().numpy()
        unique_users = np.unique(ids).astype(np.int64, copy=False)
        return self.fair_weight * self._fairness_loss(unique_users)

    def calculate_loss(self, interaction):
        """Compatibility with RecBole's ordinary Trainer (not bilevel)."""
        return self.calculate_outer_loss(interaction) + self.calculate_inner_loss(interaction)

    def predict(self, interaction):
        users, items = self.forward()
        return (users[interaction[self.USER_ID]] *
                items[interaction[self.ITEM_ID]]).sum(dim=-1)

    def full_sort_predict(self, interaction):
        users, items = self.forward()
        scores = users[interaction[self.USER_ID]] @ items.T
        return scores.reshape(-1)
