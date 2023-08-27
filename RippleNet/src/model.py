import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

class RippleNet(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation)

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)     # entity_emb：Embedding(9366,16)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)      # relation_emb：Embedding(16,256)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)       # transform_matrix：Linear(in_features=16,out_features=16,bias=False)
        self.criterion = nn.BCELoss()   # criterion：BCELoss()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):      # 这里的5个参数都是带cuda的
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)    # item_embeddings = {Tensor:(1024,16)}
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):     # 循环的次数
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))   # h_emb_list = {list:2}   0或1 = {Tensor:(1024,32,16)}
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(      # r_emb_list = {list:2}   0或1 = {Tensor:(1024,32,16,16)}
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))   # t_emb_list = {list:2}   0或1 = {Tensor:(1024,32,16)}

        o_list, item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings
        )       # o_list = {list:2}    0或1 = {Tensor:(1024,16)}     # item_embeddings = {Tensor:(1024,16)}
        scores = self.predict(item_embeddings, o_list)      # scores = {Tensor:(1024,)}

        return_dict = self._compute_loss(       # 此时的return_dict为：dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores      # 再加上scores的值进字典中

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())      # BCELoss(scores, labels.float())，计算损失值

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)    # h_expanded = {Tensor:(1024,32,1,16)}
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)    # t_expanded = {Tensor:(1024,32,16,1)}
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )       # hRt = {Tensor:(1024,32)}
            # 矩阵相乘，torch.mm(x, y) ， 矩阵大小需满足： (i, n)x(n, j)
            # torch.matmul()，如果维度更高呢？前面的维度必须要相同，然后最后面的两个维度符合矩阵相乘的形状限制：i×j，j×k
            # torch.squeeze(a)就是将a中所有为1的维度删掉，不为1的维度没有影响
            # hRt = torch.matmul(h_expanded, r_emb_list[hop])     # hRt = {Tensor:(1024,32,1,16)}
            # hRt = torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)   # hRt = {Tensor:(1024,32,1,1)}

            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss      # 计算kge的损失值

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss         # 计算l2的损失值

        loss = base_loss + kge_loss + l2_loss      # 总损失值为三者的相加
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):   # 循环的次数
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)    # h_expanded = {Tensor:(1024,32,16,1)}

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))   # Rh = {Tensor:(1024,32,16)}

            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)     # v = {Tensor:(1024,16,1)}

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))      # probs = {Tensor:(1024,32)}

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)      # 对probs进行softmax，probs_normalized = {Tensor:(1024,32)}

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)     # probs_expanded = {Tensor:(1024,32,1)}

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)   # o = {Tensor:(1024,16)}
            # o = (t_emb_list[hop] * probs_expanded)  # o = {Tensor:(1024,32,16)}

            item_embeddings = self._update_item_embedding(item_embeddings, o)   # item_embeddings = {Tensor:(1024,16)}，不过相对于原始的item_embeddings，有一些变化
            o_list.append(o)
        return o_list, item_embeddings      # o_list = {list:2}    0或1 = {Tensor:(1024,16)}

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":     # 论文里选择的是这种
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]      # 取最后一跳的
        if self.using_all_hops:     # 如果使用所有跳的话
            for i in range(self.n_hop - 1):
                y += o_list[i]      # 当前最后一跳加上之前跳的

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)   # 直接取最后一跳的，做内积
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()       # scores = {ndarray:(1024,)}
        labels = labels.cpu().numpy()       # labels = {ndarray:(1024,)}
        auc = roc_auc_score(y_true=labels, y_score=scores)      # 求auc的值
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))    # 求acc的值
        return auc, acc
