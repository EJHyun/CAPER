import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import binary_search
from utils import print_hms
import time
import numpy as np
from tqdm import tqdm
import pickle
import random
random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)
class GCRNN(nn.Module):
    def __init__(self, user_num, comp_num, rel_num, emb_dim, user_id_max, cuda):
        super(GCRNN, self).__init__()
        self.device0 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.device2 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        print("Utilizing", self.device0)
        self.user_num = user_num
        self.comp_num = comp_num
        self.entity_num = user_num + comp_num + 2
        self.ent_embedding_layer = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0)
        self.c0_embedding_layer_u = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0)
        self.rel_embedding_layer = nn.Embedding(rel_num, emb_dim, sparse = False).to(self.device0)
        self.rel_num = rel_num
        num_layers = 1

        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0)
        self.user_id_max = user_id_max

    def forward(self, user_batch, comp_batch, job_batch, start_batch, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        train_t = []
        comp_target_0 = []
        comp_target_1 = []
        for comp_list in comp_batch:
            comp_target_0.append(torch.tensor([comp_list[0]]))
            comp_target_1.append(comp_list[1:])
        comp_target_0.extend(comp_target_1)

        job_target_0 = []
        job_target_1 = []
        for job_list in job_batch:
            job_target_0.append(torch.tensor([job_list[0]]))
            job_target_1.append(job_list[1:])
        job_target_0.extend(job_target_1)

        for time_list, user, comp_list in zip(start_batch, user_batch, comp_batch):
            for time in time_list[1:]:
                train_t.append(time)
                seed_entid.append(user)
        latest_train_time = max(train_t)
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user, comp_list in zip(start_batch, user_batch, comp_batch):
            for time in time_list[1:]:
                seed_list[time].add(user)
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_train_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), sorted = True, return_inverse = True)
        user_emb_0 = self.ent_embedding_layer(torch.tensor(user_batch).to(self.device0))

        user_embs = ent_embs[index_for_ent_emb]
        u_time_embs = torch.cat([user_emb_0, user_embs])

        target_c_embs = self.ent_embedding_layer(torch.cat(comp_target_0).to(self.device0) + self.user_id_max + 1)
        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1)
        pos_score_comp = torch.sum(u_time_embs * target_c_embs, 1).unsqueeze(1)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0))
        comp_loss_procedure = pos_score_comp - torch.logsumexp(all_score_comp, 1).unsqueeze(1)
        comp_NLL_loss = -torch.sum(comp_loss_procedure)

        target_j_embs = self.rel_embedding_layer(torch.cat(job_target_0).to(self.device0)
        all_j_embs = self.rel_embedding_layer(torch.tensor(list(range(self.rel_num//2))).to(self.device0))
        pos_score_job = torch.sum(u_time_embs * target_j_embs, 1).unsqueeze(1)
        all_score_job = torch.matmul(u_time_embs, all_j_embs.transpose(1,0))
        job_loss_procedure = pos_score_job - torch.logsumexp(all_score_job, 1).unsqueeze(1)
        job_NLL_loss = -torch.sum(job_loss_procedure)

        return comp_NLL_loss, job_NLL_loss

    def inference(self, comp_batch, job_batch, user_batch, test_time, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        test_t = []
        for i in range(test_time+1):
            seed_list.append(set())
        for user in user_batch:
            test_t.append(test_time)
            seed_entid.append(user)
            seed_list[test_time].add(user)
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, test_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * test_time + torch.tensor(test_t), sorted = True, return_inverse = True)
        u_time_embs = ent_embs[index_for_ent_emb]

        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1) # (comp_num, emb_dim)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0)) # (N, comp_num)

        all_j_embs = self.rel_embedding_layer(torch.tensor(list(range(self.rel_num//2))).to(self.device0)) # (job_num, emb_dim)
        all_score_job = torch.matmul(u_time_embs, all_j_embs.transpose(1,0)) # (N, job_num)

        return all_score_comp, all_score_job
    

    def msg_GCN(self,edges): 
        return {'m' : edges.src['node_emb'] * self.rel_embedding[edges.data['relation_idx'].type(torch.LongTensor)]}
 

    def reduce_GCN(self,nodes): 
        return {'node_emb2': nodes.mailbox['m'].mean(1)}

    def update_node(self,nodes):
        return {'node_emb': nodes.data['node_emb'] + nodes.data['node_emb2']}
        
    def seq_GCRNN_batch(self, g, splitted_g, latest_train_time, seed_list, history_length, remove_list):

        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        gcn_1hopneighbor_per_time = []
        gcn_seed_2hopedge_per_time = []
        a2 = time.time()
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.comp_num)
        for i in range(latest_train_time, -1, -1):
            check_lifetime[list(seed_list[i])] = history_length

            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())

            hop1_u, hop1_v = splitted_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')

            gcn_seed_per_time.append(list(future_needed_nodes))
            
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))
            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - remove_list[i-1] - set(np.where(check_lifetime==0)[0])
            except:
                pass
        a3 = time.time()
        self.rel_embedding = self.rel_embedding_layer(torch.tensor(range(self.rel_num)).to(self.device0))
        last_entity_index = self.entity_num-2-1
        g.ndata['node_emb'] = self.ent_embedding_layer(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        entity_embs = []
        entity_index = []
        g.register_message_func(self.msg_GCN)
        g.register_reduce_func(self.reduce_GCN)
        for i in range(latest_train_time+1):
            inverse = latest_train_time -i
            if len(gcn_seed_per_time[inverse]) > 0:
                changed = sorted(gcn_seed_per_time[inverse])
                thresh = binary_search(changed, self.user_id_max + 1)
                user_seed_ = changed[:thresh]
                user_seed_ = changed
                user_prev_hn = g.ndata['node_emb'][user_seed_]
                user_prev_cn = g.ndata['cx'][user_seed_]
                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse])
                if edge_num > 0:
                    g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                user_input = g.ndata['node_emb'][user_seed_]
                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))
                g.ndata['node_emb'][user_seed_] = user_hn
                g.ndata['cx'][user_seed_] = user_cn
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i
                entity_embs.append(seed_emb)
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))
        entity_embs = torch.cat(entity_embs).to(self.device1)
        entity_index = torch.cat(entity_index)
        a4 = time.time()
        return entity_embs[entity_index.argsort()]

