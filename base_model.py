"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from attention import BiAttention
from language_model import RnnQuestionEmbedding, BertRnnQuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter


class BanModel(nn.Module):
    def __init__(self, dataset, q_emb, v_att, b_net, q_prj, c_prj, classifiers, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifiers = nn.ModuleList(classifiers)
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels, confidence, img=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        if isinstance(self.q_emb, BertModel):
            q_emb, sentence_embedding = self.q_emb(q, output_all_encoded_layers=False)  # [batch, q_len, q_dim]
        else:
            q_emb = self.q_emb(q)
        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb)  # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifiers[0](q_emb.sum(1))
        logitz = self.classifiers[1](q_emb.sum(1))

        return logits, logitz, att


def build_ban(dataset, num_hid, op='', gamma=4, q_emb_type='bert', on_do_q=False, finetune_q=False):
    if 'bert' in q_emb_type:
        q_emb = BertModel.from_pretrained('bert-base-multilingual-cased')
        q_dim = 768
    elif 'rg' in q_emb_type:
        w_dim = 100
        q_dim = num_hid
        q_emb = RnnQuestionEmbedding(dataset.dictionary.ntoken, w_dim, q_dim, op)
    elif 'pkb' in q_emb_type:
        w_dim = 200
        q_dim = num_hid
        q_emb = RnnQuestionEmbedding(dataset.dictionary.ntoken, w_dim, q_dim, op)

    if 'bertrnn' in q_emb_type:
        q_emb = BertRnnQuestionEmbedding(q_emb, 200, num_hid, op)
        q_dim = num_hid

    if not finetune_q: # Freeze question embedding
        if isinstance(q_emb, BertModel):
            for p in q_emb.parameters():
                p.requires_grad_(False)
        else:
            for p in q_emb.w_emb.parameters():
                p.requires_grad_(False)
    if not on_do_q: # Remove dropout of question embedding
        for m in q_emb.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.

    v_att = BiAttention(dataset.v_dim, q_dim, num_hid, gamma)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(dataset.v_dim, q_dim, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, q_dim], '', .2))
        c_prj.append(FCNet([objects + 1, q_dim], 'ReLU', .0))
    classifiers = [SimpleClassifier(q_dim, num_hid * 2, dataset.num_ans_candidates, .5),
                   SimpleClassifier(q_dim, num_hid * 2, 1, .5)]
    counter = Counter(objects)
    return BanModel(dataset, q_emb, v_att, b_net, q_prj, c_prj, classifiers, counter, op, gamma)
