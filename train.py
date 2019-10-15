"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import torch
import torch.nn as nn
import utils


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_zcore_with_logits(logits, labels):
    logits = torch.sigmoid(logits).data.round().byte()
    labels = labels.byte()
    scores = ~logits ^ labels # and operator
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0, logger=None, save_one_ckpt=True):
    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25
    dset = train_loader.dataset.dataset

    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt
    if logger is None:
        logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    model_path = os.path.join(output, 'model_epoch-1.pth')

    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        train_zcore = 0
        total_norm = 0
        count_norm = 0
        n_answer_type = torch.zeros(len(dset.idx2type))
        score_answer_type = torch.zeros(len(dset.idx2type))
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])

        for i, (v, b, q, a, c, at) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            q = q.cuda()
            a = a.cuda()
            c = c.cuda().unsqueeze(-1).float()
            at = at.cuda()
            answer_type = torch.zeros(v.size(0), len(dset.idx2type)).cuda()
            answer_type.scatter_(1, at.unsqueeze(1), 1)

            pred, conf, att = model(v, b, q, a, c)
            loss = instance_bce_with_logits(pred, a)
            loss.backward(retain_graph=True)
            losz = instance_bce_with_logits(conf, c)
            losz.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data)
            type_score = batch_score.sum(-1, keepdim=True) * answer_type
            batch_score = batch_score.sum()

            total_loss += loss.item() * v.size(0)
            train_score += batch_score.item()

            batch_zcore = compute_zcore_with_logits(conf, c.data).sum()
            train_zcore += batch_zcore.item()

            n_answer_type += answer_type.sum(0).cpu()
            score_answer_type += type_score.sum(0).cpu()

        total_loss /= N
        train_score = 100 * train_score / N
        train_zcore = 100 * train_zcore / N
        if None != eval_loader:
            model.train(False)
            eval_score, eval_zcore, bound, entropy, val_n_answer_type, val_score_answer_type = evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f, confidence: %.2f' % (total_loss, total_norm/count_norm, train_score, train_zcore))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tconfidence: %.2f (%.2f)' % (100 * eval_zcore, 100))

        if eval_loader is not None and entropy is not None:
            info = ''
            for i in range(entropy.size(0)):
                info = info + ' %.2f' % entropy[i]
            logger.write('\tentropy: ' + info)

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch >= saving_epoch):
            if save_one_ckpt and os.path.exists(model_path):
                os.remove(model_path)
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            best_type = val_score_answer_type
            if eval_loader is not None:
                best_eval_score = eval_score
    return best_eval_score, bound, n_answer_type, val_n_answer_type, score_answer_type/n_answer_type, best_type/val_n_answer_type


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    zcore = 0
    upper_bound = 0
    num_data = 0
    dset = dataloader.dataset.dataset
    n_answer_type = torch.zeros(len(dset.idx2type))
    score_answer_type = torch.zeros(len(dset.idx2type))
    entropy = None
    if hasattr(model.module, 'glimpse'):
        entropy = torch.Tensor(model.module.glimpse).zero_().cuda()

    for i, (v, b, q, a, c, at) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        c = c.cuda().unsqueeze(-1).float()
        at = at.cuda()
        answer_type = torch.zeros(v.size(0), len(dset.idx2type)).cuda()
        answer_type.scatter_(1, at.unsqueeze(1), 1)

        pred, conf, att = model(v, b, q, a, c)
        batch_score = compute_score_with_logits(pred, a.data)
        type_score = batch_score.sum(-1, keepdim=True) * answer_type
        batch_score = batch_score.sum()
        batch_zcore = compute_zcore_with_logits(conf, c.data).sum()
        score += batch_score.item()
        zcore += batch_zcore.item()

        n_answer_type += answer_type.sum(0).cpu()
        score_answer_type += type_score.sum(0).cpu()

        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse:
            entropy += calc_entropy(att.data)[:model.module.glimpse]

    score = score / len(dataloader.dataset)
    zcore = zcore / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)

    return score, zcore, upper_bound, entropy, n_answer_type, score_answer_type


def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + utils.EPS).log()).sum(2).sum(0) # g
