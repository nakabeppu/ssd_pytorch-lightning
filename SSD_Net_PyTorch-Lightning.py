import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np
from math import sqrt as sqrt
from itertools import product as product
from torch.autograd import Function

def vgg():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)

def extras():
    layers = [
       nn.Conv2d(1024, 256, kernel_size=(1)),
       nn.Conv2d(256, 512, kernel_size=(3), stride=2, padding=1),
       nn.Conv2d(512, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3), stride=2, padding=1),
       nn.Conv2d(256, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3)),
       nn.Conv2d(256, 128, kernel_size=(1)),
       nn.Conv2d(128, 256, kernel_size=(3))
    ]
    return nn.ModuleList(layers)

def loc(num_classes=21):
    layers = [
        nn.Conv2d(512, 4*4, kernel_size=3, padding=1),
        nn.Conv2d(1024, 6*4, kernel_size=3, padding=1),
        nn.Conv2d(512, 6*4, kernel_size=3, padding=1),
        nn.Conv2d(256, 6*4, kernel_size=3, padding=1),
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),
    ]
    return nn.ModuleList(layers)

def conf(num_classes=21):
    layers = [
        nn.Conv2d(512, 4*num_classes, kernel_size=3, padding=1),
        nn.Conv2d(1024, 6*num_classes, kernel_size=3, padding=1),
        nn.Conv2d(512, 6*num_classes, kernel_size=3, padding=1),
        nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)
    ]
    return nn.ModuleList(layers)


class L2Norm(pl.LightningModule):
    def __init__(self,n_channels=512, scale=20):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma)
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
    
class PriorBox(pl.LightningModule):
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 300
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)

def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2], 1)

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths,point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)

def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def nms(boxes, scores, overlap=0.45, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union

        idx = idx[IoU.le(overlap)]
    return keep, count

class MultiBoxLoss(pl.LightningModule):
    def __init__(self, num_classes=21, overlap_thresh=0.5, neg_pos=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        loc_t = torch.cuda.FloatTensor(num, num_priors, 4)
        conf_t = torch.cuda.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].cuda()
            labels = targets[idx][:, -1].cuda()
            defaults = priors.cuda()
            match(self.threshold, truths, defaults,
                  self.variance, labels, loc_t, conf_t, idx)
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), reduction='none')
        num_pos = pos.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0 
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

class Detect(Function):
    def forward(self, output, num_classes, top_k=200, variance=[0.1,0.2], conf_thresh=0.01, nms_thresh=0.45):
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data)
        num = loc_data.size(0)
        output = torch.zeros(num, num_classes, top_k, 5)
        conf_preds = conf_data.transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output

class Net(pl.LightningModule):

    def __init__(self, phase='train', num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.phase = phase
        self.vgg = vgg()
        self.extras = extras()
        self.L2Norm = L2Norm()
        self.loc = loc()
        self.conf = conf(num_classes)
        dbox = PriorBox()
        self.priors = dbox.forward()
        if phase == 'test':
            self.detect = Detect()

    def forward(self, x):
        bs = len(x)
        out, lout, cout = [], [], []
        for i in range(23):
            x = self.vgg[i](x)
        x1 = x
        out.append(self.L2Norm(x1))
        for i in range(23,len(self.vgg)):
            x = self.vgg[i](x)
        out.append(x)
        for i in range(0,8,2):
            x = F.relu(self.extras[i](x), inplace=True)
            x = F.relu(self.extras[i+1](x), inplace=True)
            out.append(x)
        for i in range(6):
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs,-1,4)
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs,-1, self.num_classes)
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs,-1,4)
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs,-1, self.num_classes)
            lout.append(lx)
            cout.append(cx)
        lout = torch.cat(lout, 1)
        cout = torch.cat(cout, 1)
        output = (lout, cout, self.priors)
        if self.phase == 'test':
            return self.detect.apply(output, self.num_classes)
        else:
            return output
        return output

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        xs  = [ torch.cuda.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        targets  = [ torch.cuda.FloatTensor(y) for y in ys ]
        outputs = self(images)
        criterion = MultiBoxLoss(num_classes=self.num_classes, overlap_thresh=0.5, neg_pos=3)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        xs  = [ torch.cuda.FloatTensor(x) for x in xs ]
        images = torch.stack(xs, dim=0)
        targets  = [ torch.cuda.FloatTensor(y) for y in ys ]
        outputs = self(images)
        criterion = MultiBoxLoss(num_classes=self.num_classes, overlap_thresh=0.5, neg_pos=3)
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3,momentum=0.9, weight_decay=5e-4)
        return optimizer