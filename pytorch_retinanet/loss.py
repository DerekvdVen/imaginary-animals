from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding # changed .utils to utils


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, alpha=0.25, gamma=2, classWeights=None, batch_size = 4):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.classWeights = classWeights
        self.batch_size = batch_size

    def focal_loss(self, x, y, cw):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        
        # print("x: ", x.size())
        # print("y: ", y.size())

        alpha = self.alpha
        gamma = self.gamma

        t = one_hot_embedding(y.detach().cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = t.cuda()  # [N,20]
        if t.dim()<x.dim():
          t = t.unsqueeze(0)

        
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)

        if cw is not None:
          w *= cw.expand_as(w)

        #print("x: ", x.size())
        #print("t: ", t.size())


        if x.size() != t.size():
          #batchsize = self.batch_size
          # get first value of x and use that instead of batchsize here!
          #print("xsize:", x.size()[0])
          t = t.expand(x.size()[0],-1,-1) #Check with Beni if this makes sense, also check yourself
          #print("t resized: ", t.size())

        return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction='sum')

    def focal_loss_alt(self, x, y, cw):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = self.alpha

        t = one_hot_embedding(y.detach().cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = t.cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*(pt+1e-12).log() / 2

        if cw is not None:
          w *= cw.expand_as(w)

        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.detach().long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        if num_pos:
          mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
          masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
          masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
          loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')
        else:
          loc_loss = torch.tensor(0, dtype=torch.float32, device=loc_preds.device)
        

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        if num_pos:
          pos_neg = cls_targets > -1  # exclude ignored anchors
          mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
          masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
          cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg], self.classWeights)
          loss = (loc_loss+cls_loss)/num_pos
        else:
          loss = self.focal_loss(cls_preds, torch.LongTensor(cls_preds.size(1)).to(cls_preds.device).zero_(), self.classWeights)

        # print('#GT: {}, #Pred: {}'.format(num_pos, torch.sum(cls_preds.sigmoid()>0.05)))

        #print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
        return loss
