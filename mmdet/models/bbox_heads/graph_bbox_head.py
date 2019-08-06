import torch.nn as nn
import torch
from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F
from mmdet.core import (weighted_cross_entropy, weighted_smoothl1, accuracy)

@HEADS.register_module
class GraphBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_attr_conv=0,
                 num_rela_conv=0,
                 num_spat_conv=0,
                 with_attr=False,
                 with_rela=False,
                 with_spat=False,
                 num_spat_graph=10,
                 graph_out_channels=256,
                 nf=64,
                 ratio=[4, 2, 1],
                 normalize=None,
                 num_shared_fcs=0,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super(GraphBBoxHead, self).__init__(*args, **kwargs)
        # original FPN head
        self.num_shared_fcs = num_shared_fcs
        self.normalize = normalize
        self.with_bias = normalize is None
        self.fc_out_channels = fc_out_channels
        # add shared convs and fcs
        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.in_channels, num_branch_fcs=self.num_shared_fcs)
        if num_shared_fcs > 0:
            self.cls_last_dim = last_layer_dim
            self.reg_last_dim = last_layer_dim
            self.in_channels = last_layer_dim
        else:
            self.cls_last_dim = self.in_channels
            self.reg_last_dim = self.in_channels

        # corresponding to graph compute
        self.attr_transferW = nn.ModuleList()
        self.rela_transferW = nn.ModuleList()
        self.spat_transferW = nn.ModuleList()
        if with_attr:
            self.attr_convs, _, _ = self._add_conv_fc_branch(num_attr_conv, self.in_channels, nf, ratio)
            self.attr_transferW = nn.Linear(self.in_channels, graph_out_channels)
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        if with_rela:
            self.rela_convs, _, _ = self._add_conv_fc_branch(num_rela_conv, self.in_channels, nf, ratio)
            self.rela_transferW = nn.Linear(self.in_channels, graph_out_channels)
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        if with_spat:
            self.spat_convs, _, _ = self._add_conv_fc_branch(num_spat_conv, 5, nf=5, ratio=[1])
            self.spat_transferW = nn.Linear(self.in_channels, graph_out_channels)
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        self.with_attr = with_attr
        self.with_rela = with_rela
        self.with_spat = with_spat
        self.num_spat_graph = num_spat_graph

        # classifer and bbox regression
        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else
                           4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            in_channels,
                            nf=0,
                            ratio=[0],
                            num_branch_fcs=0):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            assert num_branch_convs == len(ratio) + 1
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim
                                    if i == 0 else conv_out_channels)
                conv_out_channels = (int(nf * ratio[i])
                                     if i < num_branch_convs - 1 else 1)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        conv_out_channels,
                        1,
                        normalize=self.normalize,
                        bias=self.with_bias))

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim
                                  if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        return branch_convs, branch_fcs, last_layer_dim


    def init_weights(self):
        super(GraphBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.attr_transferW, self.rela_transferW, self.spat_transferW]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, geom_f, bs):
        # shared part
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        if x.dim() > 2:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        feat_dim = x.size(1)
        x = x.view(bs, -1, feat_dim)

        # compute A adj matrix
        a_super = []
        enhanced_feat = []
        if self.with_attr or self.with_rela:
            W1 = x.detach().unsqueeze(2)
            W2 = torch.transpose(W1, 1, 2)
            diff_W = torch.abs(W1 - W2)
            diff_W = torch.transpose(diff_W, 1, 3)
            if self.with_attr:
                A_a = diff_W
                for conv in self.attr_convs:
                    A_a = conv(A_a)
                A_a = A_a.contiguous()
                A_a = A_a.squeeze(1)
                a_super.append(A_a)
                # propogation
                enhanced_feat.append(self.propagate_em(x, A_a, self.attr_transferW))

            if self.with_rela:
                A_r = diff_W
                for conv in self.rela_convs:
                    A_r = conv(A_r)
                A_r = A_r.contiguous()
                A_r = A_r.squeeze(1)
                a_super.append(A_r)
                # propogation
                enhanced_feat.append(self.propagate_em(x, A_r, self.rela_transferW))

        if self.with_spat:
            W1 = geom_f.unsqueeze(2)
            W2 = torch.transpose(W1, 1, 2)
            diff_W = W1 - W2
            diff_W = torch.transpose(diff_W, 1, 3)
            Iden = torch.eye(diff_W.size(-1)).cuda()
            A_s = W2.new_zeros((diff_W.size(-1), diff_W.size(-1)))
            for i in range(self.num_spat_graph):
                tmp_A = diff_W
                for conv in self.spat_convs:
                    tmp_A = conv(tmp_A)
                A_s = tmp_A + A_s + Iden
            A_s = A_s.contiguous()
            A_s = A_s.squeeze(1)
            enhanced_feat.append(self.propagate_em(x, A_s, self.spat_transferW))

        enhanced_feat = torch.cat(enhanced_feat, -1)
        # separate branches
        assert len(x.size()) == len(enhanced_feat.size())
        x = torch.cat((x, enhanced_feat), -1)
        x_cls = x.view(-1, x.size(-1))
        x_reg = x.view(-1, x.size(-1))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, a_super

    def loss(self, cls_score, bbox_pred, A_pred, A_gt, labels, label_weights, bbox_targets,
             bbox_weights, reduce=True):
        losses = dict()
        if cls_score is not None:
            losses['loss_cls'] = weighted_cross_entropy(
                cls_score, labels, label_weights, reduce=reduce)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            losses['loss_reg'] = weighted_smoothl1(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=bbox_targets.size(0))
        if A_pred:
            assert len(A_pred) == len(A_gt)
            assert A_pred[0].size() == A_gt[0].size()
            num_a_pred = len(A_pred)
            for i in range(num_a_pred):
                losses['loss_adj' + str(i)] = F.mse_loss(A_pred[i], A_gt[i].detach())
        return losses

    def propagate_em(self, x, A, W):
        A = F.softmax(A, 2)
        x = torch.bmm(A, x)
        x = W(x)
        return x

