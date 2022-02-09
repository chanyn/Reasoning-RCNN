import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import sample_bboxes_return_index, bbox2roi, bbox2result, multi_apply


class ThreeStageGraphDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 graph_convolution=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ThreeStageGraphDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_rpn_head(rpn_head)


        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head[0])
            self.bbox_roi_extractor_2 = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head_en = builder.build_bbox_head(bbox_head[1])

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        self.train_cfg = train_cfg

        self.test_cfg = test_cfg
        self.n_graph_node = graph_convolution.n_graph_node

        # Graph Module
        self.latent_graph_channel = graph_convolution.latent_graph_channel
        self.n_kernels = graph_convolution.n_kernels_gc
        self.neigh_size = graph_convolution.neigh_size
        # graph learner
        self.adjacency_learner = GraphLearner(in_feature_dim=1024, combined_feature_dim=256)

        # graph convolution layers
        self.graph_convolution_1 = NeighbourhoodGraphConvolution(bbox_head[0].fc_out_channels+1,
                                                                 self.latent_graph_channel*2, self.n_kernels, 2)
        self.graph_convolution_2 = NeighbourhoodGraphConvolution(self.latent_graph_channel*2,
                                                                 self.latent_graph_channel, self.n_kernels, 2)
        self.dropout = nn.Dropout(p=0.5)
        #self.bn_1 = nn.BatchNorm1d(self.latent_graph_channel*2)
        #self.bn_2 = nn.BatchNorm1d(self.latent_graph_channel)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(ThreeStageGraphDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
            self.bbox_roi_extractor_2.init_weights()
            self.bbox_head_en.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=None,
                      proposals=None):
        batch_size = len(img_meta)
        losses = dict()

        x = self.extract_feat(img)

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_bbox:
            (pos_inds, neg_inds, pos_proposals, neg_proposals, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = multi_apply(
                 sample_bboxes_return_index,
                 proposal_list,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels, label_weights, bbox_targets,
             bbox_weights) = self.bbox_head.get_bbox_target(
                 pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
                 self.train_cfg.rcnn)

            rois = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals, neg_proposals)
            ])
            # TODO: a more flexible way to configurate feat maps
            roi_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights)
            losses.update(loss_bbox)

            #next stage
            # Get weight from fc layer to become the pool
            feature_pool_weight = torch.cat([self.bbox_head.fc_cls.weight, self.bbox_head.fc_cls.bias.unsqueeze(1)], 1).detach()
            # Get the ideal soft weight for each bbox(roi)
            #cls_prob = nn.functional.softmax(cls_score, 1)
            #fatched_mixed_weight = torch.mm(cls_prob, feature_pool_weight)

            max_cls = torch.max(cls_score, 1)
            fatched_mixed_weight = feature_pool_weight[max_cls[1], :]
            # detach here
            img_shapes = [zx['img_shape'] for zx in img_meta]
            refined_rois = self.bbox_head.refine_bboxes(
                rois,
                labels,
                bbox_pred,
                img_shapes,
                gt_labels,
                pos_inds,
                has_gt_in_roi=False)
            refined_rois = [r.detach() for r in refined_rois]

            (pos_inds, neg_inds, pos_proposals, neg_proposals, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = multi_apply(
                 sample_bboxes_return_index,
                 refined_rois,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn2)

            (labels, label_weights, bbox_targets,
             bbox_weights) = self.bbox_head.get_bbox_target(
                 pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
                 self.train_cfg.rcnn2)

            bbox_each = [torch.cat([pos, neg], dim=0) for pos, neg in zip(pos_proposals, neg_proposals)]

            rois_2 = bbox2roi(bbox_each)

            bb=[]
            for one_img_idx, bbox_each_each_img in enumerate(bbox_each):
                _bbox_each = bbox_each_each_img.clone()
                _bbox_each[:, 0] = _bbox_each[:, 0]/img_shapes[one_img_idx][0]
                _bbox_each[:, 1] = _bbox_each[:, 1]/img_shapes[one_img_idx][1]
                _bbox_each[:, 2] = _bbox_each[:, 2]/img_shapes[one_img_idx][0]
                _bbox_each[:, 3] = _bbox_each[:, 3]/img_shapes[one_img_idx][1]
                bb.append(_bbox_each)


            bb = torch.cat(bb, 0).view(batch_size, -1, 4)

            # Compute pseudo coordinates
            # extract bounding boxes and compute centres
            bb_size = (bb[:, :, 2:] - bb[:, :, :2])
            bb_centre = bb[:, :, :2] + 0.5 * bb_size
            # Compute pseudo coordinates
            pseudo_coord = self._compute_pseudo(bb_centre)


            roi_feats_2 = self.bbox_roi_extractor_2(
                x[:self.bbox_roi_extractor_2.num_inputs], rois_2)

            # shared with last fc
            roi_feats_2 = roi_feats_2.view(roi_feats_2.size(0), -1)
            for fc in self.bbox_head.shared_fcs:
                roi_feats_2 = self.relu(fc(roi_feats_2))
            input_graph_learner = roi_feats_2.detach()

            #input_graph_learner = roi_feats_2.mean(3).mean(2)
            input_graph_learner = input_graph_learner.view(batch_size, -1, input_graph_learner.size(-1))
            # Learn adjacency matrix
            adjacency_matrix = self.adjacency_learner(input_graph_learner)
            # Create the right order for fatched_mixed_weight:

            fatched_mixed_weight = fatched_mixed_weight.view(batch_size, -1, self.bbox_head.fc_out_channels + 1)
            input_graph_conv = []
            for one_img_idx in range(fatched_mixed_weight.size(0)):
                one_image_mw = fatched_mixed_weight[one_img_idx]
                pos = one_image_mw[pos_inds[one_img_idx]]
                neg = one_image_mw[neg_inds[one_img_idx]]
                new_mw = torch.cat([pos, neg], dim=0)
                input_graph_conv.append(new_mw)
            input_graph_conv = torch.cat(input_graph_conv, dim=0).view(batch_size, -1,
                                                    self.bbox_head.fc_out_channels + 1)

            # Graph convolution 1
            neighbourhood_image, neighbourhood_pseudo = self._create_neighbourhood(input_graph_conv,
                                                                                   pseudo_coord,
                                                                                   adjacency_matrix,
                                                                                   neighbourhood_size=self.neigh_size,
                                                                                   weight=True)
            hidden_graph_1 = self.graph_convolution_1(
                neighbourhood_image, neighbourhood_pseudo)

            # hidden_graph_1 = self.bn_1(hidden_graph_1)
            hidden_graph_1 = F.relu(hidden_graph_1)
            hidden_graph_1 = self.dropout(hidden_graph_1)

            # graph convolution 2
            hidden_graph_1, neighbourhood_pseudo = self._create_neighbourhood(hidden_graph_1,
                                                                              pseudo_coord,
                                                                              adjacency_matrix,
                                                                              neighbourhood_size=self.neigh_size,
                                                                              weight=False)
            hidden_graph_2 = self.graph_convolution_2(
                hidden_graph_1, neighbourhood_pseudo)

            hidden_graph_2 = hidden_graph_2.view(-1, self.latent_graph_channel)
            # hidden_graph_2 = self.bn_2(hidden_graph_2)
            hidden_graph_2 = F.relu(hidden_graph_2)

            # hidden_graph_2 = self.bn_1(hidden_graph_2)

            cls_score, bbox_pred = self.bbox_head_en(roi_feats_2, hidden_graph_2)

            loss_bbox_2 = self.bbox_head_en.loss(cls_score, bbox_pred, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights)
            loss_bbox_2 = {'loss_cls_2': loss_bbox_2['loss_cls'], 'loss_reg_2': loss_bbox_2['loss_reg'], 'acc_2': loss_bbox_2['acc']}
            losses.update(loss_bbox_2)


        if self.with_mask:
            mask_targets = self.mask_head.get_mask_target(
                pos_proposals, pos_assigned_gt_inds, gt_masks,
                self.train_cfg.rcnn)
            pos_rois = bbox2roi(pos_proposals)
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            torch.cat(pos_gt_labels))
            losses.update(loss_mask)

        return losses


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        # -- get backbone feature, sample proposals
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta,
            self.test_cfg.rpn) if proposals is None else proposals

        #test_combs = [([i], i) for i in range(2)]

        det_bboxes_mul, det_labels_mul = self.simple_test_bboxes_ms(
            x, img_meta, proposal_list, rescale=rescale)
    #
        bbox_result_mul = []
        for i in range(3):
            det_bboxes = det_bboxes_mul[i]
            det_labels = det_labels_mul[i]
            bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            bbox_result_mul.append(bbox_result)

        if not self.with_mask:
            return bbox_result_mul[2]
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_result_mul, segm_results


    #multistage test
    def simple_test_bboxes_ms(self,
                           x,
                           img_meta,
                           proposal_list,
                           rescale=False):
        batch_size = len(img_meta)
        # rpn_outs = self.rpn_head(x)
        # proposal_inputs = rpn_outs + (img_shapes, self.rpn_test_cfg)
        # proposal_list = self.rpn_head.get_proposals(*proposal_inputs)

        # -- get rois by sampling from proposals
        rois = bbox2roi(proposal_list)
        # img_shapes = [zx['img_shape'] for zx in img_meta]
        # img_shape = img_shapes[0]
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        # -- forward each stage
        (rois_mul, bbox_pred_mul, cls_score_mul) = [], [], []
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        rois_mul.append(rois)
        bbox_pred_mul.append(bbox_pred)
        cls_score_mul.append(cls_score)
        refined_rois = self.bbox_head.regress_by_class(
            rois, cls_score, bbox_pred, img_shape)

        # Get weight from fc layer to become the pool
        feature_pool_weight = torch.cat([self.bbox_head.fc_cls.weight, self.bbox_head.fc_cls.bias.unsqueeze(1)],
                                        1).detach()
        # Get the ideal soft weight for each bbox(roi)
        # cls_prob = nn.functional.softmax(cls_score, 1)
        # fatched_mixed_weight = torch.mm(cls_prob, feature_pool_weight)
        max_cls = torch.max(cls_score, 1)
        fatched_mixed_weight = feature_pool_weight[max_cls[1], :]

        _bbox_each = refined_rois[:, 1:].clone()
        _bbox_each[:, 0] = _bbox_each[:, 0] / img_shape[0]
        _bbox_each[:, 1] = _bbox_each[:, 1] / img_shape[1]
        _bbox_each[:, 2] = _bbox_each[:, 2] / img_shape[0]
        _bbox_each[:, 3] = _bbox_each[:, 3] / img_shape[1]

        bb = _bbox_each.view(batch_size, -1, 4)

        # Compute pseudo coordinates
        # extract bounding boxes and compute centres
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])
        bb_centre = bb[:, :, :2] + 0.5 * bb_size
        # Compute pseudo coordinates
        pseudo_coord = self._compute_pseudo(bb_centre)

        #Stage 3
        roi_feats_2 = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], refined_rois)

        # shared with last fc
        roi_feats_2 = roi_feats_2.view(roi_feats_2.size(0), -1)
        for fc in self.bbox_head.shared_fcs:
            roi_feats_2 = self.relu(fc(roi_feats_2))
        input_graph_learner = roi_feats_2.detach()

        #input_graph_learner = roi_feats_2.mean(3).mean(2)
        input_graph_learner = input_graph_learner.view(batch_size, -1, input_graph_learner.size(-1))
        # Learn adjacency matrix
        adjacency_matrix = self.adjacency_learner(input_graph_learner)
        # Create the right order for fatched_mixed_weight:
        input_graph_conv = fatched_mixed_weight.view(batch_size, -1, self.bbox_head.fc_out_channels + 1)

        # Graph convolution 1
        neighbourhood_image, neighbourhood_pseudo = self._create_neighbourhood(input_graph_conv,
                                                                               pseudo_coord,
                                                                               adjacency_matrix,
                                                                               neighbourhood_size=self.neigh_size,
                                                                               weight=True)
        hidden_graph_1 = self.graph_convolution_1(
            neighbourhood_image, neighbourhood_pseudo)

        #hidden_graph_1 = self.bn_1(hidden_graph_1)
        hidden_graph_1 = F.relu(hidden_graph_1)
        hidden_graph_1 = self.dropout(hidden_graph_1)

        # graph convolution 2
        hidden_graph_1, neighbourhood_pseudo = self._create_neighbourhood(hidden_graph_1,
                                                                          pseudo_coord,
                                                                          adjacency_matrix,
                                                                          neighbourhood_size=self.neigh_size,
                                                                          weight=False)
        hidden_graph_2 = self.graph_convolution_2(
            hidden_graph_1, neighbourhood_pseudo)

        hidden_graph_2 = hidden_graph_2.view(-1, self.latent_graph_channel)
        #hidden_graph_2 = self.bn_2(hidden_graph_2)
        hidden_graph_2 = F.relu(hidden_graph_2)

        # hidden_graph_2 = self.bn_1(hidden_graph_2)

        cls_score, bbox_pred = self.bbox_head_en(roi_feats_2, hidden_graph_2)

        rois_mul.append(refined_rois)
        bbox_pred_mul.append(bbox_pred)
        cls_score_mul.append(cls_score)

        #get det bboxes
        det_bboxes_mul, det_labels_mul = [], []
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois_mul[0], cls_score_mul[0],
            bbox_pred_mul[0],
            img_shape,
            scale_factor,
            rescale=rescale,
            nms_cfg=self.test_cfg.rcnn)
        det_bboxes_mul.append(det_bboxes)
        det_labels_mul.append(det_labels)

        # Stage 3
        det_bboxes, det_labels = self.bbox_head_en.get_det_bboxes(
            rois_mul[1], cls_score_mul[1],
            bbox_pred_mul[1],
            img_shape,
            scale_factor,
            rescale=rescale,
            nms_cfg=self.test_cfg.rcnn)
        det_bboxes_mul.append(det_bboxes)
        det_labels_mul.append(det_labels)

        # Stage all
        rois_all = torch.cat(rois_mul)
        bbox_pred_all = torch.cat(bbox_pred_mul)
        cls_score_all = torch.cat(cls_score_mul)
        det_bboxes, det_labels = self.bbox_head_en.get_det_bboxes(
            rois_all, cls_score_all,
            bbox_pred_all,
            img_shape,
            scale_factor,
            rescale=rescale,
            nms_cfg=self.test_cfg.rcnn)
        det_bboxes_mul.append(det_bboxes)
        det_labels_mul.append(det_labels)

        return det_bboxes_mul, det_labels_mul

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def _compute_pseudo(self, bb_centre):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        ## Inputs:
        - bb_centre (batch_size, K, coord_dim)
        - polar (bool: polar or euclidean coordinates)
        ## Returns:
        - pseudo_coord (batch_size, K, K, coord_dim)
        '''

        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord

    def _create_neighbourhood(self,
                              features,
                              pseudo_coord,
                              adjacency_matrix,
                              neighbourhood_size=16,
                              weight=True):

        '''

        Creates a neighbourhood system for each graph node/image object

        ## Inputs:
        - features (batch_size, K, feat_dim): input image features
        - pseudo_coord (batch_size, K, K, coord_dim): pseudo coordinates for graph convolutions
        - adjacency_matrix (batch_size, K, K): learned adjacency matrix
        - neighbourhood_size (int)
        - weight (bool): specify if the features should be weighted by the adjacency matrix values

        ## Returns:
        - neighbourhood_image (batch_size, K, neighbourhood_size, feat_dim)
        - neighbourhood_pseudo (batch_size, K, neighbourhood_size, coord_dim)
        '''

        # Number of graph nodes
        K = features.size(1)

        # extract top k neighbours for each node and normalise
        top_k, top_ind = torch.topk(
            adjacency_matrix, k=neighbourhood_size, dim=-1, sorted=False)
        top_k = torch.stack([F.softmax(top_k[:, k], dim=1) for k in range(K)]).transpose(0, 1)  # (batch_size, K, neighbourhood_size)

        # extract top k features and pseudo coordinates
        neighbourhood_image = \
            self._create_neighbourhood_feat(features, top_ind)
        neighbourhood_pseudo = \
            self._create_neighbourhood_pseudo(pseudo_coord, top_ind)

        # weight neighbourhood features with graph edge weights
        if weight:
            neighbourhood_image = top_k.unsqueeze(-1)*neighbourhood_image

        return neighbourhood_image, neighbourhood_pseudo

    def _create_neighbourhood_feat(self, image, top_ind):
        '''
        ## Inputs:
        - image (batch_size, K, feat_dim)
        - top_ind (batch_size, K, neighbourhood_size)
        ## Returns:
        - neighbourhood_image (batch_size, K, neighbourhood_size, feat_dim)
        '''

        batch_size = image.size(0)
        K = image.size(1)
        feat_dim = image.size(2)
        neighbourhood_size = top_ind.size(-1)
        image = image.unsqueeze(1).expand(batch_size, K, K, feat_dim)
        idx = top_ind.unsqueeze(-1).expand(batch_size,
                                           K, neighbourhood_size, feat_dim)
        return torch.gather(image, dim=2, index=idx)

    def _create_neighbourhood_pseudo(self, pseudo, top_ind):
        '''
        ## Inputs:
        - pseudo_coord (batch_size, K, K, coord_dim)
        - top_ind (batch_size, K, neighbourhood_size)
        ## Returns:
        - neighbourhood_pseudo (batch_size, K, neighbourhood_size, coord_dim)
        '''
        batch_size = pseudo.size(0)
        K = pseudo.size(1)
        coord_dim = pseudo.size(3)
        neighbourhood_size = top_ind.size(-1)
        idx = top_ind.unsqueeze(-1).expand(batch_size,
                                           K, neighbourhood_size, coord_dim)
        return torch.gather(pseudo, dim=2, index=idx)


class GraphLearner(nn.Module):
    def __init__(self, in_feature_dim, combined_feature_dim, dropout=0.5):
        super(GraphLearner, self).__init__()

        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Parameters
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim

        # Embedding layers
        self.edge_layer_1 = nn.Linear(in_feature_dim,
                                      combined_feature_dim)
        self.edge_layer_2 = nn.Linear(combined_feature_dim,
                                      combined_feature_dim)

        # Regularisation
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        bs = len(graph_nodes)

        graph_nodes = graph_nodes.view(-1, self.in_dim)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = nn.functional.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = nn.functional.relu(h)

        # outer product
        h = h.view(bs, -1, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        return adjacency_matrix


class NeighbourhoodGraphConvolution(Module):
    '''
    Implementation of: https://arxiv.org/pdf/1611.08402.pdf where we consider
    a fixed sized neighbourhood of nodes for each feature
    '''

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 coordinate_dim,
                 bias=False):
        super(NeighbourhoodGraphConvolution, self).__init__()
        '''
        ## Variables:
        - in_feat_dim: dimensionality of input features
        - out_feat_dim: dimensionality of output features
        - n_kernels: number of Gaussian kernels to use
        - coordinate_dim : dimensionality of the pseudo coordinates
        - bias: whether to add a bias to convolutional kernels
        '''

        # Set parameters
        self.n_kernels = n_kernels
        self.coordinate_dim = coordinate_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim//n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.mean_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.mean_theta = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_theta = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def forward(self, neighbourhood_features, neighbourhood_pseudo_coord):
        '''
        ## Inputs:
        - neighbourhood_features (batch_size, K, neighbourhood_size, in_feat_dim)
        - neighbourhood_pseudo_coord (batch_size, K, neighbourhood_size, coordinate_dim)
        ## Returns:
        - convolved_features (batch_size, K, neighbourhood_size, out_feat_dim)
        '''

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # compute pseudo coordinate kernel weights
        weights = self.get_gaussian_weights(neighbourhood_pseudo_coord)
        weights = weights.view(
            batch_size*K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size*K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def get_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:
        - pseudo_coord (batch_size, K, K, pseudo_coord_dim)
        ## Returns:
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        '''

        # compute rho weights
        diff = (pseudo_coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1, -1))**2
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1)**2))

        # compute theta weights
        first_angle = torch.abs(pseudo_coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2)
                                  / (1e-14 + self.precision_theta.view(1, -1)**2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True)+1e-10)

        return weights

    def convolution(self, neighbourhood, weights):
        '''
        ## Inputs:
        - neighbourhood (batch_size*K, neighbourhood_size, in_feat_dim)
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        ## Returns:
        - convolved_features (batch_size*K, out_feat_dim)
        '''
        # patch operator
        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat([i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features
