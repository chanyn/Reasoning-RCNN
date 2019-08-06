import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

import numpy as np
import pickle

@DETECTORS.register_module
class HKRMRCNN(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 adja_gt=None,
                 adjr_gt=None):
        super(HKRMRCNN, self).__init__()
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
            self.bbox_head_hkrm = builder.build_bbox_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        # read adj gts from .pkl
        self.adja_gt = None
        self.adjr_gt = None
        if adja_gt is not None:
            self.adja_gt = pickle.load(open(adja_gt, 'rb'))
            self.adja_gt = np.float32(self.adja_gt)
            self.adja_gt = nn.Parameter(torch.from_numpy(self.adja_gt), requires_grad=False)
        if adjr_gt is not None:
            self.adjr_gt = pickle.load(open(adjr_gt, 'rb'))
            self.adjr_gt = np.float32(self.adjr_gt)
            self.adjr_gt = nn.Parameter(torch.from_numpy(self.adjr_gt), requires_grad=False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(HKRMRCNN, self).init_weights(pretrained)
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
            self.bbox_head_hkrm.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()

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
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.train_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results = []
            # gt adj list
            gt_adja_list = []
            gt_adjr_list = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    has_roi_score=True,
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                # get adj matrix gt
                index_ = torch.cat((sampling_result.pos_gt_labels,
                                    sampling_result.pos_gt_labels.new_zeros((len(sampling_result.neg_bboxes)))))
                assert len(index_) == len(sampling_result.bboxes)
                if self.adja_gt is not None:
                    pos_gt = self.adja_gt[index_, :]
                    pos_gt = pos_gt.transpose(0, 1)[index_, :]
                    pos_gt = pos_gt.transpose(0, 1)
                    gt_adja_list.append(pos_gt)
                if self.adjr_gt is not None:
                    pos_gt = self.adjr_gt[index_, :]
                    pos_gt = pos_gt.transpose(0, 1)[index_, :]
                    pos_gt = pos_gt.transpose(0, 1)
                    gt_adjr_list.append(pos_gt)
            A_gt = []
            if self.adja_gt is not None:
                A_gt.append(torch.stack(gt_adja_list, 0))
            if self.adjr_gt is not None:
                A_gt.append(torch.stack(gt_adjr_list, 0))


        # bbox head forward and loss
        if self.with_bbox:
            rois, rois_index = bbox2roi(
                [(res.pos_bboxes, res.neg_bboxes) for res in sampling_results],
                return_index=True)
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            # Get grometric feature of rois
            geom_f = []
            for img_i, img_shape in enumerate(img_meta):
                h, w, _ = img_meta[img_i]['pad_shape']
                tmp_geo = rois[rois[:, 0] == img_i, 1:] / torch.Tensor([h, w, h, w]).cuda()
                tmp_geo = torch.cat((tmp_geo, sampling_results[img_i].bboxes[:, 4].unsqueeze(1)), dim=-1)
                geom_f.append(tmp_geo)
            geom_f = torch.stack(geom_f, 0)
            # bbox_feats = bbox_feats.view(len(img_meta), -1, bbox_feats.size(-1))
            assert len(geom_f.size()) == 3
            cls_score, bbox_pred, A_pred = self.bbox_head_hkrm(bbox_feats, geom_f, len(img_meta))

            bbox_targets = self.bbox_head_hkrm.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head_hkrm.loss(cls_score, bbox_pred,
                                            A_pred, A_gt,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if self.with_mask_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)

            else:
                pos_inds = (rois_index == 0)
                mask_feats = bbox_feats[pos_inds]

            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes_hkrm(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale, use_hkrm=True)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head_hkrm.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes_hkrm(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn, use_hkrm=True)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head_hkrm.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def simple_test_bboxes_hkrm(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           use_hkrm=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_upper_neck:
            roi_feats = self.upper_neck(roi_feats)
        if use_hkrm:
            # Get grometric feature of rois
            geom_f = []
            for img_i, img_shape in enumerate(img_meta):
                h, w, _ = img_meta[img_i]['pad_shape']
                tmp_geo = rois[rois[:, 0] == img_i, 1:] / torch.Tensor([h, w, h, w]).cuda()
                tmp_geo = torch.cat((tmp_geo, proposals[img_i][:, 4].unsqueeze(1)), dim=-1)
                geom_f.append(tmp_geo)
            geom_f = torch.stack(geom_f, 0)
            assert len(geom_f.size()) == 3
            cls_score, bbox_pred, A_pred = self.bbox_head_hkrm(roi_feats, geom_f, len(img_meta))
        else:
            cls_score, bbox_pred = self.bbox_head_hkrm(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head_hkrm.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes_hkrm(self, feats, img_metas, proposal_list, rcnn_test_cfg, use_hkrm=False):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_upper_neck:
                roi_feats = self.upper_neck(roi_feats)

            if use_hkrm:
                # Get grometric feature of rois
                geom_f = []
                h, w, _ = img_meta['pad_shape']
                tmp_geo = rois[:, 1:] / torch.Tensor([h, w, h, w]).cuda()
                tmp_geo = torch.cat((tmp_geo, proposals[:, 4].unsqueeze(1)), dim=-1)
                geom_f.append(tmp_geo)
                geom_f = torch.stack(geom_f, 0)
                assert len(geom_f.size()) == 3
                cls_score, bbox_pred, A_pred = self.bbox_head_hkrm(roi_feats, geom_f, len(img_meta))
            else:
                cls_score, bbox_pred = self.bbox_head_hkrm(roi_feats)
            bboxes, scores = self.bbox_head_hkrm.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels
