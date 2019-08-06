import numpy as np
import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


class RandomSamplerFixnum(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(RandomSamplerFixnum, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    # def _sample_pos(self, assign_result, num_expected, **kwargs):
    #     """Randomly sample some positive samples."""
    #     pos_inds = torch.nonzero(assign_result.gt_inds > 0)
    #     if pos_inds.numel() != 0:
    #         pos_inds = pos_inds.squeeze(1)
    #     if pos_inds.numel() <= num_expected:
    #         return pos_inds
    #     else:
    #         return self.random_choice(pos_inds, num_expected)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Balance sampling for positive bboxes/anchors.

        1. calculate average positive num for each gt: num_per_gt
        2. sample at most num_per_gt positives for each gt
        3. random sampling from rest anchors if not enough fg
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            repeat_ = num_expected // pos_inds.numel()
            return torch.cat((pos_inds.repeat(repeat_), self.random_choice(pos_inds, num_expected % pos_inds.numel())))
        else:
            return self.random_choice(pos_inds, num_expected)

    # def _sample_neg(self, assign_result, num_expected, **kwargs):
    #     """Randomly sample some negative samples."""
    #     neg_inds = torch.nonzero(assign_result.gt_inds == 0)
    #     if neg_inds.numel() != 0:
    #         neg_inds = neg_inds.squeeze(1)
    #     if len(neg_inds) <= num_expected:
    #         return neg_inds
    #     else:
    #         return self.random_choice(neg_inds, num_expected)
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Balance sampling for negative bboxes/anchors.

        Negative samples are split into 2 set: hard (balance_thr <= iou <
        neg_iou_thr) and easy(iou < balance_thr). The sampling ratio is controlled
        by `hard_fraction`.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            repeat_ = num_expected // neg_inds.numel()
            return torch.cat((neg_inds.repeat(repeat_), self.random_choice(neg_inds, num_expected % neg_inds.numel())))
        else:
            return self.random_choice(neg_inds, num_expected)



    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               has_roi_score=False,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        if has_roi_score:
            gt_bboxes_new = gt_bboxes.new_ones((gt_bboxes.shape[0], 5))
            gt_bboxes_new[:, :4] = gt_bboxes
            gt_bboxes = gt_bboxes_new
        else:
            bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        # sample pos inds must be fixed
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        # pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        # neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)


