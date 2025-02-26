"""
Non-Maximum Suppression for video proposals.
"""

import math

def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):

    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union
    

def temporal_nms(predictions, nms_thd, sigma=0.5, max_after_nms=100):
    """
    Args:
        predictions: list(sublist), each sublist is [st (float), ed(float), score (float)]
        nms_thd: float in [0, 1], threshold for IOU to decide suppression.
        sigma: float, parameter for the Gaussian decay function
        max_after_nms: int, maximum number of predictions to keep after NMS
    Returns:
        predictions_after_nms: list(sublist), each sublist is [st (float), ed(float), score (float)]
    """
    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # descending order
    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]

    # Lists to store suppressed predictions
    rstart, rend, rscore = [], [], []

    while len(tscore) > 0 and len(rscore) < max_after_nms:
        max_score_idx = tscore.index(max(tscore))
        for idx in range(len(tscore)):
            if idx == max_score_idx:
                continue
            iou = compute_temporal_iou([tstart[max_score_idx], tend[max_score_idx]], [tstart[idx], tend[idx]])
            if iou > nms_thd:
                # Decay score using a Gaussian function
                tscore[idx] *= math.exp(-(iou * iou) / sigma)

        rstart.append(tstart.pop(max_score_idx))
        rend.append(tend.pop(max_score_idx))
        rscore.append(tscore.pop(max_score_idx))

    predictions_after_nms = [[st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms
