import torch
import numpy as np


def rescoring(detections, annotations, matching, target):
    """ Generates the target tensor, given detections and annotations in an image.
    
    For each bbox, the input tensor is the concatenation of the bbox 
    confidence score, the predicted label (one-hot encoded) and the bbox
    coordinates
    
    Args:
        detections: list of detections
        annotations: list of annotations
        matching: matching algorithm, either "greedy", "gt" or "coco"
        target: target confidence score, either "iou", "binary" or "confidence"

    Returns:
        A tensor of shape (n_detections, 1)      
    """
    assert matching in ["greedy", "gt", "coco"], (
        "Matching algorithm {} not valid. Use 'greedy', 'gt' or 'threshold'".format(matching)
    )
    assert target in ["binary", "confidence", "iou"], (
        "Target {} not valid. Use 'binary', 'confidence' or 'iou'".format(target)
    )

    # make sure detections are sorted by descending confidence score
    if not all(detections[i]["score"] >= detections[i + 1]["score"] for i in range(len(detections) - 1)):
        dtind = np.argsort([-d["score"] for d in detections], kind='mergesort')
        detections = [detections[i] for i in dtind]

    # TODO: review matching algorithm names
    if matching == "greedy":
        return greedy_matching(detections, annotations, target)
    elif matching == "gt":
        return gt_overlap_matching(detections, annotations, target)
    elif matching == "coco":
        return coco_matching(detections, annotations, target)


def coco_matching(dets, gts, target, threshold=0.5):
    """ Implements COCO matching algorithm at a fixed IoU threshold
    
    Detections are matched with the ground-truth from the same class with higher IoU
    and IoU >= threshold. Higher confidence detections have priority. 
    """

    D, G = len(dets), len(gts)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covered = np.zeros(G, dtype=bool)
    targets = torch.zeros(D, device=device)

    # if no ground-truths, no matching needed
    if G == 0:
        return targets

    for i, det in enumerate(dets):
        # select valid gts: same class, not covered, IoU >= threshold, not crowd
        gts_class = [gt for ind, gt in enumerate(gts) if gt["category_id"] == det["category_id"] and not covered[ind] and IoU(det, gt) >= threshold and not gt["iscrowd"]]
        # if no ground-truth match, nothing to do
        if len(gts_class) == 0:
            continue
        index = [ind for ind, gt in enumerate(gts) if gt["category_id"] == det["category_id"] and not covered[ind] and IoU(det, gt) >= threshold and not gt["iscrowd"]]
        
        class_ious = IoU_matrix([det], gts_class)
        matched_gt_class = class_ious.argmax()  # match with gt with highest IoU
        matched_gt = index[matched_gt_class]
        covered[matched_gt] = True
        if target == "binary":
            targets[i] = 1
        elif target == "confidence":
            targets[i] = det["score"]
        elif target == "iou":
            targets[i] = class_ious[matched_gt_class][0]

    return targets.reshape(-1, 1)


def greedy_matching(dets, gts, target):
    """ Computes matching at several thresholds
    
    Detections are matched at descending threshold levels.
    Prioritizes detections with higher confidence.
    """
    # load detections and annotations
    D, G = len(dets), len(gts)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covered = np.zeros(G, dtype=bool)  # keeps track of which gt's have been matched
    det_covered = np.zeros(D, dtype=bool)
    targets = torch.zeros(D, dtype=torch.float, device=device)

    # If there are no Ground-truths
    if G == 0:
        return targets

    for iou_threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        for i, det in enumerate(dets):
            if det_covered[i]:
                continue
            # select ground-truths with same category_id not yet matched and IoU >= threshold
            gts_class, index = [], []
            for ind, gt in enumerate(gts):
                if gt["category_id"] == det["category_id"] and not covered[ind] and IoU(det, gt) >= iou_threshold:
                    gts_class.append(gt)
                    index.append(ind)
            if len(gts_class) == 0:
                continue

            # find closest match from same class
            class_ious = IoU_matrix([det], gts_class)
            matched_gt_class = class_ious.argmax()
            matched_gt = index[matched_gt_class]
            # keep detection if not already matched and iou above threshold
            det_covered[i] = True
            covered[matched_gt] = True
            if target == "binary":
                targets[i] = 1
            elif target == "confidence":
                targets[i] = det["score"]
            else:
                targets[i] = class_ious[matched_gt_class][0]

    return targets.reshape(-1, 1)


def gt_overlap_matching(dets, gts, target):
    """ Matches each ground-truth to the detection with highest overlap.
        Prioritizes localization over confidence
        TODO: explore if ignoring/postponing ground-truths with 'iscrowd' improves performance
    """
    D, G = len(dets), len(gts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covered = np.zeros(D, dtype=bool)  # keeps track of which dets have been matched
    gt_covered = np.zeros(G, dtype=bool)  # keeps track of which dets have been matched
    targets = torch.zeros(D, device=device)

    # If there are no Ground-truths
    if G == 0:
        return targets

    # Iterate over the ground_truths
    for iou_threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        for i, gt in enumerate(gts):
            if gt_covered[i]:
                continue
            # select detections with same category_id, not covered, and IoU >= threshold
            dets_class, index = [], []
            for ind, det in enumerate(dets):
                if det["category_id"] == gt["category_id"] and not covered[ind] and IoU(det, gt) >= iou_threshold:
                    dets_class.append(det)
                    index.append(ind)
            if len(dets_class) == 0:
                continue

            # find closest match from same class
            class_ious = IoU_matrix([gt], dets_class)
            matched_det_class = class_ious.argmax()
            matched_det = index[matched_det_class]
            # keep detection
            covered[matched_det] = True
            gt_covered[i] = True
            if target == "binary":
                targets[matched_det] = 1
            elif target == "confidence":
                targets[matched_det] = dets[matched_det]["score"]
            else:
                targets[matched_det] = class_ious[matched_det_class][0]

    return targets


def intersect(x1, y1, w1, h1, x2, y2, w2, h2):
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    if right > left and bottom > top:
        return (right - left) * (bottom - top)
    return 0


def IoU(det, gt):
    x1, y1, w1, h1 = det["bbox"]
    x2, y2, w2, h2 = gt["bbox"]
    intersection = intersect(x1, y1, w1, h1, x2, y2, w2, h2)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def IoU_matrix(dets, gts):
    D, G = len(dets), len(gts)
    ious = np.zeros((G, D))
    for g in range(G):
        for d in range(D):
            ious[g, d] = IoU(dets[d], gts[g])
    return ious
