import torch
import time
import os
import argparse
import sys
from rescoring import rescoring
from helper import Helper
from mmcv import ProgressBar


coco_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a set of detections in JSON format into a dict of input/target tensor pairs")
    parser.add_argument("dataset", help="which dataset to preprocess: val2017, train2017 or test-dev2017", type=str)
    parser.add_argument("architecture", help="architecture of the baseline model", type=str)
    args = parser.parse_args()
    return args


def to_one_hot(category_id, num_labels=80):
    """ Returns a one-hot encoded list for category_id (1-90) out of 80 classes

    First convert the category_id into the corresponding index (0-79), then create vector

    Args: 
        cat_id: category_id of the class
        num_labels: number of categories (default: 8)
    Returns
        list
    """
    index = coco_categories.index(category_id)
    return [0 if i != index else 1 for i in range(num_labels)]


def get_tensors(img_id, helper, device, method='gt', target_confidence='iou'):
    input_ = input_tensor(img_id, helper, device)    
    target = rescoring(
        helper.get_detections(img_id),
        helper.get_annotations(img_id),
        method,
        target_confidence,
    )
    global no_tps, w_tps
    global tps, fps
    if (target.sum() == 0):
        no_tps += 1
    else:
        w_tps += 1
    tps += (target != 0).sum()
    fps += (target == 0).sum()
    return input_, target


def get_tensors_no_target(img_id, helper, device):
    input_ = input_tensor(img_id, helper, device)
    if isinstance(input_, torch.Tensor):
        target = torch.zeros((input_.size(0), 1), device=device)
    else:
        target = None
    return input_, target


def input_tensor(img_id, helper, device, dtype=torch.float):
    """ Generates the input tensor for all detections in an image.
    
    For each bbox, the input tensor is the concatenation of the bbox 
    confidence score, the predicted label (one-hot encoded) and the bbox
    coordinates
    
    Args:
        img_id: id of the image.
        helper:
        device:
    
    Returns:
        A tensor of shape (n_detections, 85)      
    """
    dets = helper.get_detections(img_id)
    if len(dets) == 0:
        return
        
    # sort detections by confidence score
    dets.sort(key=lambda det: det["score"], reverse=True)

    W, H = helper.images[img_id]["width"], helper.images[img_id]["height"]
    inputs = []
    for det in dets:
        score = torch.tensor([det["score"]], dtype=dtype, device=device)
        label = torch.tensor(to_one_hot(det["category_id"]), dtype=dtype, device=device)
        x, y, w, h = det["bbox"]
        bbox = [x / W, y / H, w / W, h / H]  # TODO: experiment different bbox parameterizations
        bbox = torch.tensor(bbox, dtype=dtype, device=device)
        input_ = torch.cat((score, label, bbox))
        inputs.append(input_)

    # stack inputs
    inputs = torch.stack(inputs)

    return inputs


if __name__ == "__main__":
    args = parse_args()
    data_root = "data/"
    path_dets = data_root + "detections/detections_" + args.dataset + "_" + args.architecture + ".json"
    path_anns = data_root + "annotations/instances_" + args.dataset + ".json"
    outfile = data_root + "preprocessed/preprocessed_" + args.dataset + "_" + args.architecture + ".pt"

    assert os.path.exists(path_dets), "Detections path {} does not exist".format(path_dets)
    assert os.path.exists(path_anns), "Annotations path {} does not exist".format(path_anns)
    if os.path.exists(outfile):
        print("Outfile {} already exists. Overwrite? (y/n): ".format(outfile))
        reply = str(input()).lower().strip()
        if reply[0] == 'y':
            pass
        else:
            print("Aborting.")
            sys.exit(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    helper = Helper(path_anns, path_dets=path_dets)

    matching = "gt"
    target_confidence = "iou"
    no_tps, w_tps = 0, 0
    tps, fps = 0, 0
    inputs, targets = {}, {}

    prog_bar = ProgressBar(len(helper.detections))
    for i, img_id in enumerate(helper.detections):
        if "test" in args.dataset:
            input_, target = get_tensors_no_target(img_id, helper, device)
        else:
            input_, target = get_tensors(img_id, helper, device, matching, target_confidence)

        # only add if there are any detections
        if isinstance(input_, torch.Tensor):
            inputs[img_id] = input_
            targets[img_id] = target

        prog_bar.update()

    print("\nPreprocessing done. Stats:")
    im = len(helper.images)
    dt = len(helper.detections)
    print("Images with no detections: {} out of {}. Percentage: {}%".format(im - dt, im, (im - dt) / im * 100))
    if "test" not in args.dataset:
        an = len(helper.annotations)
        print("Images with no annotations: {} out of {}. Percentage: {}%".format(im - an, im, (im - an) / im * 100))
        print("Images with no true positives: {} out of {}. Percentage: {}%".format(no_tps, im, no_tps / im * 100))
        print("Images with true positives: {} out of {}. Percentage: {}%".format(w_tps, im, w_tps / im * 100))
        print("True positives: {}; False positives: {};  Percentage of TPs: {}%".format(tps, fps, round(float(tps) / float(fps + tps) * 100, 2)))

    print("Saving tensors...")
    torch.save((inputs, targets), outfile)
    print("Done. Saved tensors to {}".format(outfile))
