import torch
import argparse
import os
import json

from helper import Helper
from model import ContextualRescorer
from evaluate import write_validation_results
from preprocessing import input_tensor, get_tensors_no_target
from mmcv import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference (rescore) a set of detections in JSON format, given a trained model.")
    parser.add_argument("config", help="model config file path", type=str)
    parser.add_argument("model", help="model state dict path", type=str)
    parser.add_argument("path_dets", help="detections path", type=str)
    parser.add_argument("path_anns", help="file path containing image info", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config), "Config file {} does not exist".format(args.config)
    assert os.path.exists(args.model), "Model path {} does not exist".format(args.model)
    assert os.path.exists(args.path_dets), "Detections path {} does not exist".format(args.path_dets)
    assert os.path.exists(args.path_anns), "Annotations file {} does not exist".format(args.path_anns)

    assert args.config.endswith('.json'), "Config file {} must be in json format".format(args.config)
    assert args.model.endswith('.pt'), "Model file {} must be in .pt format".format(args.model)
    assert args.path_dets.endswith('.json'), "Detections file {} must be in json format".format(args.path_dets)
    assert args.path_anns.endswith('.json'), "Annotations file {} must be in json format".format(args.path_anns)

    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and config file
    with open(args.config) as fh:
        params = json.load(fh)
    model = ContextualRescorer(params).to(device)
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocess input detections
    helper = Helper(args.path_anns, path_dets=args.path_dets)
    prog_bar = ProgressBar(len(helper.detections))
    rescored = []

    for id_, dets in helper.detections.items():
        ipt = input_tensor(id_, helper, device).unsqueeze(0)
        dets.sort(key=lambda det: det["score"], reverse=True)
        mask = torch.ones(1, 1, ipt.size(1), device=device, dtype=torch.float)
        
        # add paddings
        if ipt.size(1) < 100:
            pad = torch.zeros((1, 1, 100-ipt.size(1)), device=device)
            mask = torch.cat((mask, pad), dim=2)
            pad = torch.zeros((1, 100-ipt.size(1), 85), device=device)
            ipt = torch.cat((ipt, pad), dim=1)
        scores = model.forward(ipt, [ipt.size(1)], mask).reshape(-1)
        for i, det in enumerate(dets):
            det['score'] = round(scores[i].item(), 3)
            rescored.append(det)
        prog_bar.update()

    # Rescore detections and write to file
    outfile = args.path_dets.replace('.json', '_rescored_results.json')
    print("\nWriting rescored detections to {}".format(outfile))
    with open(outfile, 'w') as fh:
        json.dump(rescored, fh)
