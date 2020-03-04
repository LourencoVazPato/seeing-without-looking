import torch
import json
import time
import argparse
from dataset import Dataset, PadCollate
from model import ContextualRescorer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from visualize import printBold
from helper import Helper


def coco_eval(result_file="temp/results.json", anns="data/annotations/instances_val2017.json"):
    coco = COCO(anns)
    coco_dets = coco.loadRes(result_file)
    iou_type = "bbox"
    img_ids = coco.getImgIds()
    cocoEval = COCOeval(coco, coco_dets, iou_type)
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats


def write_validation_results(dataset, model, helper, outfile="temp/results.json"):
    """ Rescore validation detections and write them to file """

    batch_size = 1024  # can increase size if enough GPU space to allow faster evaluation 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate())
    images = helper.images
    categories = helper.categories

    js_out = []
    start = time.time()
    for i, (input_tensor, target_tensor, lengths) in enumerate(dataloader):

        mask = (target_tensor != -1).float()
        prediction = model.forward(input_tensor, lengths, mask)
        for batch in range(input_tensor.size(0)):
            img_id = dataset.get_id(i * batch_size + batch)
            H, W = images[img_id]["height"], images[img_id]["width"]
            seq_len = (target_tensor[batch] != -1).sum()
            for j in range(seq_len):
                pred_score = round(input_tensor[batch, j, 0].item(), 4)
                x, y, w, h = input_tensor[batch, j, 81:85].tolist()
                x = round(x * W, 2)
                y = round(y * H, 2)
                w = round(w * W, 2)
                h = round(h * H, 2)
                bbox = [x, y, w, h]
                _, category = input_tensor[batch, j, 1:81].max(0)
                category = category.item()
                category = categories[helper.category_index[category]]["id"]
                rescore = round(prediction[batch, j].item(), 4)
                js = {
                    "image_id": img_id,
                    "category_id": category,
                    "bbox": bbox,
                    "score": rescore,
                }
                js_out.append(js)

    print("Generated evaluation results (t={:.2f}s). Writing to {}".format(time.time() - start, outfile))
    with open(outfile, "w") as f:
        json.dump(js_out, f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate model on the COCO validation set using pycocotools.COCOeval."
    )
    parser.add_argument("cfg", help="model parameters file")
    parser.add_argument("dataset", help="preprocessed detection results from base model")

    args = parser.parse_args()
    return args


def main():
    # Parse arguments: model folder and ablation studies
    args = parse_arguments()
    
    with open(args.cfg) as f:
        params = json.load(f)
    cfg = args.cfg.split('/')[1].split('.')[0]  # TODO: review file locations in final version

    args.test = "test" in args.dataset

    # Test-dev instances
    if args.test:
        instances = "data/annotations/instances_test-dev2017.json"
        outfile = "temp/detections_test-dev2017_rescored_" + cfg + "_results.json"

    # validation set annotations
    else:
        instances = "data/annotations/instances_val2017.json"
        outfile = "temp/val_results.json"
    
    helper = Helper(instances)

    # Load validation dataset
    dataset = Dataset(args.dataset)

    # Load model
    device = torch.device('cuda:0')
    model = ContextualRescorer(params).to(device)
    state_dict = torch.load("logs/" + cfg + "/model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    # Reclassify the detections in the validation set
    write_validation_results(
        dataset,
        model,
        helper,
        outfile=outfile,
    )

    # Evaluate the results on COCO metrics
    if not args.test:
        eval_stats = coco_eval(outfile)
        printBold("\n\tAP score: " + str(round(eval_stats[0], 6)) + "\n")
        print(eval_stats)


if __name__ == "__main__":
    main()
