from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

def coco_eval(result_file="temp/results.json", anns="data/coco/annotations/instances_val2017.json"):
    coco = COCO(anns)
    coco_dets = coco.loadRes(result_file)
    img_ids = coco.getImgIds()
    cocoEval = COCOeval(coco, coco_dets, "bbox")
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats

parser = argparse.ArgumentParser()
parser.add_argument("result_file")
parser.add_argument("anns")

args = parser.parse_args()
stats = coco_eval(args.result_file, args.anns)

print(stats)
