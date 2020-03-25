# Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization

**[[arXiv](https://arxiv.org/abs/1912.12290)]**

<p align="center">
  <img width="200" src="docs/265108_predicted.jpg">
  <img width="200" src="docs/265108_rescored.jpg">
</p>

## Introduction

Seeing without Looking is an approach that aims to improve Average Precision by rescoring the object detections with the use of contextual information inferred from other objects in the same image.
The model takes in a set of already made detections and predicts a new score for each object. Because our approach does not use any visual information, contextual rescoring is **widely applicable** and **inference is fast**.

<p align="center">
  <img height="200" src="docs/ApproachOverview-1.png">
</p>

### Results

| Detector*               | `val2017` AP                      | `test-dev2017` AP                 | inf time (fps)** |   Download   |
| :---------------------: | :-------------------------------: | :-------------------------------: | :--------------: | :----------: |
| RetinaNet R-50-FPN      | 35.6 &rightarrow; 36.6 **(+1.0)** | 35.9 &rightarrow; 36.8 **(+0.9)** |  57.2            | model/config |
| RetinaNet R-101-FPN     | 38.1 &rightarrow; 38.7 **(+0.6)** | 38.7 &rightarrow; 39.2 **(+0.5)** |  62.8            | model/config |
| Faster R-CNN R-50-FPN   | 36.4 &rightarrow; 37.5 **(+1.1)** | 36.7 &rightarrow; 37.5 **(+0.8)** |  83.0            | model/config |
| Faster R-CNN R-101-FPN  | 39.4 &rightarrow; 39.9 **(+0.5)** | 39.7 &rightarrow; 40.1 **(+0.4)** |  90.7            | model/config |
| Cascade R-CNN R-50-FPN  | 41.1 &rightarrow; 41.8 **(+0.7)** | 41.5 &rightarrow; 42.0 **(+0.5)** |  97.2            | model/config |
| Cascade R-CNN R-101-FPN | 42.1 &rightarrow; 42.8 **(+0.7)** | 42.4 &rightarrow; 42.8 **(+0.4)** |  96.9            | model/config/dets |

*baseline detections were generated using Open MMLab [MMDetection](https://github.com/open-mmlab/mmdetection/) implementations from [MODEL_ZOO](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md)

**inference time while processing one image at a time. If batching is used, inference should be much faster (up to 300 fps)

## Installation and requirements

**Download** annotation files


## Instructions

### 1. Generating detections

Input detections must be generated using any detection architecture of your choosing and saved in a JSON file with the official [COCO results format](http://cocodataset.org/#format-results). 
<!-- ```
[{
    "image_id"      : int, 
    "category_id"   : int, 
    "bbox"          : [x,y,width,height], 
    "score"         : float,
}]
``` -->
The generated detections must be saved under `data/detections/detections_<dataset>_<architecture>.json`, where `<architecture>` is the name of the model you have used for generating the detections and `<dataset>` refers to either `train2017`, `val2017` or `test-dev2017` splits of the COCO dataset.
You can measure the baseline AP on `val2017` by running:
```
python tools/coco_eval.py data/detections_<dataset>_<architecture>.json data/annotations/instances_val2017.json
```

### 2. Preprocessing detections

The detections must be preprocessed and saved into disk to save time during training. For that, run:
```
python preprocessing.py <dataset> <architecture>
```
The preprocessed tensors should be saved to `data/preprocessed/preprocessed_<dataset>_<architecture>.pt`.

### 3. Training the model

Once the detections have been preprocessed, you can train a model by running:
```
python train.py <config_file> <architecture>
```
Once training is completed, the training logs, model parameters and model config should be saved to `logs/<config_file>/`.
Note that the logs will override the contents of `logs/<config_file>/`. A config file must be created under a different name for different architectures.

### 4. Evaluating model on `val2017` (with preprocessed detections)
```
python evaluate.py <config_file> <path_preprocessed>
```

If the preprocessed detections belong to val2017, the rescored results will be saved in `temp/val_results.json` and the validation AP is computed.

If the preprocessed detections belong to test-dev2017 set, the rescored results will be saved in `temp/detections_test-dev2017_<config>_rescored_results.json` and can be zipped and submitted for evaluation on [[CodaLab](https://competitions.codalab.org/competitions/20794#participate)].

### 5. Performing inference without preprocessed detections

Once you have a trained model you can perform inference on the detections without preprocessing by running:
```
python inference.py <config> <model> <path_dets> <path_anns>
```
Where `<config>` is the model config file, `<model>` is the model's weights file in `.pt` format, `<path_dets>` is the detections JSON file and `<path_anns>` is the annotations file (`data/annotations/instances_test-dev2017.json`). 

We recommend using method 4 (with preprocessed detections) for generating evaluation results as it produces better scores.

## Example usage

### 0. Download trained model and preprocessed detections

### 1. Train model

### 2. Evaluate model on `val2017`

## Citation
```
@article{pato2019seeing,
  title={Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization},
  author={Pato, Lourenço V. and Negrinho, Renato and Aguiar, Pedro M. Q.},
  journal={arXiv preprint arXiv:1912.12290},
  year={2019}
}
```
