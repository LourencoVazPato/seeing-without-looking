# Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization

**[[arXiv](https://arxiv.org/abs/1912.12290)]**

<p align="center">
  <img width="200" src="docs/265108_predicted.jpg">
  <img width="200" src="docs/265108_rescored.jpg">
</p>

## Introduction

Seeing without Looking is an approach that aims to improve Average Precision by rescoring the object detections with the use of contextual information inferred from other objects in the same image.
The model takes in a set of already made detections and predicts a new score for each object. Because our method does not use any visual information, inference is **fast**.

<p align="center">
  <img height="200" src="docs/ApproachOverview-1.png">
</p>

### Results

| Detector*               | `val2017`              | improvement | `test-dev2017`         | inf time (fps)** |   Download   |
| :---------------------: | :--------------------: | :---------: | :--------------------: | :--------------: | :----------: |
| RetinaNet R-50-FPN      | 35.6 &rightarrow; 36.6 |  + 1.0 AP   | 35.9 &rightarrow; 36.7 |  57.2            | model/config |
| RetinaNet R-101-FPN     | 38.1 &rightarrow; 38.7 |  + 0.6 AP   | 38.7 &rightarrow; 39.2 |  62.8            | model/config |
| Faster R-CNN R-50-FPN   | 36.4 &rightarrow; 37.5 |  + 1.1 AP   | 36.7 &rightarrow;      |  83.0            | model/config |
| Faster R-CNN R-101-FPN  | 39.4 &rightarrow; 39.9 |  + 0.5 AP   | 39.7 &rightarrow; 40.1 |  90.7            | model/config |
| Cascade R-CNN R-50-FPN  | 41.1 &rightarrow; 41.8 |  + 0.7 AP   | 41.5 &rightarrow; 41.9 |  97.2            | model/config |
| Cascade R-CNN R-101-FPN | 42.1 &rightarrow; 42.8 |  + 0.7 AP   | 42.4 &rightarrow; 42.6 |  96.9            | model/config |

*baseline detections were generated using Open MMLab [MMDetection](https://github.com/open-mmlab/mmdetection/) implementations from [MODEL_ZOO](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md)

**inference time while processing one image at a time. If batching is used, inference can be much faster

## Installation and requirements

## Example usage

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

### 4. Evaluating model

### 5. Performing inference on `val2017`

Once you have a trained model you can perform inference without preprocessing by running:
```
python inference.py <config> <model> <path_dets> <path_anns>
```
Where `<config>` is the model config file, `<model>` is the model's weights file in `.pt` format, `<path_dets>` is the detections JSON file and `<path_anns>` is the annotations file.  

## Citation
```
@article{pato2019seeing,
  title={Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization},
  author={Pato, Lourenço V. and Negrinho, Renato and Aguiar, Pedro M. Q.},
  journal={arXiv preprint arXiv:1912.12290},
  year={2019}
}
```
