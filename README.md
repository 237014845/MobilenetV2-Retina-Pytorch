
# Lightweight Object Detection(One-Stage)

## Introduction

The code is based on the mmdetection.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

Currently, it contains these features:
- **Multiple Base Network**: Mobilenet V2, ShuffleNet V2
- **One-Stage Lightweight Detector**: MobileV2-SSD, MobileV2-RetinaNet


## Performance

| VOC2007      | SSD                                                                         | RetinaNet                                                                   
|--------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| MobilenetV2  |                                                                             | 81.9%                                                                        |
| ShufflenetV2 |                                                                             |                                                                              |



| SAR(SSDD)    | SSD                                                                          | RetinaNet                                                                   
|--------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| MobilenetV2  | 90.4%                                                                        | 91.7%                                                                      |
| ShufflenetV2 |                                                                             |                                                                              |


| COCO2017     | SSD                                                                          | RetinaNet                                                                   
|--------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| MobilenetV2  |                                                                              | 31.7                                                                         |
| ShufflenetV2 |                                                                              |                                                                             |

## Demo
![demo image](demo/V3.png)
![demo image](demo/V4.png)
![demo image](demo/1.png)

## TODO
