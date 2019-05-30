# AIC19

This repository contains our source code of Track 1 at the [AI City Challenge Workshop at CVPR 2019](https://www.aicitychallenge.org/).   
The source code of Track 1 is built in Python, and is executed on Windows.

## Introduction

### AI City Challenge Workshop at CVPR 2019 
The challenge focus on Intelligent Transportation System (ITS) problems, such as:

#### Track1 - City-scale multi-camera vehicle tracking
#### Track2 - City-scale multi-camera vehicle re-identification
#### Track3 - Traffic anomaly detection 

### Pipeline
The pipeline of our system is as follow:  
![pipeline](https://github.com/yrims/AIC19/blob/master/Images/pipeline.png)

#### Homography based multi-view fusion
This method first uses homography matrix to project the vehicles in source videos to real world coordinate(latitude, longitude), then generates the ROI images which mask the high projected error region in each camera, finally, integrates the ROI image to inspect the multi-view fusion result.    
#### Multi-Target Single-Camera Tracking (MTSC)    
In MTSC tracking, we use the `DeepSort` and `TC` tracker and adjust the result according the generated ROI.
The AI City Challenge also provides three MTSC trackers: `TC`, `DeepSort`, `Moana`.    

#### Multi-Target Multi-Camera Tracking (MTMC)
In MTMC tracking, we calculate the four conditions in loss:    
`image similarity`, `trajectory consistency`, `driving direction`, `travel time`,
and match the vehicle pairs having minimum loss.

## Code strucure
Under the `/MTMC` folder, there are 3 python files:    
1.`get_track_info.py`: Integrate the single camera tracking files in each scenario.
2.`get_bbox_img.py`: Crop the vehicle bouding boxes in integrated single camera tracking files.
3.`MTMC.py`: Multi-Target Multi-Camera Tracking based on multi-view fusion.

Under the `/Img_model` folder, there are 2 python files:
1. `train_cnn.py`: Train the vehicle classification model for feature extraction.    
2. `feature_extract.py`: Extract the feature on the cropped vehicle bboxes.    

## Dependency
- Python 3.6.7
- Tensorflow-gpu 1.8.0
- Keras 2.2.4
- OpenCV 4.0.0

## Reference
