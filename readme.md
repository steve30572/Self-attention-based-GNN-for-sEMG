# Stretchable array electromyography sensor with graph neural network for static and dynamic gestures recognition system


This is our PyTorch implementation for the paper: Stretchable array electromyography sensor with graph neural network for static and dynamic gestures recognition system

Hyeyun Lee, Soyoung Lee, Jaeseong Kim, Heesoo Jung, Kyung Jae Yoon, Srinivas Gandla1, Hogun Park, and Sunkook Kim

## Example data

The example data can be downloaded via https://drive.google.com/drive/folders/13Uzxqm9rogA-53rJebkK65nBvhEPzGy8?usp=sharing
Please place the 'data' directory in the same home directory.

For your custom dataset, the shape of the data should equal to   
(# number of data, timescale (24 in our case), # of sensors (8), # of scales (7)).

We upload the sample preprocess python file as well.

## How to run the code

After downloading the example dataset, please run the code as follows.

```bash
python train_model.py
```

## Abstract

With advances in artificial intelligence (AI)-based algorithms, gesture recognition accuracy from sEMG signals has continued to increase. Spatiotemporal multichannel-sEMG signals substantially increase the quantity and reliability of the data for any type of study. Here, we report an array of bipolar stretchable sEMG electrodes with a self-attention based graph neural network to recognize gestures with high accuracy. The array is designed to spatially cover the skeletal muscles to acquire the regional sampling data of EMG activity from 18 different gestures. The system can differentiate individual static and dynamic gestures with ~97% accuracy when training a single trial per gesture. Moreover, a sticky patchwork of holes adhered to an array sensor enables skin-like attributes such as stretchability and water vapor permeability and aids in delivering stable EMG signals. In addition, the recognition accuracy (~95%) remained unchanged even after long-term testing for over 72 h and being reused more than 10 times.

## Information

Affiliation: LearnDataLab, SKKU    
E-mail : steve305@g.skku.edu


