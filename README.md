# CP8307 - Semantic Segmentation Project
This repository contains the Python code for different semantic segmentation methods. The project focused on comparing a deep learning segmentation model to various traditional methods. The [U-Net](https://arxiv.org/abs/1505.04597) was implemented in TensorFlow and the model was trained on the free GPU resources available from Google Colaboratory. Traditional segmentation models such as [Thresholding](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html), [Chan-Vese](https://ieeexplore.ieee.org/document/902291), and [Region Adjacency Graphs](https://ieeexplore.ieee.org/document/841950) were implemented and run locally.

[**Report**](../master/Semantic_Segmentation_Report.pdf)

[**Presentation**](../master/Semantic_Segmentation_Presentation.pdf)

## U-Net Implementation
The TensorFlow implementation of the U-Net can be found at [**CP8307_U-Net_Semantic_Segmentation.ipynb**](../master/CP8307_U-Net_Semantic_Segmentation.ipynb)

## Traditional Model Implementation
The code for the traditional models can be found at [**CP8307_Traditional_Segmentation.py**](../master/CP8307_Traditional_Segmentation.py)

**Parameter Values (all experimentally determined and informed by theory):**

Thresholding (custom): Thresholding Difference: 0.005

Chan-Vese (using skimage.segmentation.chan_vese): mu: 2, lambda1: 1, lambda2: 1, tol: 1e-3, max_iter: 200, dt: 0.5, init_level_set: "checkerboard"

RAG: (using skimage.segmentation.slic): compactness: 30, n_segments: 300. (using skimage.future.graph.cut_threshold): thresh: 45

## Dataset
[**A Large Scale Fish Dataset**](https://www.kaggle.com/crowww/a-large-scale-fish-dataset) contains 9000 image-mask pairs of 9 classes of fish. An 80%/20% split was used to train and evaluate the deep learning model which left 7200 images to train on and 1800 images to evaluate. The same 1800 images used to evaluate the U-Net were used on the traditional models to ensure a fair comparison.

## Results
A summary of the results are displayed below. It is evident that the deep learning model performed the best, followed by the RAG. The Intersection-over-Union (IoU) and the Dice Score were not calculated for the Chan-Vese and Thresholding models due to very poor performance.

|      | **U-Net** |  **RAG** | **Chan-Vese** |  **Thresholding** |
|:----:|:-----:|:----:|:---------:|:-------------:|
|  **IoU** |  0.89 | 0.72 |    N/C    |      N/C      |
| **Dice** |  0.94 | 0.81 |    N/C    |      N/C      |

![image](https://user-images.githubusercontent.com/23387743/146186533-39287bf7-37ff-4e6b-af9e-ef405a4738d6.png)
