# Appearance-and-Relation Networks
We provide the code and models for the following report ([arXiv Preprint](https://arxiv.org/abs/)):

      Appearance-and-Relation Networks for Video Classification
      Limin Wang, Wei Li, Wen Li, and Luc Van Gool
      in arXiv, 2017

ARTNet aims to learn to spatiotemporal features from videos in an end-to-end manner. Its construction is based on a newly-designed moudle, termed as SMART block. ARTNet is a simple and general video architecture and all these relased models are trained **from scratch** on video dataset. Currently, for an engineering compromise between accuracy and efficiency, ARTNet is instantiated with ResNet-18 architecture and trained on the input volume of 112\*\112\*16. 

#### Performance on the validation set of Kinetics

|        Model        | Backbone architecture | Spatial resolution | Top-1 Accuracy | Top-5 Accuracy |
|:-------------------:|:--------------:|:--------------:| :--------------:| :--------------:|
| C2D |    ResNet18   |    112\*112   |  61.2 | 82.6 |
| C3D |    ResNet18   |    112\*112   |  65.6 | 85.7 |
| C3D |    ResNet34   |    112\*112   |  67.1 | 86.9 |
| ARTNet (s) |    ResNet18   |    112\*112   |  67.7 | 87.1 |
| ARTNet (d) |    ResNet18   |    112\*112   |  **69.2** | **88.3** |
| ARTNet+TSN |    ResNet18   |    112\*112   |  **70.7** | **89.3** |

These models are trained on the Kinetics dataset **from scratch** and tested on the validation set. Our training is performed based on the input volume of 112\*112\*16. The test is performed by cropping 25 clips from the videos.

#### Performance on the datasets of HMDB51 and UCF101
|        Model        | Backbone architecture | Spatial resolution | HMDB51 | UCF101 |
|:-------------------:|:--------------:|:--------------:| :--------------:| :--------------:|
| C3D |    ResNet18   |    112\*112   |  62.1 | 89.8 |
| ARTNet (d) |    ResNet18   |    112\*112   |  **67.6** | **93.5** |
| ARTNet+TSN |    ResNet18   |    112\*112   |  **70.9** | **94.3** |

These models learned on the Kinetics dataset are transferred to the HMDB51 and UCF101 datasets. The fine-tuning process is done with TSN framework where the segment number is 2. The performance is reported over three splits by using **only RGB input**.
