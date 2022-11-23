# Robust and Automatic Initialization for 2-D/3-D Rigid Registration with Error Awareness

By Lilu Liu, Yanmei Jiao, Zhou An, Honghai Ma, Chunlin Zhou, Haojian Lu, Jian Hu, Rong Xiong, Yue Wang.

&#x26A0; **More details of this repository are COMING SOON!**

## Introduction
2-D/3-D rigid registration is commonly used to provide on-site guidance for clinicians by aligning the pre-operative 3-D computed tomography with the intra-operative 2-D X-ray image. For automatic 2-D/3-D registration, a good initialization is required to supply an initial alignment within the capture range of existing intensity-based registration methods. However, performing a robust initialization is still challenging due to the low quality of X-ray images which brings unavoidable localization errors to fiducials of registration. In this paper, we propose an error-aware framework to robustly and automatically initialize the 2-D/3-D registration for thoracic surgeries. On the simulated dataset, our method achieves 10.34 mm of median mean target registration error (mTRE) and 9.17% of gross failure rate (GFR). On the clinical dataset, our method achieves 2.25 mm of median mTRE and 12.5% of GFR with intensity-based refinement, outperforming the state-of-the-art initialization approaches (p-value<0.05). This work is promising to free clinicians from long-term manual registration and can be extended to other surgical procedures demanding robust and accurate target guidance.

<!-- <img src="figs/overview.png#pic_left" alt="avatar" style="zoom:30%;" /> -->
<img src="figs/overview.png#pic_left" alt="avatar" style="zoom:40%;" />

This repository includes:
* Training and testing scripts using Python and PyTorch;
* Pretrained models of the proposed 2P-EARE and 2P-EARE-e2e; and
* Processed training, validation and testing datasets derived from thorax CT scans and X-ray images.

Code has been tested with Python 3.7, CUDA 10.1 and PyTorch 1.6.

## Setup

### Prerequisites
* Linux or OSX/mac OS
* Python 3
* CUDA


### Installation
<!-- * PyTorch >= 1.6
* SimpleITK
* OpenCV
* SciPy
* Numpy -->
Necessary Python packages can be installed by

```bash
pip install -r requirements.txt
```

## Dataset

Coming soon.

## Testing
The reuslts of proposed method on simulated dataset can be tested by  
```bash
python test_sim.py --solver 2pe2e
```

## Training
To train your own model, you can type the following code:
```bash
python train.py
```
The end-to-end training can be implemented by the following script which loads the pretrained model and refines the weights through the pose loss
```bash
python train_2p.py
```

