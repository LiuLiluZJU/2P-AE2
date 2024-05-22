# Vertebrea-based Global X-ray to CT Registration for Thoracic Surgeries

by Lilu Liu, Yanmei Jiao, Zhou An, Honghai Ma, Chunlin Zhou, Haojian Lu, Jian Hu, Rong Xiong, Yue Wang.

<!-- &#x26A0; **More details of this repository are COMING SOON!** -->

## Introduction
X-ray to CT registration is an essential technique to provide on-site guidance for clinicians and medical robots by aligning preoperative information with intraoperative images. Current methods focus on local registration with small capture ranges and necessitate a manual initial alignment before precise registration. Some existing global methods are likely to fail in thoracic surgeries because of the respiratory motion and the nearly colinear nature of vertebrae landmarks. In this study, we propose an vertebrae-based global X-ray to CT registration method with the assist of clinical setups for thoracic surgeries. Firstly, vertebrae centroids are automatically localized by CNN-based networks in CT and X-ray for establishing 2-D/3-D correspondences. Then, inspired by clinical setup, we address the degradation of colinear landmarks of 6-DoF pose estimation by introducing a 4-DoF solver. Considering the inaccurate priori and landmark mislocalization, the solver is embedded into the Adaptive Error-Aware Estimator (AE<sup>2</sup>) to simultaneously estimate weights and aggregate candidate poses. Finally, the whole method is trained in an end-to-end manner for better performance. Evaluations on both the public LIDC-IDRI dataset and clinical dataset demonstrate that our method outperforms existing optimization-based and learning-based approaches in terms of registration accuracy and success rate.

<!-- <img src="figs/overview.jpg#pic_left" alt="avatar" style="zoom:30%;" /> -->
<img src="figs/overview.jpg#pic_left" alt="avatar" style="zoom:40%;" />

This repository includes:
* Training and testing scripts using Python and PyTorch;
* Pretrained models of the proposed 2P-AE<sup>2</sup> and 2P-AE<sup>2</sup>-e2e; and
* Processed training, validation and testing datasets derived from thorax CT scans and X-ray images.

## Setup

### Prerequisites
* Ubuntu $\geq$ 16.04
* Python $\geq$ 3.7
* CUDA $\geq$ 10.1


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

The test dataset can be found in [Google Drive](https://drive.google.com/drive/folders/1w-_ldq6AyKv-kcKopkJg80LgCeUzWj9W?usp=share_link)

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

