# End-to-end Adaptive Error-aware Initialization for Fully Automatic 2-D/3-D Rigid Registration in Thoracic Surgeries

by Lilu Liu, Yanmei Jiao, Zhou An, Honghai Ma, Chunlin Zhou, Haojian Lu, Jian Hu, Rong Xiong, Yue Wang.

<!-- &#x26A0; **More details of this repository are COMING SOON!** -->

## Introduction
2-D/3-D rigid registration is an essential technique to provide on-site guidance for navigating medical robots by aligning 3-D CT with 2-D X-ray image. Current methods necessitate an initial alignment within the capture range for precise registration, also known as initialization. However, automatic initialization is challenging due to poor image quality and the degradation of landmarks. In this study, we propose an end-to-end adaptive error-aware initialization method to robustly perform the 2-D/3-D registration for thoracic surgeries. Firstly, vertebrae landmarks are automatically localized by CNN-based networks in CT and X-ray for establishing 2-D/3-D correspondences. Then, inspired by clinical setup, we address the degradation of vertebrae landmarks in 6-DoF pose estimation by introducing a 4-DoF solver. Considering the mislocalization of landmarks, the solver is embedded into the Adaptive Error-Aware Estimator AE<sup>2</sup> to simultaneously estimate weights and aggregate candidate poses. Finally, the whole method is trained in an end-to-end manner for better performance. On the simulated dataset, our method achieves 10.34mm of mean target registration error (mTRE) and 9.17% of gross failure rate (GFR). On the clinical dataset, our method achieves 2.25mm of mTRE and 12.5% of GFR with intensity-based refinement, outperforming the state-of-the-art approaches (p-value<0.05).

<!-- <img src="figs/overview.png#pic_left" alt="avatar" style="zoom:30%;" /> -->
<img src="figs/overview.png#pic_left" alt="avatar" style="zoom:40%;" />

This repository includes:
* Training and testing scripts using Python and PyTorch;
* Pretrained models of the proposed 2P-AE<sup>2</sup> and 2P-AE<sup>2</sup>-e2e; and
* Processed training, validation and testing datasets derived from thorax CT scans and X-ray images.

## Setup

### Prerequisites
* Ubuntu $\geq$ 16/04
* Python $\geq$ 3.7
* CUDA $\geq$ 10.2


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

