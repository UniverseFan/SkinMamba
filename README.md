# SkinMamba
This is the official code repository for **"SkinMamba: Segmentation and Classification of Skin Cancer with Multi-level Context Understanding" (IJCNN 2025)**. 

[![Paper]([https://img.shields.io/badge/IJCNN_2025-Paper-red)](https://doi.org/YOUR_DOI_HERE](https://ieeexplore.ieee.org/abstract/document/11227563))

## Key Contributions
Compared to conventional architectural baselines, **SkinMamba** introduces specific optimization for skin lesion imaging under complex medical scenarios:
1. **Multi-level Context Understanding:** Advanced State Space Models (Mamba) tailored for robust feature extraction and long-range dependency modeling without quadratic complexity.
2. **High-Quality Low-Light Corpus:** An independently constructed, high-quality low-light dataset specifically designed for benchmarking robust skin cancer segmentation under poor illumination.
3. **Full-Link Physical Noise Model:** Integration of a realistic physical noise pipeline to simulate and enhance low-light medical imaging degradation, closing the gap between laboratory training and real-world clinical deployment.

---

Main Environments
```bash
conda create -n skinmamba python=3.8
conda activate skinmamba
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

1. Prepare the Dataset
ISIC datasets (with Low-Light Corpus & Noise Simulation)
Download the baseline ISIC17/ISIC18 datasets and place them under ./datasets/.

Apply our proposed full-link physical noise model using makedata_ISIC.py to generate the low-light robust evaluation corpus.

The directory structure should look like this:

Plaintext
./datasets/
  ├── isic17/
  │   ├── train/
  │   │   ├── images/ (*.png)
  │   │   └── masks/  (*.png)
  │   └── val/
  │       ├── images/ (*.png)
  │       └── masks/  (*.png)
  └── Synapse/
      ├── lists/
      ├── test_vol_h5/

Train & Evaluate SkinMamba
Bash
# Run training and evaluation on ISIC datasets (with multi-level context processing)
python train.py

# Run training and evaluation on Synapse dataset
python train_synapse.py

# For evaluation and testing
python test.py

If you find our work or the low-light physical noise model corpus useful for your research, please cite our IJCNN 2025 paper:
@inproceedings{zhang2025skinmamba,
  title={SkinMamba: Segmentation and Classification of Skin Cancer with Multi-level Context Understanding},
  author={Zhang, Yufan and Nan, Guoshun and Yue, Yang and Jia, Chengyao and Gao, Yutong and Xiao, Zuye and Wu, Xu},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
