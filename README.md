# SkinMamba
This is the official code repository for **"SkinMamba: Segmentation and Classification of Skin Cancer with Multi-level Context Understanding" (IJCNN 2025)**. 

[![Paper]([https://img.shields.io/badge/IJCNN_2025-Paper-red)](https://doi.org/YOUR_DOI_HERE]([https://ieeexplore.ieee.org/abstract/document/11227563](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11227563&tag=1)))

## Key Contributions
1. **Dual-Task Architecture (SkinMamba):** We introduce **SkinMamba**, a comprehensive joint framework comprising two specialized modules: **MVMNet** tailored for dense skin lesion segmentation and **AMNet** designed for precise classification.
2. **Core Feature Extractor (SkinBlock):** We design **SkinBlock** as the foundational block of both MVMNet and AMNet. By capturing multi-level context understanding efficiently, the SkinBlock serves as the core engine for robust deep feature representation.
3. **Robust Low-Light Benchmarking:** Leveraging our independently constructed high-quality low-light corpus and a full-link physical noise model, we evaluate SkinMamba under rigorous real-world degradation scenarios. 
4. **Outstanding Performance:** Extensive experiments conducted on the ISIC and Synapse datasets demonstrate that SkinMamba delivers outstanding, competitive performance in both medical image segmentation and classification tasks.

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
