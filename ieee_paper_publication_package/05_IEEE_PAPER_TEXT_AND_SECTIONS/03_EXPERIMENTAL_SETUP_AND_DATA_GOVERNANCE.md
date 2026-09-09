# Section III: Experimental Setup & Data Governance

## A. Dataset Governance & Split Protocol
All model training and hyperparameter tuning were conducted strictly on the **RiceLeafDiseaseBD** benchmark dataset under strict zero-leakage governance:
- **Primary Training Set**: 2,824 images (80%)
- **Validation Set**: 706 images (10%)
- **Primary Test Set**: 706 images (10%)
- **Pathology Classes**: Healthy (118), Blast (114), Brown Spot (132), Leaf Smut (109), Tungro (113), Sheath Blight (120).

External datasets (*Sethy5932*, *RiceLeafDiseaseBD5*, and *RiceSeg5932*) were strictly isolated during training to prevent data contamination and reserved exclusively for post-training explainability validation and domain-shift robustness tests.

## B. Training Environment & Hardware Target
- **Hardware Target**: NVIDIA GeForce GTX 1650 (4 GB GDDR6 VRAM, CUDA Compute Capability 7.5).
- **Optimizer**: AdamW ($eta_1 = 0.9, eta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Learning Rate Schedule**: Initial learning rate $\eta_0 = 10^{-4}$ with Cosine Annealing decay over 30 epochs ($\eta_{	ext{min}} = 10^{-6}$).
- **Batch Size**: 16 with Automatic Mixed Precision (AMP `float16`) to maintain VRAM consumption below 1.2 GB.
- **Data Augmentation**: Random affine rotation ($\pm 15^\circ$), horizontal/vertical flips, color jitter ($\pm 10\%$), and normalization to ImageNet distribution.
