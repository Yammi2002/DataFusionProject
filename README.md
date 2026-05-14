# 3D Perception - Heuristic vs. ML Sensor Fusion

This repository implements and compares different sensor fusion strategies for semantic segmentation using the MFNet Dataset. The project evaluates two main architectural approaches: individual U-Net models fused via heuristic methods and a specialized DualNet architecture for machine learning-based fusion.
To improve performance, the use of transfer learning was explored.

## Project Overview
The core objective is to segment images across 9 classes (background, car, person, bike, curve, stop, guardrail, cone, and bump) by exploiting data from both RGB cameras and thermal sensors. 

### 1. Heuristic Fusion (Late Fusion)
Two identical U-Net architectures were trained separately on RGB and Thermal data. Their outputs are combined using heuristic-based Late Fusion methods:
* **Mean Fusion**: Arithmetic mean of probabilities from both sensors.
* **Joint Probability**: Normalized element-wise product of probability maps.
* **Maximum Confidence**: Selection of the highest probability value per pixel.

### 2. DualNet Architecture (ML Fusion)
The DualNet model uses two parallel encoder branches to process RGB and Thermal tensors separately. The fusion occurs at the bottleneck using a concatenation operator, allowing the model to learn the optimal way to combine sensor data. It also incorporates **Modality Dropout** to ensure the model remains robust even if one sensor fails.

## Experimental Results
Performance is measured using the Mean Intersection over Union (mIoU). Results show that fusing information consistently outperforms individual sensors, with Heuristic (Product) achieving the highest score in this setup.

| Model Strategy | Scratch | Augmented | Transfer Learning |
| :--- | :--- | :--- | :--- |
| RGB Only | 0.2356 | 0.2621 | 0.4420 |
| Thermal Only | 0.2066 | 0.2186 | 0.4248 |
| Heuristic (Product) | 0.2827 | 0.3043 | 0.4963 |
| DualNet (ML Fusion) | 0.2451 | 0.2407 | 0.4305 |

## Setup and Dataset
The MFNet dataset contains over 1500 image pairs[cite: 1]. To run the evaluation:

1. **Download the Dataset**: Place the raw data in a `data/` folder in the root directory.
2. **Pre-processing**: The dataloader expects thermal data integrated into the 4th channel of an RGBA input.
3. **Model Weights**: Place the `.pth` files (e.g., `smp_dualnet_pretrained_v1.pth`) in a `models/` folder.

```text
project-root/
├── models/             # Pretrained weights
├── data/               # MFNet Dataset
└── noteooks/notebook.py # Core logic
```

## Methodology Note
All models were trained using Weighted Cross-Entropy to address class imbalance, particularly the dominance of the background class. Transfer Learning from the SMP library was utilized to significantly improve convergence and accuracy.
