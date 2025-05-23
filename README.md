# Brain Tumor Detection and Segmentation in MATLAB (YOLOv2 + ResNet50 + SAM)

This project implements a hybrid deep learning system for detecting and segmenting brain tumors from MRI images using MATLAB. It integrates YOLOv2 for object detection, ResNet50 as a feature extractor, and the Segment Anything Model (SAM) for advanced segmentation.

## 👤 Developer

- **Mertcan Kankılıç**

##  Techniques Used

- **YOLOv2**: For real-time and accurate tumor detection.
- **ResNet50**: Backbone network to improve feature extraction in YOLOv2.
- **Segment Anything Model (SAM)**: Used for high-precision segmentation of detected regions.

## 🗂 Project Structure

```
tumor-segmentation-matlab/
├── src/                   # MATLAB core scripts
├── gui/                   # GUI components
├── models/                # Pre-trained YOLOv2 model (.mat)
├── data/                  # MRI images and annotations
├── results/               # Output visualizations
├── README.md              # Project documentation
├── LICENSE                # License file (MIT)
```

##  Requirements

- MATLAB R2021a or newer
- Required toolboxes:
  - Image Processing Toolbox
  - Deep Learning Toolbox
  - Computer Vision Toolbox

##  How to Run

1. Open MATLAB and navigate to the `src` folder:
   ```matlab
   cd src
   main
   ```
2. In the GUI:
   - **Load Data**: Load the labeled .mat dataset
   - **Load Model**: Import the trained YOLOv2 model
   - **Select Image**: Choose an MRI image for analysis
   - **Analyze**: Perform detection and segmentation

##  Training Parameters

| Parameter                 | Value              |
|---------------------------|--------------------|
| Epochs                   | 30                 |
| Optimizer                | SGDM               |
| Learning Rate            | 1e-3               |
| Mini Batch Size          | 8                  |
| Augmentation             | Rotation, Shear, Reflection, Scaling |
| Train/Validation/Test    | 70% / 15% / 15%     |

##  Evaluation Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

##  Results

The system achieved high accuracy in detecting and segmenting brain tumors. The GUI provides an integrated, user-friendly interface suitable for clinical workflows. SAM enhances post-detection segmentation precision.

##  Dataset

- [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

##  License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

##  Contact

For inquiries or collaboration:  
📧 mertcankankilic27@gmail.com

##  Note on Large Files

This project uses **Git Large File Storage (LFS)** to manage `.mat` files and other large assets.

If you are cloning this repository or contributing to it, please make sure to install Git LFS first:

```bash
git lfs install

##  Large File Notice (Git LFS Required)

This project includes `.mat` files managed via **Git Large File Storage (LFS)**.

To ensure proper download of large files (e.g., result datasets), make sure you have Git LFS installed:

```bash
git lfs install
git lfs pull
