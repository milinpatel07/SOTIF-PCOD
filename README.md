# **Simulation-Based Evaluation of SOTA 3D Object Detection on a SOTIF-Related LiDAR Dataset**

## **Overview**
This repository contains the **LiDAR point cloud dataset** and **evaluation scripts** used for analyzing the performance of **State-of-the-Art (SOTA) 3D object detection models** in the context of **Safety of the Intended Functionality (SOTIF - ISO 21448:2022).** The dataset is generated through simulation and follows the **KITTI format** for compatibility with benchmark models.

### **Key Highlights**
- **Dataset Format**: **KITTI-formatted** LiDAR dataset.
- **Simulation Environment**: Generated using **CARLA**.
- **Weather Conditions**: 21 diverse conditions (**clear, cloudy, rainy**) across different times of the day (**noon, sunset, night**).
- **Dataset Size**: **547 frames**.
- **Pre-Trained Models**: Tested models trained on the **KITTI dataset**.
- **Toolkits Used**: **MMDetection3D** and **OpenPCDet**.
- **Evaluation Metrics**: **Average Precision (AP)** and **Recall**.

---

## **Repository Structure**
