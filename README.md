# **Evaluation of SOTA 3D Object Detection on a SOTIF-Related Dataset**

## **Overview**
This repository contains a **simulation-based dataset** and **evaluation scripts** for analyzing the performance of **State-of-the-Art (SOTA) 3D object detection models** on **LiDAR point cloud data** in the context of **Safety of the Intended Functionality (SOTIF - ISO 21448:2022).** 

The dataset is prepared in **KITTI format** and is tested using **pre-trained models trained on the KITTI dataset**. The goal is to assess how well these models adapt to a dataset generated from a **SOTIF-related Use Case** under diverse environmental conditions.

---

## **Key Highlights**
- **Dataset Format**: **KITTI**-formatted LiDAR dataset.
- **Weather Conditions**: **Clear, cloudy, rainy** at **noon, sunset, and night**.
- **Frames**: **547 frames** generated via simulation.
- **Pre-Trained Models**: Used **KITTI-trained** models for evaluation.
- **Toolkits Used**: **MMDetection3D** and **OpenPCDet**.
- **Metrics**: **Average Precision (AP)** and **Recall**.

---

## **Repository Structure**
