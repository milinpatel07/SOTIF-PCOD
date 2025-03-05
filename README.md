# **Performance Evaluation of 3D Object Detection Models in a Simulation-Based Generated Dataset of a SOTIF-Related Use Case**

## 📍 Overview
This repository contains the **LiDAR dataset and scripts** used for evaluating **State-of-the-Art (SOTA) 3D object detection models** in a **simulation-based generated dataset for a SOTIF-related use case**. The dataset follows the **KITTI format** and is generated using **CARLA**, simulating **21 diverse weather conditions** across different times of the day.

The evaluation is conducted using **pre-trained models trained on KITTI**, tested with **MMDetection3D** and **OpenPCDet** toolkits to assess **performance variations under simulated conditions**.

---

## 📂 **Repository Structure**
This repository is organized as follows:

📂 SOTIF_Scenario_Dataset/ # KITTI-formatted LiDAR dataset │-- 📂 ImageSets/ # Training/testing splits │-- 📂 kitti_gt_database/ # Ground truth database │-- 📂 testing/ # Testing dataset │-- 📜 kitti_dbinfos_train.pkl # Training dataset information │-- 📜 kitti_infos_test.pkl # Test dataset info │-- 📜 kitti_infos_train.pkl # Train dataset info │-- 📜 kitti_infos_trainval.pkl # Train+Validation dataset info │-- 📜 kitti_infos_val.pkl # Validation dataset info │ 

📂 Carla dataset generation/ # Dataset generation scripts │-- 📜 carla_data_descriptor.py # Describes dataset properties │-- 📜 carla_weather_presets.txt # Weather settings for simulation │-- 📜 CARTI_Dataset_V1.0.py # Dataset generation script │-- 📜 CMM_CARLA_Config.py # CARLA configuration script │-- 📜 lane_change.py # Lane change maneuver script │-- 📜 LaneChange.xml # XML configuration for scenario │ 📜 README.md # Project documentation 📜 requirements.txt # Dependencies 📜 LICENSE # MIT License 📜 dataset_snapshot.png # Preview of dataset (image) 📜 scenario_snapshot.png # Snapshot of the simulation environment


## 📍 **Scenario Description**
This project simulates a **SOTIF-related use case** in a **multi-lane highway scenario** where an **Ego-Vehicle (LiDAR-equipped)** navigates under varying **weather conditions**. The key elements of the scenario include:

- **Ego-Vehicle** (Blue) drives in Lane 3, adjusting speed based on traffic.
- **Fast-moving vehicle (Red, 90 km/h)** overtakes a **slow-moving vehicle (Green, 60 km/h)**.
- **Ego-Vehicle detects the slow vehicle** and decelerates to avoid collision.
- **LiDAR sensor performance** is evaluated under **21 different weather conditions**.
- **Dataset records LiDAR point cloud frames** in **KITTI format**, ensuring compatibility with benchmark models.

📸 **Scenario Visualization**:  
![Use Case_Description](https://github.com/user-attachments/assets/ac1801aa-6c6d-4784-b428-16021ebedeb4)

---

## 📸 **Dataset Visualization**
Here’s a **2D visualization of a sample LiDAR point cloud frame** from the dataset:

📸 **Dataset Sample**:  
![dataset_overview](https://github.com/user-attachments/assets/197b4a9c-59c8-4168-a6d3-75db4065d15f)



---

## 📊 **Evaluation Methodology**
### **1️⃣ Dataset Preparation**
- **547 frames** of **LiDAR point cloud data** formatted in **KITTI standard**.
- **Simulated using CARLA**, with custom weather conditions.
- **Dataset split** into training, validation, and test sets.

### **2️⃣ Object Detection Models Used**
- Pre-trained **KITTI-based models** applied using:
  - **MMDetection3D**
  - **OpenPCDet**
  
### **3️⃣ Performance Metrics**
- **Average Precision (AP)**
- **Recall**
- **Intersection over Union (IoU) Thresholds** (0.30, 0.50, 0.70)

---

## 📈 **Results & Findings**
- **KITTI-trained models** show **performance variations** under different simulated weather conditions.
- **Domain gap** between **real-world KITTI data** and **simulation-based data** is analyzed.
- Detailed results are presented in the research papers cited below.

---

## 📄 **Citations & References**
This dataset and methodology were used in the following research papers:

### **1️⃣ Conference Paper**
📖 **Patel, Milin and Jung, Rolf**  
*"Simulation-Based Performance Evaluation of 3D Object Detection Methods with Deep Learning for a LiDAR Point Cloud Dataset in a SOTIF-related Use Case"*  
🚀 *Proceedings of the 10th International Conference on Vehicle Technology and Intelligent Transport Systems (VEHITS 2024)*  
🔗 [DOI: 10.5220/0012707300003702](https://doi.org/10.5220/0012707300003702)  


### **2️⃣ arXiv Preprint**
📖 **Patel, Milin and Jung, Rolf**  
*"Uncertainty Representation in a SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection"*  
🚀 *arXiv preprint, 2025.*  
🔗 [arXiv Link](https://arxiv.org/abs/2503.02087)  


🔗 License
This project is licensed under the MIT License. See LICENSE for details.

📬 Contact
For any inquiries or collaborations, please reach out via milinp101996@gmail.com or open an issue on GitHub.


