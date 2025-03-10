import os
import random
import shutil

dataset_folder = "D:/mmdetection3d/data/kitti/Dataset Combined"
training_folder = os.path.join(dataset_folder, "training")
testing_folder = os.path.join(dataset_folder, "testing")

# Specify the split ratio (e.g., 80% training, 20% testing)
split_ratio = 0.8

# Get a list of all data files in the dataset folder
data_files = os.listdir(dataset_folder)
random.shuffle(data_files)  # Shuffle the data files randomly

# Determine the split index
split_index = int(len(data_files) * split_ratio)

# Move files to training folder
for filename in data_files[:split_index]:
    source_path = os.path.join(dataset_folder, filename)
    destination_path = os.path.join(training_folder, filename)
    shutil.move(source_path, destination_path)

# Move files to testing folder
for filename in data_files[split_index:]:
    source_path = os.path.join(dataset_folder, filename)
    destination_path = os.path.join(testing_folder, filename)
    shutil.move(source_path, destination_path)
