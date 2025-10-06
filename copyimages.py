import os
import shutil

def get_label_files_with_class(label_dir, target_class, max_files):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    selected = []
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file)) as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                if class_id == target_class:
                    selected.append(label_file)
                    break
        if len(selected) >= max_files:
            break
    return selected

images_dir = r"E:\Combine_Model_Version_2\DETECTION_DATA\Train\images"
labels_dir = r"E:\Combine_Model_Version_2\DETECTION_DATA\Train\labels"
output_dir = r"D:\workspace\Projects\transformer\class_18_subset"
test_output_dir = r"D:\workspace\Projects\transformer\test_images_18"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

chosen_labels_3 = get_label_files_with_class(labels_dir, 18, 3)
chosen_labels_5 = get_label_files_with_class(labels_dir, 18, 5)

# Copy subset (3 samples) with labels and images
for lbl in chosen_labels_3:
    img_name = lbl.replace('.txt', '.jpg')
    shutil.copy(os.path.join(images_dir, img_name), os.path.join(output_dir, img_name))
    shutil.copy(os.path.join(labels_dir, lbl), os.path.join(output_dir, lbl))

# Copy test images (5 samples)
for lbl in chosen_labels_5:
    img_name = lbl.replace('.txt', '.jpg')
    shutil.copy(os.path.join(images_dir, img_name), os.path.join(test_output_dir, img_name))
