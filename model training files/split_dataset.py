import os
import random
import shutil

dataset_path = r"C:\Users\lanre\OneDrive\Desktop\dataset"

images_dir = os.path.join(dataset_path, "images")
labels_dir = os.path.join(dataset_path, "labels")

# Make sure folders exist
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, folder), exist_ok=True)

# Get all image files
image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(image_files)

num_images = len(image_files)
train_split = int(num_images * 0.8)
val_split = int(num_images * 0.9)

for i, img in enumerate(image_files):
    if i < train_split:
        split = "train"
    elif i < val_split:
        split = "val"
    else:
        split = "test"

    # Move image
    src_img = os.path.join(images_dir, img)
    dst_img = os.path.join(images_dir, split, img)
    shutil.move(src_img, dst_img)

    # Find corresponding label (handles filenames with multiple dots)
    base_name = os.path.splitext(img)[0]  # removes ONLY last extension
    label_name = base_name + ".txt"
    src_label = os.path.join(labels_dir, label_name)
    dst_label = os.path.join(labels_dir, split, label_name)

    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"[WARNING] No label found for image: {img} (skipping)")
