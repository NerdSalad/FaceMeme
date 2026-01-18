import os
import random
import shutil

random.seed(42)

base_dir = r"D:\Comdur\Main Projects\FaceMeme\Datasets\FER"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

split_ratio = 0.2

os.makedirs(val_dir, exist_ok=True)

for cls in classes:
    src_dir = os.path.join(train_dir, cls)
    dst_dir = os.path.join(val_dir, cls)

    os.makedirs(dst_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    random.shuffle(files)

    val_count = int(len(files) * split_ratio)
    val_files = files[:val_count]

    for f in val_files:
        shutil.move(
            os.path.join(src_dir, f),
            os.path.join(dst_dir, f)
        )

    print(f"{cls}: moved {val_count} files to validation")