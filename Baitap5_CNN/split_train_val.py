import os
import shutil
import random

def split_train_val_from_30_images(train_dir, output_train_dir, output_val_dir, train_count=20, val_count=10, seed=42):
    random.seed(seed)

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    classes = os.listdir(train_dir)
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        images = os.listdir(class_path)
        images = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        if len(images) < train_count + val_count:
            print(f"⚠️ Không đủ ảnh trong lớp '{cls}' ({len(images)} ảnh). Bỏ qua.")
            continue

        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]

        os.makedirs(os.path.join(output_train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(output_val_dir, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_train_dir, cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_val_dir, cls, img))

    print("✅ Đã chia xong dữ liệu: 20 train + 10 val mỗi lớp.")

split_train_val_from_30_images(
    train_dir='HoaVietNam/train',
    output_train_dir='HoaVietNam/new_train',
    output_val_dir='HoaVietNam/val'
)