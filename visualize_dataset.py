import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from config import *


# # ============================================================
# # Paths
# # ============================================================

DATASET_PATH = "brisc2025"

# CLASSIFICATION_PATH = os.path.join(
#     DATASET_PATH,
#     "classification_task",
#     "train"
# )

SEGMENTATION_IMAGE_PATH = os.path.join(
    DATASET_PATH,
    "segmentation_task",
    "train",
    "images"
)

SEGMENTATION_MASK_PATH = os.path.join(
    DATASET_PATH,
    "segmentation_task",
    "train",
    "masks"
)


# ============================================================
# 1. Show a classification Train image
# ============================================================

def show_classification_Train_image():

    selected_class = random.choice(CLASS_NAMES)

    class_path = os.path.join(
        Classification_Train,
        selected_class
    )

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_name = random.choice(images)

    image_path = os.path.join(
        class_path,
        image_name
    )

    image = Image.open(image_path)

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap="gray")

    plt.title(
        f"Classification\nClass: {selected_class}\nFile: {image_name}"
    )

    plt.axis("off")
    plt.show()


# ============================================================
# 1. Show a classification Test image
# ============================================================


def show_classification_Test_image():

    selected_class = random.choice(CLASS_NAMES)

    class_path = os.path.join(
        Classification_Test,
        selected_class
    )

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_name = random.choice(images)

    image_path = os.path.join(
        class_path,
        image_name
    )

    image = Image.open(image_path)

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap="gray")

    plt.title(
        f"Classification\nClass: {selected_class}\nFile: {image_name}"
    )

    plt.axis("off")
    plt.show()


# ============================================================
# 3. Show segmentation Train image
# ============================================================

def show_segmentation_Train_image():

    images = [
        file for file in os.listdir(SEGMENTATION_IMAGE_PATH)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_name = random.choice(images)

    image_path = os.path.join(
        SEGMENTATION_IMAGE_PATH,
        image_name
    )

    # Assuming mask has the same filename
    mask_path = os.path.join(
        SEGMENTATION_MASK_PATH,
        image_name
    )

    image = Image.open(image_path).convert("L")

    if not os.path.exists(mask_path):
        print("Mask not found for:", image_name)
        return

    mask = Image.open(mask_path).convert("L")

    # --------------------------------------------------------
    # Display image and mask
    # --------------------------------------------------------

    plt.figure(figsize=(12, 4))

    # Original MRI
    plt.subplot(1, 3, 1)

    plt.imshow(image, cmap="gray")
    plt.title("MRI Image")
    plt.axis("off")

    # Mask
    plt.subplot(1, 3, 2)

    plt.imshow(mask, cmap="gray")
    plt.title("Tumor Mask")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)

    plt.imshow(image, cmap="gray")
    plt.imshow(mask, cmap="Reds", alpha=0.4)

    plt.title("MRI + Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    print("Showing a classification Train image...")
    show_classification_Train_image()

    print("Showing a classification Test image...")
    show_classification_Test_image()

    print("Showing a segmentation Train image...")
    show_segmentation_Train_image()

    