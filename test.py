# ============================================================================
# IMPORT ESSENTIAL LIBRARIES
# ============================================================================
import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import Video, display
from ultralytics import YOLO
import pathlib
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & VISUALIZATION SETTINGS
# ============================================================================
# Configure the visual appearance of Seaborn plots
sns.set(style='darkgrid')
sns.set(rc={'axes.facecolor': '#eae8fa'}, style='darkgrid')

# Dataset paths
Image_dir = '/kaggle/input/cardetection/car/train/images'
dataset_path = '/kaggle/input/cardetection/car'
valid_images_path = os.path.join(dataset_path, 'test', 'images')
post_training_files_path = '/kaggle/working/runs/detect/train'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return image / 255.0

def resize_image(image, size=(640, 640)):
    """Resize image to specified size"""
    return cv2.resize(image, size)

def display_images(post_training_files_path, image_files):
    """Display images from a directory"""
    for image_file in image_files:
        image_path = os.path.join(post_training_files_path, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10), dpi=120)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# ============================================================================
# PART 1: LOAD AND VISUALIZE TRAINING DATA
# ============================================================================
print("Loading and visualizing training data...")

num_samples = 9
image_files = os.listdir(Image_dir)

# Randomly select num_samples images
rand_images = random.sample(image_files, num_samples)

fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i in range(num_samples):
    image = rand_images[i]
    ax = axes[i // 3, i % 3]
    ax.imshow(plt.imread(os.path.join(Image_dir, image)))
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Get the size of the image
image = cv2.imread("/kaggle/input/cardetection/car/train/images/00000_00000_00012_png.rf.23f94508dba03ef2f8bd187da2ec9c26.jpg")
h, w, c = image.shape
print(f"The image has dimensions {w}x{h} and {c} channels.")

# ============================================================================
# PART 2: MODEL SETUP AND QUICK PREDICTION
# ============================================================================
print("\nSetting up YOLO model and performing initial prediction...")

# Use a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Use the model to detect objects
image = "/kaggle/input/cardetection/car/train/images/FisheyeCamera_1_00228_png.rf.e7c43ee9b922f7b2327b8a00ccf46a4c.jpg"
result_predict = model.predict(source=image, imgsz=(640))

# Show results
plot = result_predict[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot))

# ============================================================================
# PART 3: TRAIN THE MODEL
# ============================================================================
print("\nTraining the model...")

# Build from YAML and transfer weights
Final_model = YOLO('yolov8n.pt')

# Training the final model
Result_Final_model = Final_model.train(
    data="/kaggle/input/cardetection/car/data.yaml",
    epochs=30,
    batch=-1,
    optimizer='auto'
)

# ============================================================================
# PART 4: ANALYZE TRAINING RESULTS
# ============================================================================
print("\nAnalyzing training results...")

# Display training result images
image_files = [
    'confusion_matrix_normalized.png',
    'F1_curve.png',
    'P_curve.png',
    'R_curve.png',
    'PR_curve.png',
    'results.png'
]

display_images(post_training_files_path, image_files)

# Read and display results
Result_Final_model = pd.read_csv('/kaggle/working/runs/detect/train/results.csv')
print("\nLast 10 epochs:")
print(Result_Final_model.tail(10))

# Clean column names
Result_Final_model.columns = Result_Final_model.columns.str.strip()

# Create subplots for training metrics
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=Result_Final_model, ax=axs[0, 0])
sns.lineplot(x='epoch', y='train/cls_loss', data=Result_Final_model, ax=axs[0, 1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=Result_Final_model, ax=axs[1, 0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=Result_Final_model, ax=axs[1, 1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=Result_Final_model, ax=axs[2, 0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=Result_Final_model, ax=axs[2, 1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=Result_Final_model, ax=axs[3, 0])
sns.lineplot(x='epoch', y='val/box_loss', data=Result_Final_model, ax=axs[3, 1])
sns.lineplot(x='epoch', y='val/cls_loss', data=Result_Final_model, ax=axs[4, 0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=Result_Final_model, ax=axs[4, 1])

# Set titles and axis labels for each subplot
axs[0, 0].set(title='Train Box Loss')
axs[0, 1].set(title='Train Class Loss')
axs[1, 0].set(title='Train DFL Loss')
axs[1, 1].set(title='Metrics Precision (B)')
axs[2, 0].set(title='Metrics Recall (B)')
axs[2, 1].set(title='Metrics mAP50 (B)')
axs[3, 0].set(title='Metrics mAP50-95 (B)')
axs[3, 1].set(title='Validation Box Loss')
axs[4, 0].set(title='Validation Class Loss')
axs[4, 1].set(title='Validation DFL Loss')

plt.suptitle('Training Metrics and Loss', fontsize=24)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()

# ============================================================================
# PART 5: LOAD AND EVALUATE MODEL ON VALIDATION SET
# ============================================================================
print("\nEvaluating model on validation set...")

# Loading the best performing model
Valid_model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')

# Evaluating the model on the validation set
metrics = Valid_model.val(split='val')

# Print final results
print("\n=== FINAL VALIDATION METRICS ===")
print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
print("recall(B): ", metrics.results_dict["metrics/recall(B)"])
print("mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
print("mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])

# ============================================================================
# PART 6: VALIDATION SET INFERENCE
# ============================================================================
print("\nPerforming inference on validation set...")

# List of all jpg images in the directory
image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]

# Check if there are images in the directory
if len(image_files) > 0:
    # Select 9 images at equal intervals
    num_images = len(image_files)
    step_size = max(1, num_images // 9)  # Ensure the interval is at least 1
    selected_images = [image_files[i] for i in range(0, num_images, step_size)]

    # Prepare subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('Validation Set Inferences', fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])
            
            # Load image
            image = cv2.imread(image_path)
            
            # Check if the image is loaded correctly
            if image is not None:
                # Resize image
                resized_image = resize_image(image, size=(640, 640))
                # Normalize image
                normalized_image = normalize_image(resized_image)
                
                # Convert the normalized image to uint8 data type
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)
                
                # Predict with the model
                results = Valid_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)
                
                # Plot image with labels
                annotated_image = results[0].plot(line_width=1)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ax.imshow(annotated_image_rgb)
            else:
                print(f"Failed to load image {image_path}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 7: VIDEO PREDICTION
# ============================================================================
print("\nPerforming prediction on video...")

Valid_model.predict(source="/kaggle/input/cardetection/video.mp4", show=True, save=True)

# Display the video
Video("result_out.mp4", width=960)

print("\nDone!")
