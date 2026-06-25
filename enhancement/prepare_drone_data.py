import os
import shutil
from PIL import Image, ImageEnhance
import random

# Paths
TRAIN_DIR = "data/train"
OUTPUT_DIR = "data/drone_pairs"

# Create output folders
os.makedirs(f"{OUTPUT_DIR}/low", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/high", exist_ok=True)

# Get all jpg images
images = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
print(f"Found {len(images)} images")

for i, img_name in enumerate(images):
    img_path = os.path.join(TRAIN_DIR, img_name)
    
    # Open image
    img = Image.open(img_path).convert('RGB')
    
    # Save original as HIGH (bright)
    new_name = f"drone_{i:04d}.jpg"
    img.save(f"{OUTPUT_DIR}/high/{new_name}")
    
    # Create dark version as LOW
    # Reduce brightness randomly between 0.1 and 0.3
    factor = random.uniform(0.1, 0.3)
    dark_img = ImageEnhance.Brightness(img).enhance(factor)
    dark_img.save(f"{OUTPUT_DIR}/low/{new_name}")
    
    print(f"Processed {i+1}/{len(images)}: {new_name}")

print("\nDone! Dataset ready for fine-tuning!")
print(f"LOW images:  {OUTPUT_DIR}/low/")
print(f"HIGH images: {OUTPUT_DIR}/high/")