import os
import numpy as np
from PIL import Image, ImageEnhance
import random

def augment_tb_images():
    # Kode augmentasi yang saya berikan sebelumnya
    input_dir = r"D:\A15-CNN\dataset\tuberkulosis"
    output_dir = r"D:\A15-CNN\dataset\tuberkulosis_augmented"
    os.makedirs(output_dir, exist_ok=True)
    
    augmentations_per_image = 4
    tb_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Memulai augmentasi {len(tb_files)} gambar TB...")
    
    for i, filename in enumerate(tb_files):
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path)
        
        # Simpan original
        original_output_path = os.path.join(output_dir, f"Tuberculosis-{i*6 + 1}.png")
        image.save(original_output_path)
        
        # Augmentasi
        for aug_idx in range(augmentations_per_image):
            augmented_image = image.copy()
            augmentation_type = random.choice(['contrast', 'brightness', 'flip', 'combined'])
            
            if augmentation_type == 'contrast':
                contrast_factor = random.uniform(0.8, 1.5)
                enhancer = ImageEnhance.Contrast(augmented_image)
                augmented_image = enhancer.enhance(contrast_factor)
                
            elif augmentation_type == 'brightness':
                brightness_factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Brightness(augmented_image)
                augmented_image = enhancer.enhance(brightness_factor)
                
            elif augmentation_type == 'flip':
                augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
                
            elif augmentation_type == 'combined':
                contrast_factor = random.uniform(0.8, 1.4)
                enhancer = ImageEnhance.Contrast(augmented_image)
                augmented_image = enhancer.enhance(contrast_factor)
                
                brightness_factor = random.uniform(0.8, 1.3)
                enhancer = ImageEnhance.Brightness(augmented_image)
                augmented_image = enhancer.enhance(brightness_factor)
                
                if random.random() > 0.5:
                    augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            output_filename = f"Tuberculosis-{i*6 + aug_idx + 2}.png"
            output_path = os.path.join(output_dir, output_filename)
            augmented_image.save(output_path)
        
        if (i + 1) % 100 == 0:
            print(f"Diproses {i + 1}/{len(tb_files)} gambar...")
    
    print(f"Augmentasi selesai! Total gambar TB: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    augment_tb_images()