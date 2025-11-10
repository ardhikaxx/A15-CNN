import os
import shutil
import hashlib
from sklearn.model_selection import train_test_split
import numpy as np

def create_clean_splits():
    """Create clean train/val/test splits without leakage"""
    
    print("=== CREATING CLEAN DATA SPLITS ===")
    
    # Paths
    normal_dir = r"D:\A15-CNN\dataset\normal"
    tb_dir = r"D:\A15-CNN\dataset\tuberkulosis_augmented"
    output_dir = r"D:\A15-CNN\data_clean"
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Structure
    splits = ['train', 'val', 'test']
    classes = ['normal', 'tuberkulosis']
    
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Get all files with hashes to ensure uniqueness
    def get_files_with_hashes(directory):
        files = []
        hashes = set()
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(directory, filename)
                
                # Calculate hash
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Only add if unique
                if file_hash not in hashes:
                    hashes.add(file_hash)
                    files.append((filename, filepath, file_hash))
        
        return files
    
    # Get unique files
    normal_files = get_files_with_hashes(normal_dir)
    tb_files = get_files_with_hashes(tb_dir)
    
    print(f"Unique normal files: {len(normal_files)}")
    print(f"Unique TB files: {len(tb_files)}")
    
    # Split data (stratified)
    normal_filenames = [f[0] for f in normal_files]
    normal_filepaths = [f[1] for f in normal_files]
    
    tb_filenames = [f[0] for f in tb_files]
    tb_filepaths = [f[1] for f in tb_files]
    
    # Create labels
    normal_labels = [0] * len(normal_files)
    tb_labels = [1] * len(tb_files)
    
    all_files = normal_filepaths + tb_filepaths
    all_labels = normal_labels + tb_labels
    all_filenames = normal_filenames + tb_filenames
    
    # First split: train vs temp (val+test)
    train_files, temp_files, train_labels, temp_labels, train_names, temp_names = train_test_split(
        all_files, all_labels, all_filenames, 
        test_size=0.3, 
        stratify=all_labels,
        random_state=42
    )
    
    # Second split: val vs test
    val_files, test_files, val_labels, test_labels, val_names, test_names = train_test_split(
        temp_files, temp_labels, temp_names,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Copy files to respective directories
    def copy_files(files, filenames, labels, split_name):
        for filepath, filename, label in zip(files, filenames, labels):
            class_name = 'normal' if label == 0 else 'tuberkulosis'
            dest_dir = os.path.join(output_dir, split_name, class_name)
            shutil.copy2(filepath, os.path.join(dest_dir, filename))
    
    copy_files(train_files, train_names, train_labels, 'train')
    copy_files(val_files, val_names, val_labels, 'val')
    copy_files(test_files, test_names, test_labels, 'test')
    
    # Verify
    print(f"\n=== CLEAN SPLITS CREATED ===")
    for split in splits:
        normal_count = len(os.listdir(os.path.join(output_dir, split, 'normal')))
        tb_count = len(os.listdir(os.path.join(output_dir, split, 'tuberkulosis')))
        print(f"{split.upper()}: Normal={normal_count}, TB={tb_count}, Total={normal_count + tb_count}")
    
    return output_dir

if __name__ == "__main__":
    clean_data_dir = create_clean_splits()
    print(f"\nClean data created at: {clean_data_dir}")