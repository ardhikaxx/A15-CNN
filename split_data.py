import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset():
    # Kode splitting dataset yang saya berikan
    normal_dir = r"D:\A15-CNN\dataset\normal"
    tb_dir = r"D:\A15-CNN\dataset\tuberkulosis_augmented"
    output_base_dir = r"D:\A15-CNN\data"
    
    # Buat struktur direktori
    directories = [
        os.path.join(output_base_dir, 'train', 'normal'),
        os.path.join(output_base_dir, 'train', 'tuberkulosis'),
        os.path.join(output_base_dir, 'val', 'normal'),
        os.path.join(output_base_dir, 'val', 'tuberkulosis'),
        os.path.join(output_base_dir, 'test', 'normal'),
        os.path.join(output_base_dir, 'test', 'tuberkulosis')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Daftar file
    normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tb_files = [f for f in os.listdir(tb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Jumlah gambar Normal: {len(normal_files)}")
    print(f"Jumlah gambar Tuberkulosis: {len(tb_files)}")
    
    # Split data
    normal_train, normal_temp = train_test_split(normal_files, test_size=0.3, random_state=42)
    normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)
    
    tb_train, tb_temp = train_test_split(tb_files, test_size=0.3, random_state=42)
    tb_val, tb_test = train_test_split(tb_temp, test_size=0.5, random_state=42)
    
    # Copy files
    def copy_files(file_list, source_dir, target_dir):
        for filename in file_list:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.copy2(src_path, dst_path)
    
    # Copy normal
    copy_files(normal_train, normal_dir, os.path.join(output_base_dir, 'train', 'normal'))
    copy_files(normal_val, normal_dir, os.path.join(output_base_dir, 'val', 'normal'))
    copy_files(normal_test, normal_dir, os.path.join(output_base_dir, 'test', 'normal'))
    
    # Copy TB
    copy_files(tb_train, tb_dir, os.path.join(output_base_dir, 'train', 'tuberkulosis'))
    copy_files(tb_val, tb_dir, os.path.join(output_base_dir, 'val', 'tuberkulosis'))
    copy_files(tb_test, tb_dir, os.path.join(output_base_dir, 'test', 'tuberkulosis'))
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Train - Normal: {len(normal_train)}, TB: {len(tb_train)}")
    print(f"Val - Normal: {len(normal_val)}, TB: {len(tb_val)}")
    print(f"Test - Normal: {len(normal_test)}, TB: {len(tb_test)}")
    print(f"Total: {len(normal_files) + len(tb_files)} gambar")

if __name__ == "__main__":
    split_dataset()