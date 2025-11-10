# A15-CNN: Sistem Deteksi Tuberkulosis Pernapasan (ICD-10 A15) Berbasis Deep Learning pada Citra X-Ray Dada dengan Optimasi Class Imbalance

A15-CNN adalah sistem kecerdasan buatan canggih yang dirancang khusus untuk mendeteksi Tuberkulosis Pernapasan (ICD-10 A15) dari citra X-Ray dada menggunakan arsitektur Convolutional Neural Network (CNN). Sistem ini mengatasi tantangan class imbalance dan memberikan akurasi tinggi dalam klasifikasi biner antara kondisi Normal dan Tuberkulosis.

# Fitur Utama

1. Deteksi Tuberkulosis berdasarkan standar ICD-10 A15
2. Optimasi Class Imbalance dengan dataset seimbang (3,500 gambar per kelas)
3. Arsitektur CNN Kustom yang dioptimalkan untuk citra X-Ray
4. Evaluasi Komprehensif dengan metrik klinis yang relevan
5. Visualisasi Lengkap untuk analisis performa model
6. Analisis Misklasifikasi untuk peningkatan model
7. Generasi Laporan LaTeX untuk publikasi penelitian

# Statistik Dataset

Jumlah gambar Normal: 3,500
Jumlah gambar Tuberkulosis: 3,500
Total: 7,000 gambar

=== PEMBAGIAN DATASET ===
Train - Normal: 2,450, TB: 2,450
Val   - Normal: 525,   TB: 525
Test  - Normal: 525,   TB: 525

# Teknologi yang Digunakan
## Deep Learning Framework
- PyTorch 2.0+ - Framework deep learning utama
- TorchVision - Preprocessing dan augmentasi data
- CUDA Support - Akselerasi GPU untuk training

## Computer Vision & Processing
- PIL/Pillow - Image processing dan manipulasi
- OpenCV - Computer vision operations
- NumPy - Komputasi numerik

## Evaluasi & Visualisasi
- Scikit-learn - Metrik evaluasi dan analisis
- Matplotlib & Seaborn - Visualisasi data dan hasil
- Pandas - Data processing dan analisis