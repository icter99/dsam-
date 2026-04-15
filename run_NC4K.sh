#!/bin/bash

# ===== 产生npz =====
echo "===== 产生NC4K.npz ====="
python pre_npz.py \
--img_path data/TestDataset/NC4K/Imgs \
--gt_path data/TestDataset/NC4K/GT \
--depth_path data/TestDataset/NC4K/depth \
--npz_path data/COD_test \
--data_name NC4K \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

# ===== 推理 =====
echo "===== 推理NC4K ====="
python test.py \
-i data/COD_test_vit_b/NC4K \
-o data/results/NC4K \
--seg_png_path data/inference_img/NC4K \
--device cuda:0 \
-chk work_dir_cod/DSAM/DSAM.pth

# ===== 评估 =====
echo "===== 评估NC4K ====="
python eval2_NC4K.py
