#!/bin/bash

# ===== 产生npz =====
echo "===== 产生COD10K.npz ====="
python pre_npz.py \
--img_path data/TestDataset/COD10K/Imgs \
--gt_path data/TestDataset/COD10K/GT \
--depth_path data/TestDataset/COD10K/depth \
--npz_path data/COD_test \
--data_name COD10K \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

# ===== 推理 =====
echo "===== 推理COD10K ====="
python test.py \
-i data/COD_test_vit_b/COD10K \
-o data/results/COD10K \
--seg_png_path data/inference_img/COD10K \
--device cuda:0 \
-chk work_dir_cod/DSAM/DSAM.pth

# ===== 评估 =====
echo "===== 评估COD10K ====="
python eval2_COD10K.py
