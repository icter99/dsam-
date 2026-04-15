#!/bin/bash

# ===== 产生npz =====
echo "===== 产生CAMO.npz ====="
python pre_npz.py \
--img_path data/TestDataset/CAMO/Imgs \
--gt_path data/TestDataset/CAMO/GT \
--depth_path data/TestDataset/CAMO/depth \
--npz_path data/COD_test \
--data_name CAMO \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

# ===== 推理 =====
echo "===== 推理CAMO ====="
python test.py \
-i data/COD_test_vit_b/CAMO \
-o data/results/CAMO \
--seg_png_path data/inference_img/CAMO \
--device cuda:0 \
-chk work_dir_cod/DSAM/DSAM.pth

# ===== 评估 =====
echo "===== 评估CAMO ====="
python eval2_CAMO.py
