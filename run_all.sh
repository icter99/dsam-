#!/bin/bash
# ===== npz =====
python pre_npz.py \
--img_path data/TestDataset/CAMO/Imgs \
--gt_path data/TestDataset/CAMO/GT \
--depth_path data/TestDataset/CAMO/depth \
--npz_path data/COD_test \
--data_name CAMO \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

python pre_npz.py \
--img_path data/TestDataset/COD10K/Imgs \
--gt_path data/TestDataset/COD10K/GT \
--depth_path data/TestDataset/COD10K/depth \
--npz_path data/COD_test \
--data_name COD10K \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

python pre_npz.py \
--img_path data/TestDataset/NC4K/Imgs \
--gt_path data/TestDataset/NC4K/GT \
--depth_path data/TestDataset/NC4K/depth \
--npz_path data/COD_test \
--data_name NC4K \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth

# ===== 推理 =====
python Mytest.py -i data/TestDataset/CAMO -o results/CAMO -chk work_dir_cod/DSAM/DSAM.pth
python Mytest.py -i data/TestDataset/COD10K -o results/COD10K -chk work_dir_cod/DSAM/DSAM.pth
python Mytest.py -i data/TestDataset/NC4K -o results/NC4K -chk work_dir_cod/DSAM/DSAM.pth

# ===== 评估 =====
echo "===== CAMO ====="
python eval_CAMO.py
echo "===== COD10K ====="
python eval_COD10K.py
echo "===== NC4K ====="
python eval_NC4K.py
