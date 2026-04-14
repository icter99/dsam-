#!/bin/bash
# ===== npz =====


python pre_npz.py \
--img_path data/TestDataset/NC4K/Imgs \
--gt_path data/TestDataset/NC4K/GT \
--depth_path data/TestDataset/NC4K/depth \
--npz_path data/COD_test \
--data_name NC4K \
--checkpoint work_dir_cod/SAM/sam_vit_b_01ec64.pth



# ===== 推理 =====
python test.py \
-i data/COD_test_vit_b/COD10K \
-o data/results/COD10K \
--seg_png_path data/inference_img/COD10K \
--device cuda:0 \
-chk work_dir_cod/DSAM/DSAM.pth

python test.py \
-i data/COD_test_vit_b/NC4K \
-o data/results/NC4K \
--seg_png_path data/inference_img/NC4K \
--device cuda:0 \
-chk work_dir_cod/DSAM/DSAM.pth


# ===== 评估 =====
# echo "===== CAMO ====="
# python eval_CAMO.py
echo "===== COD10K ====="
python eval2_COD10K.py
echo "===== NC4K ====="
python eval2_NC4K.py
