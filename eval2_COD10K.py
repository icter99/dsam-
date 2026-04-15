# -*- coding: utf-8 -*-
import os
import cv2
from tqdm import tqdm
import torch
import sod_metrics as M
import torch.nn.functional as F

# ===== metrics =====
FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

# ===== path =====
mask_root = 'data/TestDataset/COD10K/GT'
pred_root = 'data/results/COD10K'

# ===== upsample =====
def _upsample_like(src, tar):
    src = torch.tensor(src, dtype=torch.float32)
    tar = torch.tensor(tar)
    src = F.interpolate(src.unsqueeze(0).unsqueeze(0), size=tar.shape, mode='bilinear')
    src = src.squeeze(0).squeeze(0).numpy()
    return src

# ===== 改动1：以pred为基准 =====
pred_name_list = sorted([f for f in os.listdir(pred_root) if f.endswith('.png')])

print("GT total:", len(os.listdir(mask_root)))
print("Pred total:", len(pred_name_list))

valid_count = 0

# ===== evaluation loop =====
for pred_name in tqdm(pred_name_list, total=len(pred_name_list)):
    pred_path = os.path.join(pred_root, pred_name)
    mask_path = os.path.join(mask_root, pred_name)

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # ===== 安全判断 =====
    if pred is None or mask is None:
        print(f"[WARNING] skip: {pred_name}")
        continue

    if len(pred.shape) != 2:
        pred = pred[:, :, 0]
    if len(mask.shape) != 2:
        mask = mask[:, :, 0]

    # resize pred to GT size
    pred = _upsample_like(pred, mask)

    assert pred.shape == mask.shape

    # ===== metrics =====
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

    valid_count += 1

# ===== results =====
fm = FM.get_results()['fm']
wfm = WFM.get_results()['wfm']
sm = SM.get_results()['sm']
em = EM.get_results()['em']
mae = MAE.get_results()['mae']

print("\n===== Evaluation Results =====")
print("Valid samples:", valid_count)

print(
    'Smeasure:', sm.round(3), '; ',
    'wFmeasure:', wfm.round(3), '; ',
    'meanFm:', fm['curve'].mean().round(3), '; ',
    'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
    'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
    'MAE:', mae.round(3), '; ',
    'adpEm:', em['adp'].round(3), '; ',
    'adpFm:', fm['adp'].round(3), '; ',
    'maxFm:', fm['curve'].max().round(3),
    sep=''
)

# ===== save =====
with open("result_COD10K.txt", "a+") as f:
    print(
        'CAMO -> ',
        'Smeasure:', sm.round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'maxFm:', fm['curve'].max().round(3),
        file=f
    )