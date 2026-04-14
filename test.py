# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
join = os.path.join

import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback
import shutil

torch.manual_seed(1)
np.random.seed(1)

# %% parser
parser = argparse.ArgumentParser(description='run inference on testing set')
parser.add_argument('-i', '--data_path', type=str, required=True,
                    help='path to ONE dataset folder (e.g., data/COD_test_vit_b/CAMO)')
parser.add_argument('-o', '--seg_path_root', type=str, required=True,
                    help='path to save npz results')
parser.add_argument('--seg_png_path', type=str, required=True,
                    help='path to save visualization images')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('-chk', '--checkpoint', type=str,
                    default='work_dir_cod/DSAM/DSAM.pth')
args = parser.parse_args()

# %% utils
def show_mask(mask, ax):
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               edgecolor='blue',
                               facecolor=(0, 0, 0, 0), lw=2))

def compute_dice(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def finetune_model_predict(img_np, box_np, depth_np, boundary,
                           sam_trans, sam_model_tune, device):
    H, W = img_np.shape[:2]

    # image
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None])

    # depth
    resize_depth = sam_trans.apply_image(depth_np)
    resize_dep_tensor = torch.as_tensor(resize_depth.transpose(2, 0, 1)).to(device)
    depth_image = sam_model_tune.preprocess(resize_dep_tensor[None])

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image)
        depth_embedding = sam_model_tune.image_encoder(depth_image)

        # box
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        sparse_box, dense_box = sam_model_tune.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

        # DSAM modules
        pvt_embedding = sam_model_tune.pvt(input_image)[3]
        bc_embedding, pvt_64 = sam_model_tune.BC(pvt_embedding)
        hybrid = torch.cat([pvt_64, bc_embedding], dim=1)
        high_freq = sam_model_tune.DWT(hybrid)

        dense, sparse = sam_model_tune.ME(dense_box, high_freq, sparse_box)

        seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        final_mask = sam_model_tune.loop_finer(seg_prob, depth_embedding, depth_embedding)
        seg_prob = 0.1 * final_mask + 0.9 * seg_prob
        seg_prob = torch.sigmoid(seg_prob)

        seg = seg_prob.cpu().numpy().squeeze()
        seg = (seg > 0.5).astype(np.uint8)

    return seg

# %% model
device = args.device
sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

# %% path setup
npz_data_path = args.data_path
save_path = args.seg_path_root
png_path = args.seg_png_path

os.makedirs(save_path, exist_ok=True)
os.makedirs(png_path, exist_ok=True)

npz_files = [f for f in os.listdir(npz_data_path) if f.endswith('.npz')]

all_dice_scores = []

# %% inference
for npz_file in tqdm(npz_files):
    try:
        npz = np.load(join(npz_data_path, npz_file))

        ori_imgs = npz['imgs']
        ori_gts = npz['gts']
        ori_number = npz['number']
        boundary = npz['boundary']
        dep_imgs = npz['depth_imgs']

        sam_segs = []
        sam_bboxes = []
        sam_dice_scores = []

        for img_id, ori_img in enumerate(ori_imgs):
            gt2D = ori_gts[img_id]
            depth_img = dep_imgs[img_id]

            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))

            bbox = np.array([x_min, y_min, x_max, y_max])

            seg_mask = finetune_model_predict(
                ori_img, bbox, depth_img, boundary,
                sam_trans, sam_model_tune, device
            )

            sam_segs.append(seg_mask)
            sam_bboxes.append(bbox)

            dice = compute_dice(seg_mask > 0, gt2D > 0)
            sam_dice_scores.append(dice)
            all_dice_scores.append(dice)
            # === 新增：保存单张预测mask（用于evaluation） ===
            save_name = ori_number[img_id]
            if save_name.endswith('.jpg'):
                save_name = save_name.replace('.jpg', '.png')
            elif save_name.endswith('.png'):
                pass
            else:
                save_name = save_name + '.png'

            save_path_png = os.path.join(args.seg_path_root, save_name)
            cv2.imwrite(save_path_png, seg_mask * 255)
        # save npz
        np.savez_compressed(
            join(save_path, npz_file),
            medsam_segs=sam_segs,
            gts=ori_gts,
            number=ori_number,
            sam_bboxes=sam_bboxes
        )

        # visualize
        img_id = np.random.randint(0, len(ori_imgs))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(ori_imgs[img_id])
        show_box(sam_bboxes[img_id], axes[0])
        show_mask(ori_gts[img_id], axes[0])
        axes[0].set_title('GT')
        axes[0].axis('off')

        axes[1].imshow(ori_imgs[img_id])
        show_box(sam_bboxes[img_id], axes[1])
        show_mask(sam_segs[img_id], axes[1])
        axes[1].set_title(f'DSAM Dice={sam_dice_scores[img_id]:.3f}')
        axes[1].axis('off')

        fig.savefig(join(png_path, npz_file.replace('.npz', '.png')))
        plt.close(fig)

    except Exception:
        traceback.print_exc()
        print(f'error in {npz_file}')

# %% final result
mDSC = np.nanmean(all_dice_scores)
print("Final mDSC:", mDSC)