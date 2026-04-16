import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import logging
import csv
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.CWDLoss import CriterionCWD
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual

# set seeds
torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
np.random.seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _upsample_like_1024(src):
    src = F.interpolate(src, size=(1024, 1024), mode='bilinear')
    return src


class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted([f for f in os.listdir(self.data_root) if f.endswith('.npz')])
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU)
        # if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.ori_imgs = np.vstack([d['imgs'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        self.boundary = np.vstack([d['boundary'] for d in self.npz_data])
        self.depth_embeddings = np.vstack([d['depth_embeddings'] for d in self.npz_data])
        print(f"img_embeddings.shape={self.img_embeddings.shape}, ori_gts.shape={self.ori_gts.shape}, "
              f"boundary.shape={self.boundary.shape}", f"depth_embeddings.shape={self.depth_embeddings.shape}")

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        img = self.ori_imgs[index]
        boundary = self.boundary[index]
        depth_embed = self.depth_embeddings[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(img).float(), torch.tensor(gt2D[None, :, :]).long(), torch.tensor(bboxes).float(),\
               torch.tensor(boundary[None, :, :]).long(), torch.tensor(depth_embed).float()

def validate(model, val_dataset, device):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    all_mae = []
    all_dice = []
    with torch.no_grad():
        for image_embedding, img, gt2D, boxes, boundary, depth_embedding in val_loader:
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            boundary = torch.as_tensor(boundary, dtype=torch.float, device=device)
            image_embedding = torch.as_tensor(image_embedding, dtype=torch.float, device=device)
            depth_embedding = torch.as_tensor(depth_embedding, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings_box, dense_embeddings_box = model.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
            resize_img_tensor = np.transpose(img, (0, 3, 1, 2)).to(device)
            input_image = _upsample_like_1024(resize_img_tensor)
            pvt_embedding = model.pvt(input_image)[3]
            bc_embedding, pvt_64 = model.BC(pvt_embedding)
            hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
            high_frequency = model.DWT(hybrid_embedding)
            dense_embeddings, sparse_embeddings = model.ME(
                dense_embeddings_box, high_frequency, sparse_embeddings_box
            )
            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            final_mask = model.loop_finer(mask_predictions, depth_embedding, depth_embedding)
            mask_predictions = 0.1 * final_mask + 0.9 * mask_predictions
            pred = torch.sigmoid(mask_predictions)
            mae = torch.abs(pred - gt2D.to(device).float()).mean().item()
            all_mae.append(mae)
            pred_binary = (pred > 0.5).float()
            gt_binary = (gt2D.to(device) > 0.5).float()
            intersection = (pred_binary * gt_binary).sum(dim=(1, 2, 3))
            union = pred_binary.sum(dim=(1, 2, 3)) + gt_binary.sum(dim=(1, 2, 3))
            dice = ((2. * intersection + 1e-6) / (union + 1e-6)).mean().item()
            all_dice.append(dice)
    model.train()
    return sum(all_mae) / len(all_mae), sum(all_dice) / len(all_dice)


# %% test dataset class and dataloader
npz_tr_path = 'data/npz_vit_b/COD_train_subset_20'
val_npz_path = 'data/npz_vit_b/COD_train_val_10'
work_dir = './work_dir_cod'
task_name = 'DSAM_exp_C'
# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'
device = 'cuda:0'
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)

# setup logging and result dirs
log_dir = join('log', 'exp_C')
result_dir = join('train_result', 'exp_C')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = join(log_dir, f'expC_train_{timestamp}.log')
result_csv = join(result_dir, f'expC_metrics_{timestamp}.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_file}")
logger.info(f"Result CSV: {result_csv}")
logger.info(f"Train subset: {npz_tr_path}")
logger.info(f"Val subset: {val_npz_path}")

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
CWD_loss = CriterionCWD(norm_type='channel', divergence='kl', temperature=4.0)

num_epochs = 20
losses = []
metric_records = []
best_loss = 1e10
train_dataset = NpzDataset(npz_tr_path)
val_dataset = NpzDataset(val_npz_path)
mask_threshold = 0.0
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for epoch in range(1, num_epochs+1):
    epoch_loss = 0
    # train
    for step, (image_embedding, img, gt2D, boxes, boundary, depth_embedding) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):

        with torch.no_grad():
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            boundary = torch.as_tensor(boundary, dtype=torch.float, device=device)
            # boundary = boun_conv(boundary)
            image_embedding = torch.as_tensor(image_embedding, dtype=torch.float, device=device)
            depth_embedding = torch.as_tensor(depth_embedding, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            # get prompt embeddings
            sparse_embeddings_box, dense_embeddings_box = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )

            resize_img_tensor = np.transpose(img, (0, 3, 1, 2)).to(device)
            input_image = _upsample_like_1024(resize_img_tensor)
            pvt_embedding = sam_model.pvt(input_image)[3]

        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
        # bc_embedding shape:" 1, 256, 64, 64
        distill_loss = CWD_loss(bc_embedding, depth_embedding)
        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model.DWT(hybrid_embedding)

        dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_box,
                                                           high_frequency, sparse_embeddings_box)

        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)

        mask_predictions = 0.1*final_mask + 0.9*mask_predictions

        b_target = boundary.squeeze(1)
        b_weight = b_target * 4.0 + 1.0  # 边缘区域权重 5，背景区域权重 1
        b_loss = F.binary_cross_entropy_with_logits(
            mask_predictions.squeeze(1), b_target, weight=b_weight, reduction='mean'
        )
        loss = 0.8*seg_loss(mask_predictions, gt2D.to(device).float()) + 0.1*distill_loss + 0.1*b_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    losses.append(epoch_loss)
    logger.info(f'EPOCH: {epoch}, Loss: {epoch_loss:.6f}')
    record = {'epoch': epoch, 'train_loss': epoch_loss, 'val_mae': None, 'val_dice': None}
    # validation every 5 epochs
    if epoch % 5 == 0 or epoch == num_epochs:
        val_mae, val_dice = validate(sam_model, val_dataset, device)
        logger.info(f'>>> VAL: Epoch {epoch}, MAE: {val_mae:.4f}, Dice: {val_dice:.4f}')
        record['val_mae'] = val_mae
        record['val_dice'] = val_dice
    metric_records.append(record)
    # save metrics after each epoch
    with open(result_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_mae', 'val_dice'])
        writer.writeheader()
        writer.writerows(metric_records)
    # save the latest model checkpoint
    if epoch >= 15 and epoch % 5 == 0:
        ckpt_path = join(model_save_path, str(epoch) + 'sam_model.pth')
        torch.save(sam_model.state_dict(), ckpt_path)
        logger.info(f'Saved checkpoint: {ckpt_path}')
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_ckpt = join(model_save_path, 'sam_model_best.pth')
        torch.save(sam_model.state_dict(), best_ckpt)
        logger.info(f'Saved best model: {best_ckpt}')
# plot loss
plt.plot(range(1, len(losses)+1), losses)
plt.title('Dice + Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
loss_plot = join(model_save_path, 'train_loss.png')
plt.savefig(loss_plot)
plt.close()
logger.info(f"Training finished. Total epochs: {num_epochs}. Best loss: {best_loss:.6f}")
logger.info(f"Loss plot saved: {loss_plot}")
logger.info(f"Metrics CSV saved: {result_csv}")