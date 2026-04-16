import os
import numpy as np

np.random.seed(2024)

src_dir = 'data/npz_vit_b/COD_train'
dst_train_dir = 'data/npz_vit_b/COD_train_subset_20'
dst_val_dir = 'data/npz_vit_b/COD_train_val_10'

os.makedirs(dst_train_dir, exist_ok=True)
os.makedirs(dst_val_dir, exist_ok=True)

npz_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.npz')])
print(f"Found {len(npz_files)} npz files: {npz_files}")

npz_data = [np.load(os.path.join(src_dir, f)) for f in npz_files]

imgs = np.vstack([d['imgs'] for d in npz_data])
gts = np.vstack([d['gts'] for d in npz_data])
depth_imgs = np.vstack([d['depth_imgs'] for d in npz_data])
img_embeddings = np.vstack([d['img_embeddings'] for d in npz_data])
boundary = np.vstack([d['boundary'] for d in npz_data])
depth_embeddings = np.vstack([d['depth_embeddings'] for d in npz_data])
numbers = []
for d in npz_data:
    numbers.extend(list(d['number']))

N = gts.shape[0]
print(f"Total samples: {N}")

indices = np.random.permutation(N)
train_num = max(1, int(N * 0.2))
val_num = max(1, int(N * 0.1))

train_idx = indices[:train_num]
val_idx = indices[train_num:train_num + val_num]

print(f"Train subset: {len(train_idx)}, Val subset: {len(val_idx)}")

np.savez_compressed(
    os.path.join(dst_train_dir, 'COD_train_subset_20.npz'),
    imgs=imgs[train_idx],
    gts=gts[train_idx],
    depth_imgs=depth_imgs[train_idx],
    number=[numbers[i] for i in train_idx],
    img_embeddings=img_embeddings[train_idx],
    boundary=boundary[train_idx],
    depth_embeddings=depth_embeddings[train_idx],
)

np.savez_compressed(
    os.path.join(dst_val_dir, 'COD_train_val_10.npz'),
    imgs=imgs[val_idx],
    gts=gts[val_idx],
    depth_imgs=depth_imgs[val_idx],
    number=[numbers[i] for i in val_idx],
    img_embeddings=img_embeddings[val_idx],
    boundary=boundary[val_idx],
    depth_embeddings=depth_embeddings[val_idx],
)

print("Done! Saved to:")
print(f"  {dst_train_dir}/COD_train_subset_20.npz")
print(f"  {dst_val_dir}/COD_train_val_10.npz")
