import os
import io
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


def collate_fn_instances(batch):
    images, targets, infos = [], [], []
    for img, tgt, info in batch:
        images.append(img)
        targets.append(tgt)
        infos.append(info)
    images = torch.stack(images, dim=0)
    return images, targets, infos

class CSVDataset(Dataset):
    def __init__(self, args, df, augment=False):
        self.df = df
        self.args = args
        self.augment = augment
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image_id = row['image_id']
        label = row['label']
        
        # Load Image
        image_path = os.path.join(self.args.dataset_dir, self.args.image_dir, label, f"{image_id}.png")
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.new('RGB', (self.args.image_size, self.args.image_size))

        # Load Masks
        raw_masks = None
        if label == 'forged':
            mask_path = os.path.join(self.args.dataset_dir, self.args.mask_dir, f"{image_id}.npy")
            if os.path.exists(mask_path):
                try:
                    raw_masks = np.load(mask_path)
                    if len(raw_masks.shape) == 2:
                        raw_masks = raw_masks[np.newaxis, :, :]
                except Exception:
                    raw_masks = None 

        if self.augment:
            # --- NEW: Inject Copy-Move Augmentation Here ---
            # This is done BEFORE resizing to preserve high-freq artifacts
            # img, raw_masks = augment_copy_move(img, raw_masks, p=0.5)
            # -----------------------------------------------

            # Geometric: Flips
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if raw_masks is not None:
                    raw_masks = np.flip(raw_masks, axis=2)

            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                if raw_masks is not None:
                    raw_masks = np.flip(raw_masks, axis=1)

            # Geometric: Rotations
            k = random.randint(0, 3)
            if k > 0:
                modes = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
                img = img.transpose(modes[k])
                if raw_masks is not None:
                    raw_masks = np.rot90(raw_masks, k=k, axes=(1, 2))
            
            if raw_masks is not None:
                raw_masks = raw_masks.copy()


        # Resize Image
        img = img.resize((self.args.image_size, self.args.image_size), resample=Image.BILINEAR)
        
        # Resize Masks
        if raw_masks is not None:
            resized_stack = []
            for j in range(raw_masks.shape[0]):
                m_pil = Image.fromarray(raw_masks[j].astype(np.uint8))
                m_pil = m_pil.resize((self.args.image_size, self.args.image_size), resample=Image.NEAREST)
                resized_stack.append(np.array(m_pil))
            
            target_instances = np.stack(resized_stack, axis=0)
            
            # Filter empty masks
            valid_indices = [idx for idx, m in enumerate(target_instances) if np.sum(m) > 0]
            if len(valid_indices) > 0:
                target_instances = target_instances[valid_indices]
            else:
                target_instances = None
        else:
            target_instances = None

        img_tensor = self.norm_transform(img)
        
        if target_instances is None:
            target_tensor = torch.zeros((0, self.args.image_size, self.args.image_size), dtype=torch.float32)
            vis_mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.uint8)
        else:
            target_tensor = torch.tensor(target_instances).float()
            vis_mask = target_instances

        info = {'image_id': image_id, 'orig_masks': vis_mask}

        return img_tensor, target_tensor, info