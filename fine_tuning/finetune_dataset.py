import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pandas as pd
import os
from functools import partial
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeviceDefectDataset(Dataset):
    
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.placeholder_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        text = str(row['text_prompt'])
        image = None
        
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
                image.load()
            
        except (OSError, IOError, Exception) as e:
            error_type = type(e).__name__
            error_msg = str(e)[:80]  
            
            if idx % 100 == 0 or 'truncated' in error_msg.lower():
                print(f"Image error at idx {idx}: {img_path}")
                print(f"    {error_type}: {error_msg}")
                print(f"    Using placeholder image")
            
            image = self.placeholder_image.copy()
        
        return {'image': image, 'text': text}


def collate_batch(batch, processor):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    
    try:
        encoded = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        return encoded
        
    except Exception as e:
        print(f" Batch processing error: {e}")
        placeholder_imgs = [Image.new('RGB', (224, 224), color='gray')] * len(images)
        placeholder_texts = ["placeholder"] * len(texts)
        
        return processor(
            text=placeholder_texts,
            images=placeholder_imgs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )


def create_dataloaders(phone_csv_path, laptop_csv_path, processor, batch_size=32, train_split=0.8):

    phone_df = pd.read_csv(phone_csv_path)
    laptop_df = pd.read_csv(laptop_csv_path)
    
    combined_df = pd.concat([phone_df, laptop_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int(len(combined_df) * train_split)
    train_df = combined_df[:train_size]
    val_df = combined_df[train_size:]
    
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv', index=False)
    
    train_dataset = DeviceDefectDataset('data/splits/train.csv', processor)
    val_dataset = DeviceDefectDataset('data/splits/val.csv', processor)
    
    collate_fn = partial(collate_batch, processor=processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, combined_df 