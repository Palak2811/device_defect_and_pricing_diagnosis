import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

from aws_cust.fine_tuning.finetune_dataset import create_dataloaders


class CLIPFineTuner:
    
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def contrastive_loss(self, logits_per_image, logits_per_text):
        
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss
    
    def train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )
            
            loss = self.contrastive_loss(
                outputs.logits_per_image,
                outputs.logits_per_text
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=False
                )
                loss = self.contrastive_loss(
                    outputs.logits_per_image,
                    outputs.logits_per_text
                )
                total_loss += loss.item()
                logits = outputs.logits_per_image
                predictions = logits.argmax(dim=1)
                labels = torch.arange(len(predictions), device=self.device)
                
                correct += (predictions == labels).sum().item()
                total += len(predictions)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=10,
        learning_rate=5e-6,
        warmup_epochs=1,
        save_dir='models/finetuned_clip'
    ):
        
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7
        )
        
        for epoch in range(1, num_epochs + 1):
            
            print(f"EPOCH {epoch}/{num_epochs}")
           
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            val_loss, val_accuracy = self.validate(val_loader)
            
            if epoch > warmup_epochs:
                scheduler.step()
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch} Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val Acc:    {val_accuracy:.2%}")
            print(f"   LR:         {current_lr:.2e}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, 'best_model')
                self.model.save_pretrained(best_model_path)
                self.processor.save_pretrained(best_model_path)
                print(f"   Saved best model (val_loss: {val_loss:.4f})")
            
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}')
                self.model.save_pretrained(checkpoint_path)
                print(f"    Saved checkpoint at epoch {epoch}")
        
        print("\n" + "="*70)
        print(" TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved to: {os.path.join(save_dir, 'best_model')}")
        
        return self.training_history
    
    def save_training_plot(self, save_path='models/finetuned_clip/training_plot.png'):
     
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.training_history['val_accuracy'], 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

def main():
    CONFIG = {
        'phone_csv': r'E:\fortransferee\mlproject7\aws_cust\data\phone\phone_training_data2134.csv',
        'laptop_csv': r'E:\fortransferee\mlproject7\aws_cust\data\laptop\laptop_training_data2134.csv',
        'batch_size': 16,
        'num_epochs': 15,
        'learning_rate': 5e-6,
        'train_split': 0.8,
        'save_dir': 'models/finetuned_clip'
    }
    
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    finetuner = CLIPFineTuner()
    train_loader, val_loader, df = create_dataloaders(
        phone_csv_path=CONFIG['phone_csv'],
        laptop_csv_path=CONFIG['laptop_csv'],
        processor=finetuner.processor,
        batch_size=CONFIG['batch_size'],
        train_split=CONFIG['train_split']
    )
    history = finetuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        save_dir=CONFIG['save_dir']
    )
    finetuner.save_training_plot()
    

if __name__ == "__main__":
    main()