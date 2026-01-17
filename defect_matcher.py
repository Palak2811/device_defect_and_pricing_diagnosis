from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class DefectMatcher:
    
    def __init__(self, defect_db_path="data/defect_database.json",
        use_finetuned=True,
        finetuned_model_path="models/finetuned_clip/best_model"):

        if use_finetuned and os.path.exists(finetuned_model_path):
            
            self.model = CLIPModel.from_pretrained(finetuned_model_path)
            self.processor = CLIPProcessor.from_pretrained(finetuned_model_path)
            self.model_type = "fine-tuned"
        else:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model_type = "pre-trained"
        
        self.model.eval()
        
        with open(defect_db_path, 'r') as f:
            self.defect_db = json.load(f)['defects']
    
        self._precompute_defect_embeddings()
    
    def _precompute_defect_embeddings(self):
        defect_descriptions = [d['description'] for d in self.defect_db]
        
        inputs = self.processor(
            text=defect_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.defect_embeddings = text_features.cpu().numpy()
    
    def match(self, image_path, description_text=None, top_k=3):
        image = Image.open(image_path).convert('RGB')
        
        if description_text and len(description_text.strip()) > 0:
            inputs = self.processor(
                text=[description_text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                text_features = self.model.get_text_features(input_ids=inputs['input_ids'])
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
                fused_features = 0.6 * image_features + 0.4 * text_features
                fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)
            
            query_embedding = fused_features.cpu().numpy()
            method = "multimodal"
            
        else:
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            query_embedding = image_features.cpu().numpy()
            method = "image_only"
        similarities = cosine_similarity(query_embedding, self.defect_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for idx in top_indices:
            defect = self.defect_db[idx]
            score = float(similarities[idx])
            matches.append({
                'defect': defect,
                'similarity_score': score,
                'confidence': score  
            })
        
        return {
            'top_match': matches[0]['defect'],
            'confidence': matches[0]['confidence'],
            'all_matches': matches,
            'method': method,
            'match_count': len(matches)
        }

