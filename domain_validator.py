from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
#1
class DomainValidator:
    
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        
        self.valid_categories = [
            "a smartphone or mobile phone",
            "a laptop computer",
            "a desktop computer or monitor",
            "electronic device with screen"
        ]
        
        self.invalid_categories = [
            "a person or human face",
            "an animal or pet",
            "nature or landscape",
            "food or drink",
            "random object or item"
        ]
    def validate(self, image_path, threshold=0.3):
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            all_categories = self.valid_categories + self.invalid_categories
            
            inputs = self.processor(
                text=all_categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)[0]
            
            max_idx = probs.argmax().item()
            max_prob = probs[max_idx].item()
            detected_category = all_categories[max_idx]
            
            is_valid = detected_category in self.valid_categories
            
            if is_valid and max_prob > threshold:
                return {
                    'is_valid': True,
                    'confidence': max_prob,
                    'detected_category': detected_category,
                    'reason': f"Valid device detected: {detected_category}",
                    'device_type': self._extract_device_type(detected_category)
                }
            elif is_valid:
                return {
                    'is_valid': False,
                    'confidence': max_prob,
                    'detected_category': detected_category,
                    'reason': f"Device detected but confidence too low ({max_prob:.2f})",
                    'device_type': None
                }
            else:
                return {
                    'is_valid': False,
                    'confidence': max_prob,
                    'detected_category': detected_category,
                    'reason': f"Not a device - detected: {detected_category}",
                    'device_type': None
                }
                
        except Exception as e:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'detected_category': 'error',
                'reason': f"Error processing image: {str(e)}",
                'device_type': None
            }
    
    def _extract_device_type(self, category):
        """Extract simple device type from category"""
        if "phone" in category.lower():
            return "phone"
        elif "laptop" in category.lower():
            return "laptop"
        elif "computer" in category.lower() or "monitor" in category.lower():
            return "computer"
        else:
            return "device"


