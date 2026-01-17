import re
import torch
from transformers import pipeline
#2
class DescriptionExtractor:
    
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        self.part_keywords = [
            "screen", "display", "glass", "battery", "power",
            "charging port", "port", "hinge", "keyboard", "keys",
            "speaker", "audio", "microphone", "body", "frame",
            "casing", "lid", "touchpad", "camera"
        ]
        
        self.symptom_keywords = [
            "crack", "broken", "damage", "not working", "loose",
            "drain", "hot", "overheat", "scratch", "dent",
            "bent", "water", "liquid", "sound", "audio"
        ]
    
    def extract(self, description):
        
        if not description or len(description.strip()) < 5:
            return {
                'original': description,
                'summary': description,
                'affected_parts': [],
                'symptoms': [],
                'keywords': [],
                'length_category': 'none'
            }
        
        desc_lower = description.lower()
        
        word_count = len(description.split())
        if word_count < 10:
            length_category = 'short'
            summary = description
        elif word_count < 50:
            length_category = 'medium'
            summary = description
        else:
            length_category = 'long'
            try:
                summary_result = self.summarizer(
                    description,
                    max_length=50,
                    min_length=10,
                    do_sample=False
                )
                summary = summary_result[0]['summary_text']
            except:
                summary = ' '.join(description.split()[:40]) + "..."
        affected_parts = [
            part for part in self.part_keywords
            if part in desc_lower
        ]
        
        symptoms = [
            symptom for symptom in self.symptom_keywords
            if symptom in desc_lower
        ]
        keywords = list(set(affected_parts + symptoms))
        
        return {
            'original': description,
            'summary': summary,
            'affected_parts': affected_parts,
            'symptoms': symptoms,
            'keywords': keywords,
            'length_category': length_category,
            'word_count': word_count
        }
    
    def create_search_text(self, description_info):
        if not description_info['keywords']:
            return description_info['summary']
        search_text = f"{description_info['summary']} {' '.join(description_info['keywords'])}"
        return search_text

