from domain_validator import DomainValidator
from description_extractor import DescriptionExtractor
from defect_matcher import DefectMatcher
from condition_grader import ConditionGrader
from aws_cust.predictor.price_predictor import PricePredictor
import time
import json

class DiagnosisSystem:
    
    
    def __init__(self, load_price_model=True):
        
        print("\n[1/5] Loading Domain Validator...")
        self.validator = DomainValidator()
        
        print("[2/5] Loading Description Extractor...")
        self.extractor = DescriptionExtractor()
        
        print("[3/5] Loading Defect Matcher...")
        self.matcher = DefectMatcher()
        
        print("[4/5] Loading Condition Grader...")
        self.grader = ConditionGrader()
        
        print("[5/5] Loading Price Predictor...")
        self.predictor = PricePredictor()
        if load_price_model:
            try:
                self.predictor.load_model()
            except Exception as e:
                print(f"   Could not load price model - {e}")
                self.predictor = None
        
        print("SYSTEM READY")

    
    def diagnose(self, image_path, description="", device_info=None):
       
        start_time = time.time()
        
        result = {
            'success': False,
            'stages': {},
            'final_diagnosis': None,
            'processing_time': 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        validation = self.validator.validate(image_path)
        result['stages']['validation'] = validation
        
        if not validation['is_valid']:
            result['error'] = validation['reason']
            result['error_stage'] = 'validation'
            result['processing_time'] = time.time() - start_time
            
            return result
        
        
        if description and len(description.strip()) > 0:
            desc_info = self.extractor.extract(description)
            search_text = self.extractor.create_search_text(desc_info)
            result['stages']['description'] = desc_info
          
        else:
            search_text = None
            result['stages']['description'] = None
           
        match_result = self.matcher.match(image_path, search_text, top_k=3)
        result['stages']['matching'] = match_result
        
        top_defect = match_result['top_match']
        confidence = match_result['confidence']
       
        detected_defects = [top_defect]
        grading_result = self.grader.assign_grade(detected_defects)
        result['stages']['grading'] = grading_result
        
        
        if self.predictor and device_info:
            try:
                price_result = self.predictor.predict(
                    device_info,
                    detected_defects,
                    grading_result['condition_score']
                )
                result['stages']['pricing'] = price_result
                
            except Exception as e:
                print(f" Price prediction failed: {e}")
                result['stages']['pricing'] = None
        else:
            result['stages']['pricing'] = None
            if not device_info:
                print("ℹ No device info provided - skipping price prediction")
            else:
                print("ℹPrice predictor not available")
       
        result['success'] = True
        result['final_diagnosis'] = {
            'device_type': validation['device_type'],
            'validation_confidence': validation['confidence'],
            
            'defect': {
                'id': top_defect['id'],
                'name': top_defect['name'],
                'confidence': confidence,
                'severity': top_defect['severity_score'],
                'category': top_defect['category'],
                'critical': top_defect.get('critical', False),
                'symptoms': top_defect.get('symptoms', []),
                'affected_parts': top_defect.get('affected_parts', []),
                'recommendation': top_defect.get('recommendation', '')
            },
            
            'condition': {
                'grade': grading_result['grade'],
                'description': grading_result['description'],
                'score': grading_result['condition_score'],
                'breakdown': grading_result['breakdown']
            },
            
            'analysis_method': match_result['method'],
            'has_description': search_text is not None
        }
        
        if result['stages']['pricing']:
            result['final_diagnosis']['pricing'] = {
                'predicted_price': result['stages']['pricing']['predicted_price'],
                'price_range': result['stages']['pricing']['price_range'],
                'confidence': result['stages']['pricing']['confidence'],
                'depreciation_factors': result['stages']['pricing']['depreciation_factors']
            }
        
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def format_report(self, result):
        
        if not result['success']:
            return f""" diagnosis failed

Error: {result.get('error', 'Unknown error')}
Stage: {result.get('error_stage', 'Unknown')}
Time: {result['timestamp']}
"""
        
        diag = result['final_diagnosis']
        
        report = f"""

 Timestamp: {result['timestamp']}
 Processing Time: {result['processing_time']:.2f} seconds


 Device Type: {diag['device_type'].upper()}
 Validation Confidence: {diag['validation_confidence']:.1%}
 Analysis Method: {diag['analysis_method'].replace('_', ' ').title()}

 Issue: {diag['defect']['name']}
 Detection Confidence: {diag['defect']['confidence']:.1%}
  Severity: {diag['defect']['severity']}/10 {'CRITICAL' if diag['defect']['critical'] else ''}
 Category: {diag['defect']['category'].title()}

Affected Parts:
{chr(10).join(f'   • {part.title()}' for part in diag['defect']['affected_parts'])}

Symptoms:
{chr(10).join(f'   • {symptom.title()}' for symptom in diag['defect']['symptoms'][:5])}

━━━

Condition Grade: {diag['condition']['grade']}
 Description: {diag['condition']['description']}
 Condition Score: {diag['condition']['score']}/10

Breakdown:
   • Number of Defects: {diag['condition']['breakdown']['num_defects']}
   • Max Severity: {diag['condition']['breakdown']['max_severity']}/10
   • Critical Defect Present: {'Yes' if diag['condition']['breakdown']['has_critical'] else 'No'}
"""
        if 'pricing' in diag:
            pricing = diag['pricing']
            report += f"""

Estimated Resale Price: ₹{pricing['predicted_price']:,}
 Price Range: ₹{pricing['price_range']['min']:,} - ₹{pricing['price_range']['max']:,}
 Prediction Confidence: {pricing['confidence']:.0%}

 Price Impact Factors:
{chr(10).join(f"   • {f['factor']}: {f['impact']} - {f['description']}" for f in pricing['depreciation_factors'])}
"""
        
        report += f"""end of report
"""
        
        return report
    
    def save_result(self, result, output_path='results/diagnosis.json'):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\ Result saved to {output_path}")
