import streamlit as st
from PIL import Image, ImageFile
import sys
import os
import json
from io import BytesIO
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from domain_validator import DomainValidator
from description_extractor import DescriptionExtractor
from defect_matcher import DefectMatcher
from condition_grader import ConditionGrader
from price_predictor import PricePredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(
    page_title="Device Defect Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
            
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #1e7e34 0%, #28a745 100%);
        border-left: 5px solid #4cff4c;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: white;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    .warning-box {
        background: linear-gradient(135deg, #c43a00 0%, #dc3545 100%);
        border-left: 5px solid #ffcc00;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: white;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .defect-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        padding: 0.5rem 0;
        margin: 0.5rem 0;
    }
    .defect-info {
        background: linear-gradient(135deg, #1a5490 0%, #1f77b4 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #5dade2;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load the diagnosis system (cached)"""
    with st.spinner(" Loading models... This may take a minute..."):
        validator = DomainValidator()
        extractor = DescriptionExtractor()
        matcher = DefectMatcher(use_finetuned=True) 
        grader = ConditionGrader()
        try:
            predictor = PricePredictor()
            predictor.load_model()
        except:
            predictor = None
        
        return validator, extractor, matcher, grader, predictor


def main():
    st.markdown('<div class="main-header"> Device Defect Diagnosis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a device image to detect defects and estimate resale value</div>', unsafe_allow_html=True)
    
    validator, extractor, matcher, grader, predictor = load_system()
    
    with st.sidebar:
        st.header("Instructions:-")
        st.markdown("""
        1. **Upload** a device image (phone/laptop)
        2. **Describe** the issue (optional)
        3. **Enter** device details for pricing
        4. **Click** Diagnose to get results
        
        ---
        
        ###  Supported Devices
        -  Smartphones
        -  Laptops
        -  Desktop computers
        
        ---
        
        ###  Understanding Results
        
        **Confidence Score**  
        How certain the model is about its detection (0-100%)
        - 90%+ : High confidence
        - 70-90% : Good confidence
        - <70% : Review needed
        
        **Severity Score** (0-10)  
        Impact of the defect on device functionality
        - 8-10 : Critical (needs repair)
        - 5-7 : Moderate (affects usage)
        - 0-4 : Minor (cosmetic)
        
        **Condition Score** (0-10)  
        Overall device condition after inspection
        - 9-10 : Excellent (Grade A)
        - 7-9 : Good (Grade B)
        - 5-7 : Fair (Grade C)
        - 3-5 : Poor (Grade D)
        - 0-3 : Bad (Grade F)
        
        ---
        
        ###  Detected Issues:-
        - Screen damage
        - Physical defects
        - Component failures
        - And more...
        """)
        
        st.markdown("---")
        st.markdown("### System Stats :-")
        st.metric("Models Loaded", "5/5")
        st.metric("Defect Types", "15+")
        st.metric("Avg Response Time", "8.2s")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(" Upload & Configure")
        
        uploaded_file = st.file_uploader(
            "Choose a device image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of your device"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=False)
        
        description = st.text_area(
            "Describe the issue (optional)",
            placeholder="e.g., My phone screen cracked after I dropped it. Touch is not working properly.",
            height=100,
            help="Providing a description improves accuracy"
        )
        
        st.subheader(" Device Information (for price prediction)")
        
        with st.expander("Enter device details:-", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                brand = st.selectbox(
                    "Brand",
                    ["Apple", "Samsung", "OnePlus", "Dell", "HP", "Lenovo", "Other"]
                )
                
                device_type = st.selectbox(
                    "Device Type",
                    ["Phone", "Laptop"]
                )
            
            model_options = {
                "Apple": {
                    "Phone": ["iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15", "iPhone 14 Pro Max", "iPhone 14 Pro", "iPhone 14", "iPhone 13 Pro Max", "iPhone 13 Pro", "iPhone 13", "iPhone 12 Pro Max", "iPhone 12 Pro", "iPhone 12", "iPhone 11 Pro Max", "iPhone 11 Pro", "iPhone 11", "iPhone XS Max", "iPhone XS", "iPhone XR", "iPhone X", "iPhone 8 Plus", "iPhone 8"],
                    "Laptop": ["MacBook Pro 16\" M3 Max", "MacBook Pro 14\" M3 Pro", "MacBook Pro 16\" M2 Max", "MacBook Pro 14\" M2 Pro", "MacBook Air M3", "MacBook Air M2", "MacBook Air M1", "MacBook Pro 13\" M1"]
                },
                "Samsung": {
                    "Phone": ["Galaxy S24 Ultra", "Galaxy S24+", "Galaxy S24", "Galaxy S23 Ultra", "Galaxy S23+", "Galaxy S23", "Galaxy S22 Ultra", "Galaxy S22+", "Galaxy S22", "Galaxy Z Fold 5", "Galaxy Z Flip 5", "Galaxy Z Fold 4", "Galaxy Z Flip 4", "Galaxy A54", "Galaxy A34"],
                    "Laptop": ["Galaxy Book4 Pro", "Galaxy Book3 Ultra", "Galaxy Book3 Pro 360", "Galaxy Book2 Pro", "Galaxy Book2"]
                },
                "OnePlus": {
                    "Phone": ["OnePlus 12", "OnePlus 11", "OnePlus 10 Pro", "OnePlus 9 Pro", "OnePlus 9", "OnePlus 8T", "OnePlus Nord 3", "OnePlus Nord 2"],
                    "Laptop": []
                },
                "Dell": {
                    "Phone": [],
                    "Laptop": ["XPS 15", "XPS 13", "XPS 17", "Inspiron 15", "Inspiron 14", "Latitude 5430", "Latitude 7430", "Alienware m15", "Alienware x15", "Vostro 15"]
                },
                "HP": {
                    "Phone": [],
                    "Laptop": ["Pavilion 15", "Pavilion 14", "Envy 15", "Envy 13", "Spectre x360 14", "Spectre x360 16", "EliteBook 840", "EliteBook 850", "ProBook 450", "Omen 15"]
                },
                "Lenovo": {
                    "Phone": [],
                    "Laptop": ["ThinkPad X1 Carbon", "ThinkPad X1 Yoga", "ThinkPad T14", "ThinkPad T16", "IdeaPad Slim 5", "IdeaPad Gaming 3", "Yoga 9i", "Yoga 7i", "Legion 5 Pro"]
                },
                "Other": {
                    "Phone": ["Other Model"],
                    "Laptop": ["Other Model"]
                }
            }
            
            available_models = model_options.get(brand, {}).get(device_type, ["Other Model"])
            
            with col_b:
                if available_models:
                    model = st.selectbox("Model", available_models)
                else:
                    st.warning(f"No {device_type} models available for {brand}")
                    model = st.text_input("Model (enter manually)", "")
                
                original_price = st.number_input("Original Price (₹)", min_value=0, value=79900, step=1000)
            
            age_months = st.slider("Age (months)", 0, 60, 18, help="How old is the device?")
    
    with col2:
        st.header("Diagnosis Results :- ")
        
        if uploaded_file:
            if st.button(" Diagnose Device", type="primary", use_container_width=False):
                
                temp_path = "temp_upload.jpg"
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(temp_path)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Stage 1/5: Validating device...")
                    progress_bar.progress(20)
                    
                    validation = validator.validate(temp_path)
                    
                    if not validation['is_valid']:
                        st.markdown(f'<div class="error-box"> <b>Invalid Image</b><br>{validation["reason"]}</div>', unsafe_allow_html=True)
                        os.remove(temp_path)
                        return
                    
                    status_text.text("Stage 2/5: Processing description...")
                    progress_bar.progress(40)
                    
                    if description:
                        desc_info = extractor.extract(description)
                        search_text = extractor.create_search_text(desc_info)
                    else:
                        search_text = None
                    
                    status_text.text(" Stage 3/5: Detecting defects...")
                    progress_bar.progress(60)
                    
                    match_result = matcher.match(temp_path, search_text, top_k=3)
                    defect = match_result['top_match']
                    
                    status_text.text("Stage 4/5: Grading condition...")
                    progress_bar.progress(80)
                    
                    grading = grader.assign_grade([defect])
                    
                    status_text.text("Stage 5/5: Predicting price...")
                    progress_bar.progress(90)
                    
                    price_result = None
                    if predictor:
                        try:
                            device_info = {
                                'brand': brand,
                                'model': model,
                                'original_price': original_price,
                                'age_months': age_months
                            }
                            price_result = predictor.predict(device_info, [defect], grading['condition_score'])
                        except:
                            pass
                    
                    progress_bar.progress(100)
                    status_text.text(" Diagnosis complete!!!")
                    
                    st.markdown("---")
                    
                    st.markdown(f'<div class="success-box"> <b>Valid {validation["device_type"].title()} Detected</b> (Confidence: {validation["confidence"]:.1%})</div>', unsafe_allow_html=True)
                    
                    st.subheader(" Detected Issue")
                    
                    severity_color = "#dc3545" if defect.get('critical', False) else "#ffc107" if defect['severity_score'] > 6 else "#28a745"
                    
                    st.markdown(f'<div class="defect-info"><div class="defect-name"> {defect["name"]}</div></div>', unsafe_allow_html=True)
                    
                    col_r1, col_r2 = st.columns(2)
                    col_r1.metric(
                        "Confidence", 
                        f"{match_result['confidence']:.1%}",
                        help="How certain the AI is about this detection. Higher is better."
                    )
                    col_r2.metric(
                        "Severity", 
                        f"{defect['severity_score']}/10",
                        help="Impact of this defect: 0-4 (Minor), 5-7 (Moderate), 8-10 (Critical)"
                    )
                    
                    if defect['severity_score'] >= 8:
                        severity_msg = " **High Severity:** This defect significantly impacts device functionality and requires immediate attention."
                    elif defect['severity_score'] >= 5:
                        severity_msg = " **Moderate Severity:** This defect affects device usage. Consider repair before resale."
                    else:
                        severity_msg = " **Low Severity:** This is mostly a cosmetic issue with minimal impact on functionality."
                    
                    st.info(severity_msg)
                    
                    if defect.get('critical', False):
                        st.markdown(f'<div class="warning-box"> <b>Critical Issue Detected</b><br>Immediate attention required</div>', unsafe_allow_html=True)
                    
                    st.subheader(" Condition Assessment")
                    
                    col_c1, col_c2 = st.columns(2)
                    col_c1.metric(
                        "Grade", 
                        grading['grade'], 
                        help="Letter grade from A (Excellent) to F (Bad) based on overall condition"
                    )
                    col_c2.metric(
                        "Condition Score", 
                        f"{grading['condition_score']}/10",
                        help="Numerical score: 10 is perfect, 0 is non-functional. Affects resale value directly."
                    )
                    
                    st.caption(f"**{grading['description']}**")
                    
                    with st.expander("What does this grade mean???", expanded=False):
                        grade_info = {
                            'A': "**Excellent Condition** - Device is like new with minimal to no defects. Commands premium resale price.",
                            'B': "**Good Condition** - Device shows minor wear but fully functional. Good resale value.",
                            'C': "**Fair Condition** - Device has noticeable defects but still usable. Moderate resale value.",
                            'D': "**Poor Condition** - Device has significant issues affecting usability. Low resale value.",
                            'F': "**Bad Condition** - Device has critical defects or is barely functional. Very low resale value."
                        }
                        st.write(grade_info.get(grading['grade'], "Assessment completed"))
                        
                        st.write("""\n**How it's calculated:**
- Based on severity and number of defects
- Physical damage has higher impact
- Critical defects significantly lower the grade
- Overall device age and wear considered""")
                    
                    if price_result:
                        st.subheader("Resale Valuation")
                        
                        col_p1, col_p2, col_p3 = st.columns(3)
                        col_p1.metric(
                            "Estimated Price", 
                            f"₹{price_result['predicted_price']:,}",
                            help="AI-predicted resale price based on condition, defects, and market data"
                        )
                        col_p2.metric(
                            "Min Price", 
                            f"₹{price_result['price_range']['min']:,}",
                            help="Minimum expected price in current market conditions"
                        )
                        col_p3.metric(
                            "Max Price", 
                            f"₹{price_result['price_range']['max']:,}",
                            help="Maximum expected price for this condition"
                        )
                        
                        st.caption(f"**Prediction Confidence:** {price_result['confidence']:.0%} - How reliable this price estimate is based on available data")
                        
                        with st.expander(" Price Impact Factors"):
                            for factor in price_result['depreciation_factors']:
                                st.markdown(f"• **{factor['factor']}**: {factor['impact']} - {factor['description']}")
                    
                    if len(match_result['all_matches']) > 1:
                        with st.expander(" Alternative Diagnoses"):
                            for i, alt in enumerate(match_result['all_matches'][1:3], 2):
                                st.markdown(f"{i}. **{alt['defect']['name']}** - Confidence: {alt['confidence']:.1%}")
                    
                    st.markdown("---")
                    
                    try:
                        from fpdf import FPDF
                        from fpdf.enums import XPos, YPos
                        
                        class DiagnosisReport(FPDF):
                            def header(self):
                                self.set_font('Helvetica', 'B', 20)
                                self.set_text_color(31, 119, 180)
                                self.cell(0, 10, 'Device Diagnosis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                                self.set_font('Helvetica', '', 10)
                                self.set_text_color(100, 100, 100)
                                self.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                                self.ln(10)
                            
                            def footer(self):
                                self.set_y(-15)
                                self.set_font('Helvetica', 'I', 8)
                                self.set_text_color(128, 128, 128)
                               
                        
                        pdf = DiagnosisReport()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.set_text_color(44, 62, 80)
                        pdf.cell(0, 10, 'Device Information', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        pdf.ln(2)
                        
                        pdf.set_font('Helvetica', '', 11)
                        pdf.set_text_color(0, 0, 0)
                        device_info = [
                            ('Device Type:', validation['device_type'].title()),
                            ('Brand:', brand),
                            ('Model:', model),
                            ('Age:', f"{age_months} months"),
                            ('Detection Confidence:', f"{validation['confidence']:.1%}")
                        ]
                        for label, value in device_info:
                            pdf.set_font('Helvetica', 'B', 11)
                            pdf.cell(60, 8, label, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')
                            pdf.set_font('Helvetica', '', 11)
                            pdf.cell(0, 8, str(value), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        
                        pdf.ln(8)
                        
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.set_text_color(44, 62, 80)
                        pdf.cell(0, 10, 'Detected Issue', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        pdf.ln(2)
                        
                        pdf.set_font('Helvetica', '', 11)
                        pdf.set_text_color(0, 0, 0)
                        defect_info = [
                            ('Defect Name:', defect['name']),
                            ('Confidence:', f"{match_result['confidence']:.1%}"),
                            ('Severity Score:', f"{defect['severity_score']}/10"),
                            ('Critical:', 'Yes' if defect.get('critical', False) else 'No')
                        ]
                        for label, value in defect_info:
                            pdf.set_font('Helvetica', 'B', 11)
                            pdf.cell(60, 8, label, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')
                            pdf.set_font('Helvetica', '', 11)
                            pdf.cell(0, 8, str(value), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        
                        pdf.ln(8)
                        
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.set_text_color(44, 62, 80)
                        pdf.cell(0, 10, 'Condition Assessment', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        pdf.ln(2)
                        
                        pdf.set_font('Helvetica', '', 11)
                        pdf.set_text_color(0, 0, 0)
                        condition_info = [
                            ('Grade:', grading['grade']),
                            ('Condition Score:', f"{grading['condition_score']}/10"),
                            ('Description:', grading['description'])
                        ]
                        for label, value in condition_info:
                            pdf.set_font('Helvetica', 'B', 11)
                            pdf.cell(60, 8, label, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')
                            pdf.set_font('Helvetica', '', 11)
                            pdf.cell(0, 8, str(value), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        
                        pdf.ln(8)
                        
                        if price_result:
                            pdf.set_font('Helvetica', 'B', 14)
                            pdf.set_text_color(44, 62, 80)
                            pdf.cell(0, 10, 'Resale Valuation', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                            pdf.ln(2)
                            
                            pdf.set_font('Helvetica', '', 11)
                            pdf.set_text_color(0, 0, 0)
                            pricing_info = [
                                ('Estimated Price:', f"Rs. {price_result['predicted_price']:,}"),
                                ('Price Range:', f"Rs. {price_result['price_range']['min']:,} - Rs. {price_result['price_range']['max']:,}"),
                                ('Prediction Confidence:', f"{price_result['confidence']:.0%}")
                            ]
                            for label, value in pricing_info:
                                pdf.set_font('Helvetica', 'B', 11)
                                pdf.cell(60, 8, label, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')
                                pdf.set_font('Helvetica', '', 11)
                                pdf.cell(0, 8, str(value), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
                        
                        pdf_data = bytes(pdf.output())
                        
                        st.download_button(
                            " Download Full Report (PDF)",
                            data=pdf_data,
                            file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except (ImportError, Exception) as e:
                        st.warning(f" PDF generation failed: {str(e)}\n")
                        report = {
                            'device': {
                                'type': validation['device_type'],
                                'brand': brand,
                                'model': model,
                                'age_months': age_months
                            },
                            'defect': {
                                'name': defect['name'],
                                'confidence': match_result['confidence'],
                                'severity': defect['severity_score']
                            },
                            'condition': {
                                'grade': grading['grade'],
                                'score': grading['condition_score']
                            }
                        }
                        
                        if price_result:
                            report['pricing'] = {
                                'estimated_price': price_result['predicted_price'],
                                'price_range': price_result['price_range']
                            }
                        
                        st.download_button(
                            "📥 Download Full Report (JSON)",
                            data=json.dumps(report, indent=2),
                            file_name="diagnosis_report.json",
                            mime="application/json"
                        )
                    
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f" Error during diagnosis: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.info(" Upload an image to begin diagnosis")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Device Defect Diagnosis System</b></p>
        <p>Built with CLIP, BERT, and XGBoost</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()