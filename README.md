# AI Device Defect Diagnosis & Resale Price Estimation System

A production-grade, multi-modal AI system that automatically validates device images, detects physical defects, grades overall condition, and predicts resale price — deployed as a multi-user web application.

🚀 **Live Demo (Hugging Face Spaces)**
👉 [https://palakmathur-device-price-detector.hf.space](https://palakmathur-device-price-detector.hf.space)

---

## 🚩 Problem

Manual inspection of used devices (phones/laptops) for resale is:

- ❌ **Time-intensive:** 15–20 minutes per device
- ❌ **Expensive:** Requires trained inspectors
- ❌ **Inconsistent:** Subjective human judgment
- ❌ **Not scalable:** For large marketplaces

## ✅ Solution

This system fully automates device inspection using Computer Vision + NLP + Machine Learning.

Users can:

1. **Upload** a single device image
2. Optionally provide a text description (not required)
3. Instantly receive:
    - ✔️ Image validity check (device vs invalid image)
    - ✔️ Detected physical defect
    - ✔️ Condition grade (A–F)
    - ✔️ AI-predicted resale price with confidence range

⏱️ **End-to-end processing time:** ~8 seconds

---

## 🧠 High-Level Architecture

```
User Image (+ Optional Text)
        ↓
Image Domain Validation (CLIP)
        ↓
Text Processing (Optional)
        ↓
Defect Detection (Fine-Tuned CLIP)
        ↓
Condition Grading Engine
        ↓
ML Price Prediction (XGBoost)
        ↓
User-Readable Report + PDF
```

---

## 🔍 Key Features

- ✔️ **Image Validity Detection:** Automatically verifies whether the uploaded image is a valid device and rejects invalid inputs (people, animals, random objects), preventing garbage inference and improving system reliability.
- ✔️ **Optional Text Input:** The system works even without user text. If provided, the description improves defect detection accuracy. Designed for real users who may skip descriptions.
- ✔️ **Multi-Modal AI (Vision + Language):** Combines image understanding with semantic text understanding to improve robustness in ambiguous defect cases.
- ✔️ **Multi-User Safe Deployment:** Session-isolated uploads, temporary file isolation, cached model loading, and safe concurrent inference.

---

## 🏗️ Detailed Pipeline

### 1️⃣ Domain Validation (Image Gatekeeper)

Uses CLIP image–text similarity to confirm whether the image contains a smartphone or a laptop, rejecting invalid images early.

**Why it matters:** Prevents incorrect inference and protects downstream models.

### 2️⃣ Description Processing (Optional)

If a description is provided, the system extracts keywords (e.g., "crack," "screen," "battery") and affected components to create a search query that helps the model focus on the relevant parts of the image. If no text is provided, the system falls back to pure vision-based inference.

### 3️⃣ Defect Detection (Fine-Tuned CLIP – Zero-Shot)

Uses a fine-tuned CLIP model to match the uploaded image against textual defect descriptions like “cracked screen with broken glass,” “battery swelling,” and “physical dent on chassis.”

**Zero-shot approach:** New defect types can be added via text with no retraining required.

📈 **Accuracy improved from 14.7% → 68.9%** after fine-tuning.

### 4️⃣ Condition Grading (A–F)

Converts detected defects into a numerical score (0–10) and a letter grade (A–F), considering the severity, criticality, and number of defects. This ensures consistent and explainable grading, similar to refurbishing standards.

### 5️⃣ Resale Price Prediction

An ML regression model (XGBoost-based) that inputs the device brand & model, original price, device age, defect severity, and condition grade to output an estimated resale price, a min–max range, and a confidence score.

📊 **Model performance:**

- **R² = 0.87**
- **MAE ≈ ₹485**

---

## 🤖 Models & Storage

All trained models are stored and versioned on Hugging Face Hub, not inside the repository.

- **Fine-Tuned CLIP (Defect Detection):** [https://huggingface.co/palakmathur/device-defect-clip](https://huggingface.co/palakmathur/device-defect-clip)
- **Price Prediction Model:** [https://huggingface.co/palakmathur/device-defect-pricing](https://huggingface.co/palakmathur/device-defect-pricing)

✔️ Lightweight GitHub repo
✔️ Scalable deployment
✔️ Clean CI/CD-friendly design

---

## 🌐 Deployment

- **Platform:** Hugging Face Spaces
- **Framework:** Streamlit
- **Inference:** CPU (optimized, cached)
- **Concurrency:** Supported

🔗 **Live Application:** [https://huggingface.co/spaces/palakmathur/device_price_detector](https.huggingface.co/spaces/palakmathur/device_price_detector)

---

## 🧩 Tech Stack

- **Computer Vision:** CLIP (fine-tuned), PIL
- **NLP:** Text processing
- **Machine Learning:** XGBoost regression
- **Application:** Streamlit, Hugging Face Hub
- **Engineering:** Cached model loading, session-state isolation, temporary file safety, robust logging & error handling

---

## 📈 Metrics & Results

| Component                 | Result      |
| ------------------------- | ----------- |
| Defect Detection Accuracy | 68.96%      |
| Precision                 | 78.84%      |
| Recall                    | 68.96%      |
| Price Prediction R²       | 0.87        |
| Mean Absolute Error       | ₹485        |
| Avg Response Time         | ~8 sec      |

---

## 🧠 What This Project Demonstrates

- ✔️ Production-aware ML system design
- ✔️ Multi-modal AI (vision + language)
- ✔️ Transfer learning & fine-tuning
- ✔️ Zero-shot inference
- ✔️ Multi-user concurrency handling
- ✔️ Real-world business impact

---

## 🗣️ FAANG-Style One-Liner

Designed and deployed a multi-modal AI system using fine-tuned CLIP, BERT-based NLP, and ML regression to automatically validate device images, detect physical defects, grade condition, and predict resale price in a scalable multi-user web application.

---

## 📄 License

Apache 2.0
