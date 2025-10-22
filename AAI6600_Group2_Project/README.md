# Group 2 – What Type of Care Is Best for Me?

This project is part of the **AAI6600: Applied Artificial Intelligence** course at Northeastern University.  
It builds an intelligent recommendation system that analyzes a student's text input and predicts **which type of health care** may be most suitable for them.

---

## 🧠 Overview
The goal of this project is to support health service triage.  
Given a student's self-description (e.g., *"I've been feeling anxious and having trouble sleeping lately"*),  
the model recommends the **most relevant care category**, such as:
- Crisis Counseling  
- Skills Group  
- Specialist Support  
- Cultural Adjustment Counseling  
- General Wellness Coaching  

The system uses **sentence embeddings** from `sentence-transformers` and a **Logistic Regression** classifier to make predictions.

---

## 📁 Project Structure
```
Group2_Final_Demo/
│
├── Group2_Classification_Pipeline.py     # Main model pipeline script
├── training_data_embedding_1000.csv      # Training dataset (with embeddings)
├── test_data_embedding.csv               # Test dataset
├── requirements.txt                      # Dependencies
└── README.md                             # Project documentation
```

---

## ⚙️ Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
Run the following command in the terminal:
```bash
python health_classifier.py
```

Then enter a short text describing the student's situation, for example:
```
📝 Enter your scenario text: I’ve been having trouble focusing and feeling anxious.
```

The system will return:
```
Prediction Result:
{
    "input_text": "I’ve been having trouble focusing and feeling anxious.",
    "predicted_category": "Crisis counseling",
    "confidence": "28.93%",
    "top_recommendations": [
        {"category": "Crisis counseling", "confidence": "28.93%"},
        {"category": "Specialist", "confidence": "12.27%"},
        {"category": "Self-help", "confidence": "9.87%"}
    ]
}

```
Note: The training dataset is provided as a ZIP file (training_data_embedding_1000.zip) due to its large size.
Please unzip it before running the pipeline so that training_data_embedding_1000.csv is available.
---

## 🧩 Model Details
- **Embedding model:** all-MiniLM-L6-v2 (SentenceTransformers)  
- **Classifier:** Logistic Regression (default)  
- **Input:** Student text  
- **Output:** Recommended mental health care type + confidence score  
- **Feature dimension:** 1536  

---

## 👥 Team
**Group 2 – “What Type of Care Is Best for Me”**  

---

## 🗓️ Project Goal
To build a prototype that helps automatically recommend the most suitable care type based on a student’s text input.  