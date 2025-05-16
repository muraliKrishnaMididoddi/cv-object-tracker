# Healthcare Risk Dashboard

This project is a full-stack, interactive data analytics application built using Streamlit. It allows users to explore healthcare data (such as the UCI Heart Disease dataset), apply filters, visualize data distributions, and extract insights related to cardiovascular health risks.

## Objective

The goal is to enable healthcare analysts, students, and data scientists to dynamically analyze medical data without needing to code. It supports user-uploaded data, URLs, and a local dataset to simulate real-world health monitoring and decision-making dashboards.

---

## Application Overview

Users can:

- Load heart disease data from:
  - File uploads
  - Public CSV URLs
  - A built-in local dataset
- Automatically assign column headers based on the UCI Heart Disease dataset
- Filter and analyze data using:
  - Sliders for numeric values (age, cholesterol, heart rate, etc.)
  - Dropdowns for categorical values (sex, chest pain type, etc.)
- Create visualizations:
  - Histograms
  - Boxplots
  - Barplots
  - Scatterplots
- Download the filtered dataset as a CSV

---

## Dataset Details

The primary dataset used is the **UCI Heart Disease (Cleveland subset)**:

**Features**:
- `age`: Age of the patient
- `sex`: 1 = male, 0 = female
- `cp`: Chest pain type (0–3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalach`: Max heart rate achieved
- `exang`: Exercise-induced angina
- `oldpeak`: ST depression
- `slope`: Slope of the peak ST segment
- `ca`: Major vessels colored by fluoroscopy
- `thal`: Thalassemia
- `target`: 0 = no disease, 1 = disease

---

## Model Used

Although the dashboard focuses primarily on **exploratory data analysis**, a backend Random Forest model (`train_model.py`) can be trained to predict heart disease risk using the above features.

**Model Type**:  
- `RandomForestClassifier` from scikit-learn

**Why Random Forest?**
- Handles mixed-type data well
- Robust to outliers and missing values
- Performs well on small-to-medium tabular datasets

**Evaluation Metrics**:
- Accuracy
- Precision/Recall
- Confusion Matrix

---

## Key Insights from Dashboard

Based on the visual analysis and filtering of the UCI dataset:

- Patients with **higher cholesterol and older age** showed higher risk profiles.
- **Male patients (sex = 1)** were more likely to be diagnosed with heart disease in this dataset.
- The `thal` and `cp` features showed strong correlation with the target, suggesting the importance of thalassemia and chest pain type in diagnosis.
- `ca` (number of major vessels) was a critical feature in predicting heart disease when visualized through boxplots and scatterplots.

These findings align with existing cardiovascular risk literature.

---

## Pros

- No coding required for end-user
- Compatible with custom datasets (healthcare or non-healthcare)
- Interactive filters and real-time visualizations
- Exportable insights for reporting or presentation
- Deployable to Streamlit Cloud, Heroku, or any VM

---

## Cons

- Only supports CSV format (no Excel/JSON)
- Does not include clinical diagnosis or validation
- Assumes clean and numerical data types
- Prediction module (ML model) is optional and not integrated into UI

---

## Technologies Used

| Component       | Tech Stack                  |
|----------------|-----------------------------|
| UI / Dashboard | Streamlit                   |
| Data Wrangling | Pandas                      |
| Visualization  | Matplotlib, Seaborn         |
| ML Model       | Scikit-learn (RandomForest) |
| Web Support    | Requests                    |
| Deployment     | Streamlit / GitHub          |

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-username/healthcare-risk-dashboard.git
cd healthcare-risk-dashboard

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run dashboard.py
