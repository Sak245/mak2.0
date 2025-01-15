# README: 10-Year Risk of Coronary Heart Disease Prediction App

## Overview

This Streamlit app predicts the 10-year risk of coronary heart disease (CHD) based on user inputs related to health factors. It utilizes a pre-trained model to analyze the data and provide a risk assessment.

## Features

- **User Inputs**: Collects personal health data such as age, gender, smoking habits, blood pressure, cholesterol levels, and more.
- **Risk Prediction**: Predicts the likelihood of developing coronary heart disease in the next 10 years.
- **Result Interpretation**: Provides a message based on the predicted risk.

## Setup Instructions

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the pre-trained model (`model.h5`), scaler (`scaler.pkl`), and imputer (`imputer.pkl`) files are in the project directory.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How to Use

1. Input your personal health details in the provided fields.
2. Click the button to submit the data.
3. View the predicted probability of developing coronary heart disease within the next 10 years.

## Example Output

**Prediction**:  
- If the probability is greater than 0.5, the app advises taking precautions, as there is a higher chance of heart disease.
- If the probability is less than 0.5, the app indicates a lower risk.

---

Stay informed about your health and take preventive measures with this coronary heart disease risk prediction tool! 💖
