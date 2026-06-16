# Titanic Survival Prediction

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning.

## Dataset

Titanic dataset from Kaggle.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

## Machine Learning Pipeline

1. Data Cleaning
2. Missing Value Handling
3. Feature Encoding
4. Feature Scaling
5. Column Transformer
6. Random Forest Classifier
7. Model Evaluation

## Results

* Accuracy: 82.1%
* Confusion Matrix:

[[90, 15],
[17, 57]]

## Key Features

* Sex
* Age
* Fare
* Passenger Class

## Future Improvements

* Cross Validation
* Hyperparameter Tuning
* Feature Importance Analysis
* Streamlit Deployment

Feature Importance Analysis

The Random Forest model identified Fare and Age as the most influential numerical features for predicting survival. Gender was also a strong predictor, with the combined importance of Sex-related features exceeding 29% of the model's decision-making process. Passenger class contributed moderately, while embarkation port had minimal predictive value. These findings align with historical accounts indicating that wealth, age, and gender significantly influenced survival outcomes during the Titanic disaster.

| Model               | Accuracy | Cross Validation |
| ------------------- | -------- | ---------------- |
| Random Forest       | 83.80%   | 79.78%           |
| Logistic Regression | 81.56%   | 82.72%           |


## Author

Rohit
