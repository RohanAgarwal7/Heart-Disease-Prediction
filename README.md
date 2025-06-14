# Heart-Disease-Prediction
This notebook looks into building a machine learning model that is capable of predicting whether or not someone has heart disease based on their medical attributes. These medical attributes include age, sex, cholestrol etc. The problem this model aims to solve is that given clinical parameters about a patient, is it possible to predict whether or not the patient has heart disease. 

The Original data came from Cleavland data from the UCI Machine Learning Repository. There is also a version available from Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

## How it was made:
1. Initially, this project begins with Exploratory Data Analysis (EDA) to assess data type, identify missing values, detect outliers, and optimize feature selection to enhance data quality.
2. Building on EDA insights, we applied three models—Logistic Regression, KNN, and Random Forest—to the dataset, evaluating them based on accuracy. Logistic Regression emerged as the most effective baseline model, guiding further hyperparameter optimization.
3. Hyperparameter tuning, employing techniques like RandomizedSearchCV for Logistic Regression and Random Forest, refined model performance. The top-performing Logistic Regression model underwent additional fine-tuning using GridSearchCV.
4. The project resulted in evaluating model performance through metrics — precision, recall, and F1 score—using cross-validation.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-Learn

## Outcome
The project successfully builds and evaluates machine learning models capable of predicting heart disease, with a focus on improving healthcare decision-making through accurate, data-driven predictions.

This project was completed as part of the Zero-to-Mastery online AI and Machine Learning Data Science course.
