# Heart-Disease-Prediction

This notebook looks into building a machine learning model that is capable of predicting whether or not someone has heart disease based on their medical attributes. These medical attributes include age, sex, cholestrol etc. The problem this model aims to solve is that given clinical parameters about a patient, is it possible to predict whether or not the patient has heart disease. 

The Original data came from Cleavland data from the UCI Machine Learning Repository. There is also a version available from Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

## How it was made:
1. Initially this project conducts an Exploratory Data Analysis (EDA) to determine the type of data we have, whether there was any missing data, were there any outliers and how can we add or remove features to get more out fo our data
2. Based on the findings from the EDA, we started to model our data. For this dataset, 3 models were chosen: Logisitic Regression, KNN and Random Forest. Once these models were fitted they were compared based on accuracy. From this, logistic regression proved the be the best. We now had 3 baseline models to continue with hyperparameter tuning.
3. In order to tune and improve our model, methods such as hyperparameter tuning were used. Hyperparameter tuning of the logisitic regression model and random forest model were conducted with RandomizedSearchCV(). Since the LogisticRegression model provided the best scores the model was tunes again using GridSearchCV().
4. Finally, precision, recall, f1 score were calculated using cross_val_score.

This project was conducted through zero-to-mastery online complete AI and Machine Learning Data Science course
