# credit-risk-classification

## Overview of the Analysis

The purpose of this project is to train and evaluate a model based on loan risk to evaluate the performance of machine learning model on predictions of loan statuses. Loans will be categorized into Healthy and High Risk Loans to identify the creditworthiness of borrowers. The model (logistic regression model) will identify healthy loans (0) and high risk loans(1) with the original data provided and resampled training data (training data that has been oversampled). We use a RandomOverSampler module from the imbalanced-learn library to resample the data. 

The data provided and used contains financial information including:
* loan size
* interest rate
* borrower income
* debt to income ratio
* number of accounts
* derogatory marks
* total debt
* loan status ( 0 = healthy loan,1 = high risk loans)

Machine Learning Process: 
* Data Loading
* Data Splittinginto Training and Testing Sets
* Model 1 - Logistic Regression Model with Original Data
* Model 2 - Logistic Regression Model with Resampled Training Data
* Evaluating Model Performance and Accuracy

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Confusion Matrix:
    * True Positive: 55973
    * False Positive: 304
    * True Negative: 1712
    * False Negative: 163
  * Balanced Accuracy Score: 0.954
  * Precision:
        * Healthy Loan `0` - 1.00
        * High Risk Loan `1` - 0.85
  * Recall:
        * Healthy Loan `0` - 0.99
        * High Risk Loan `1` - 0.91
  * f1-score: 
        * Healthy Loan `0` - 1.00
        * High Risk Loan `1` - 0.88


* Machine Learning Model 2:
  * * Confusion Matrix:
    * True Positive: 55946
    * False Positive: 331
    * True Negative: 1864
    * False Negative: 11
  * Balanced Accuracy Score: 0.994
  * Precision:
        * Healthy Loan `0` - 1.00
        * High Risk Loan `1` - 0.85
  * Recall:
        * Healthy Loan `0` - 0.99
        * High Risk Loan `1` - 0.99
  * f1-score: 
        * Healthy Loan `0` - 1.00
        * High Risk Loan `1` - 0.92

## Summary
The logistic regression model with oversampled data outperforms the original dataset model in predicting loan statuses, especially high-risk loans. The resampled model has a higher balanced accuracy score, recall score, and f1 score for high-risk loans. Since there is a significant class imbalance in the dataset, accurately predicting high-risk loans is crucial in determining a borrower's creditworthiness. False negatives are higher in the original dataset model, making the resampled model a better choice.

Through the dataset, we can see a disproportion in the number of healthy loans and high risk loans where healthy loans greatly outnumber high risk loans. Because of this, accurately predicting high risk loans is more important in determining a borrowers creditworthiness. Therefore, I would recommend the use of a logistic regression model. In this case, using it with the oversampled works better. 