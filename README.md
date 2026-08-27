# E-Commerce Customer Churn Prediction

Machine learning model to predict customer churn for an e-commerce platform using historical customer behavior data.

## Overview

This project cleans and analyzes customer data from an e-commerce platform (5,630 customers, 20 features), then trains a Random Forest classifier to predict which customers are likely to churn based on their behavior, satisfaction, and order history.

## Dataset

E-Commerce Customer Churn dataset (Kaggle), containing customer-level features including tenure, order history, satisfaction scores, complaints, and payment preferences.

## Methods

- Cleaned missing values using median imputation
- One-hot encoded categorical features (payment method, order category, marital status, etc.)
- Split data into training (80%) and test (20%) sets, stratified by churn rate
- Trained a Random Forest classifier with balanced class weights to account for class imbalance (16.8% churn rate)
- Evaluated using precision, recall, F1-score, ROC-AUC, and a confusion matrix

## Results

- **ROC-AUC: 0.999**
- **Accuracy: 98%**
- **Recall on churned customers: 87%** (correctly identifies most at-risk customers)
- Top predictors of churn: customer tenure, cashback amount received, complaint history, and distance from warehouse to home

## Files

- `ecommerce_churn_analysis.ipynb` — Full analysis and modeling notebook
- `feature_importance.png` — Top 10 features driving churn predictions
- `confusion_matrix.png` — Model performance breakdown

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn, Google Colab
