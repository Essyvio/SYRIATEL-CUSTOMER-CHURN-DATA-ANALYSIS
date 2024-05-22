# SYRIATEL-CUSTOMER-CHURN-DATA-ANALYSIS

Author:
Violet Musyoka

# Project overview
This project employs machine learning algorithms to construct a model capable of accurately forecasting customer churn, leveraging data within the dataset. The dataset comprises 21 predictor variables, predominantly pertaining to customer usage patterns. The focal variable of interest is 'churn', which denotes whether a customer has churned or not. Given the categorical nature of the target variable, classification algorithms are employed to develop the predictive model. Recall serves as the metric for assessing the model's performance.This project analyzes SyriaTel data and builds models that predict whether a customer will churn or not.

# Business Understanding
The foundation of every business is its customers. They determine the value of products or services a business offers.Business depend on customers, failure to that no business exists hence businesses strive hard to get new customers as well as retained the existing ones.Understanding and managing customer churn is vital for the success of any business, as customers are the backbone of its operations and revenue generation. For SyriaTel, a telecommunications company facing a notable churn rate among its customer base, addressing this issue is critical for maintaining its market position and profitability.

By leveraging a predictive model, SyriaTel aims to forecast which customers are likely to terminate their services ("churn") in the near future. This predictive capability enables proactive measures to be taken, such as targeted retention strategies or personalized offers, to mitigate churn and preserve customer loyalty.
# Objectives:
- To find out the features thats are most important to our target variable

- To come up with a predictive model that predicts whether a customer will churn soon

- Come up with recommendations for customers predicted to churn.
  # Stakeholders
- SyriaTel company: A telecommunication company in Syria that is looking for help in increasing its profits by retaining its customers. That is, reducing the rate of customer churn. They are also in need of a model that predicts whether a customer (new or existing) will churn soon.

- Employees

- Customer
# Data Understanding:
The dataset obtained from Kaggle contains 3333 entries and 21 columns, including information about the state, account length, area code, phone number, international plan, voice mail plan, number of voice mail messages, total day minutes, total day calls, total day charge, total evening minutes, total evening calls, total evening charge, total night minutes, total night calls, total night charge, total international minutes, total international calls, total international charge, customer service calls and churn.
# Problem statement
This is a binary classification problem where the target variable is whether a customer churns or not.SyriaTel is facing an increasing churn rate,resulting to a comprehensive analysis to highlight the factors contributing to the customer attrition.The business context is focused on reducing losses incurred due to customer churn. By accurately identifying customers at risk of leaving, SyriaTel can implement targeted retention strategies to minimize churn and maximize customer lifetime value.The key question driving this project is whether there are predictable patterns in customer behavior that signal an increased likelihood of churn. By leveraging historical data on customer interactions, demographics, usage patterns, and churn status, the goal is to build a model that can effectively distinguish between customers who are likely to churn and those who are not.
# Data Preparation
The analysis  performed on the datasetincluded the following steps;Data cleaning, The dataset was checked per column to find the missing values,there were no duplicates & the outliers were handled accordingly using box plots.Data transformation; Categorical data in the churn column was converted into numerical data. Exploratory  Data Analysis; Univariate, Bivariate & Multivariate was done to identify possible correlations between features .The label encoding was used before generating the heat map to address the correlation issue  between the different attributes.Feature engineering involves creating new features from existing ones to improve model performance or extract more meaningful information from the data. In this case, i have added two columns ('Total Charge' and 'Total Minutes') by combining existing columns related to charges and minutes in different periods of the day.
 All these steps were done to gain a comprehensive understanding of the dataset and prepare it for further analysis.

 # Modeling
 Several classification models were used to generate reports.

 - # Model 1: Baseline model
   The baseline model was done using a dummy  classifier .The baseline model's outcome on precision, recall, and F1-score being zero while the accuracy is high indicated a severe class imbalance issue in the dataset.Based on these metrics and the confusion matrix, it appeared that the model was not performing well. Hence a need to investigate why the model was making such predictions and consider potential improvements such as feature engineering, model selection, hyperparameter tuning, or addressing class imbalance issues. In this case have choosen to address issue using the class imbalance.
   - Class imbalance  was performed using linear regression to address the issue of imbalance.
     In summary, while the Dummy Classifier achieved a higher accuracy, it fails to capture the minority class, leading to poor precision, recall, and F1 score. The model after addressing imbalance provided a more balanced performance, achieving a reasonable trade-off between precision and recall, resulting in a higher F1 score and better discrimination between classes, as evidenced by the higher ROC AUC.
     - # Model 2: Logistic Regression.
       The findings was as follows.
       Class 0 (majority class): Precision (0.87): Out of all instances predicted as class 0, 87% are actually class 0. This indicates a relatively high percentage of true positives compared to false positives. Recall (0.98): 98% of actual class 0 instances are correctly identified by the model. This indicates a very high percentage of true positives compared to false negatives, suggesting the model is good at capturing instances of class 0. F1-score (0.92): The harmonic mean of precision and recall for class 0 is 0.92, indicating a good balance between precision and recall for this class. Support (566): There are 566 instances of class 0 in the dataset.
Class 1 (minority class): Precision (0.55): Out of all instances predicted as class 1, only 55% are actually class 1. This suggests a higher number of false positives compared to true positives. Recall (0.16): Only 16% of actual class 1 instances are correctly identified by the model. This indicates a low percentage of true positives compared to false negatives, suggesting the model struggles to capture instances of class 1. F1-score (0.25): The harmonic mean of precision and recall for class 1 is 0.25, indicating a poor balance between precision and recall for this class. Support (101): There are 101 instances of class 1 in the dataset. The weighted average precision, recall, and F1-score are 0.82, 0.85, and 0.82 respectively. Weighted average takes into account class imbalance, giving more weight to the majority class.
-    Hyperparameter Tuned Logistic Regression
  The tuned model slightly outperforms the untuned model in terms of ROC AUC score (0.8325 vs. 0.8297), indicating better overall performance in distinguishing between classes.
       # Model 3 : Random Forest classifier
     The outcome of the models was as follows;
     Churn (Class 1):

precision (1.00): Out of all customers predicted as churners, 100% are actually churners. This indicates a perfect identification of customers who are likely to churn.
Recall (0.87): 87% of actual churners are correctly identified by the model. This suggests that while the model is good at capturing a majority of churners, there are still some churners it misses(13%).
F1-score (0.93): The F1-score of 0.93 suggests a good balance between precision and recall for identifying churners.
Support (101): There are 101 instances of churn in the dataset.
Non-Churn (Class 0):

Precision (0.98): Out of all customers predicted as non-churners, 98% are actually non-churners. This indicates a very high percentage of true negatives compared to false positives.

Recall (1.00): 100% of actual non-churners are correctly identified by the model. This indicates a perfect capture of customers who are not likely to churn.

F1-score (0.99): The F1-score of 0.99 suggests an excellent balance between precision and recall for identifying non-churners.

Support (566): There are 566 instances of non-churners in the dataset.

The model performs exceptionally well in identifying non-churners, with a very high precision, recall, and F1-score.The model performs well but we can make improvement in recall by identifying more of the actual churners.
# Tuned Hyperparameter for Random Forest classifier
- Comparison between The tuned & Untuned was as follows:
Both models achieve high ROC AUC scores, indicating strong performance in distinguishing between positive and negative instances. The untuned model has a slightly higher ROC AUC score (0.928) compared to the tuned model (0.925), suggesting that the untuned model performs marginally better in this aspect. However, the difference is relatively small and might not be practically significant.
- Looking at the confusion matrices, both models correctly classify the majority of instances. Both models have the same number of true negatives (560) and false positives (6). However, there are slight differences in the number of false negatives and true positives between the two models. The untuned model has 28 false negatives and 73 true positives, while the tuned model has 31 false negatives and 70 true positives.
The untuned model appears to have a slightly better balance between true positives and false negatives compared to the tuned model. This is evident from the higher number of true positives and lower number of false negatives in the confusion matrix of the untuned model.

#Model 4;  K-NeighborsClassifier
