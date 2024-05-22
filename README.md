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
The following libraries were imported.

# Importing libraries.
- import pandas as pd
- import numpy as np
- import seaborn as sns
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler, OneHotEncoder
- from sklearn.pipeline import Pipeline
- from sklearn.compose import ColumnTransformer
- from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
- from sklearn.linear_model import LogisticRegression
- from sklearn.neighbors import KNeighborsClassifier
 -from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.metrics import classification_report, confusion_matrix
- from imblearn.over_sampling import RandomOverSampler
- from imblearn.over_sampling import SMOTE
- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
- from matplotlib import pyplot as plt
- %matplotlib inline

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
       # Class 0 (majority class): Precision (0.87): Out of all instances predicted as class 0, 87% are actually class 0. This indicates a relatively high percentage of true positives compared to false positives. Recall (0.98): 98% of actual class 0 instances are correctly identified by the model. This indicates a very high percentage of true positives compared to false negatives, suggesting the model is good at capturing instances of class 0. F1-score (0.92): The harmonic mean of precision and recall for class 0 is 0.92, indicating a good balance between precision and recall for this class. Support (566): There are 566 instances of class 0 in the dataset.
# Class 1 (minority class): Precision (0.55): Out of all instances predicted as class 1, only 55% are actually class 1. This suggests a higher number of false positives compared to true positives. Recall (0.16): Only 16% of actual class 1 instances are correctly identified by the model. This indicates a low percentage of true positives compared to false negatives, suggesting the model struggles to capture instances of class 1. F1-score (0.25): The harmonic mean of precision and recall for class 1 is 0.25, indicating a poor balance between precision and recall for this class. Support (101): There are 101 instances of class 1 in the dataset. The weighted average precision, recall, and F1-score are 0.82, 0.85, and 0.82 respectively. Weighted average takes into account class imbalance, giving more weight to the majority class.
- Hyperparameter Tuned Logistic Regression
  The tuned model slightly outperforms the untuned model in terms of ROC AUC score (0.8325 vs. 0.8297), indicating better overall performance in distinguishing between classes.
     # Model 3 : Random Forest classifier
     The outcome of the models was as follows;

    # churn (Class 1):

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
The outcome was as follows; 
# Non-Churn (Class 0):

Precision (0.91): Out of all instances predicted as non-churners, 91% are actually non-churners. This indicates a high percentage of true negatives compared to false positives.
Recall (0.99): 99% of actual non-churners are correctly identified by the model. This indicates a very high percentage of true positives compared to false negatives, suggesting the model is excellent at capturing instances of non-churn.
F1-score (0.95): The F1-score of 0.95 suggests a good balance between precision and recall for identifying non-churners.
Support (566): There are 566 instances of non-churners in the dataset.
# Churn (Class 1):

Precision (0.92): Out of all instances predicted as churners, 92% are actually churners. This indicates a high percentage of true positives compared to false positives.
Recall (0.45): Only 45% of actual churners are correctly identified by the model. This indicates a relatively low percentage of true positives compared to false negatives, suggesting the model struggles to capture instances of churn.
F1-score (0.60): The F1-score of 0.60 suggests a moderate balance between precision and recall for identifying churners.
Support (101): There are 101 instances of churn in the dataset.
The model performs exceptionally well in identifying non-churners, with high precision, recall, and F1-score. While the model also performs reasonably well in identifying churners, there is significant room for improvement in recall (identifying more of the actual churners).
The weighted average metrics suggest that the model's performance is good overall, with slightly better performance in identifying non-churn instances due to the larger support for that class.

# Tuned Hyperparameter for KNN
After tuning the model ; the performance improved as follows;
Comparing the two models, we can observe the following improvements in the tuned model:
The overall accuracy increased from 91.00% to 91.75%.
Precision for class 1 improved from 0.92 to 0.93, indicating better ability to correctly identify positive instances.
Recall for class 1 increased from 0.45 to 0.50, indicating that the tuned model is better at capturing positive instances.
F1-score for class 1 improved from 0.60 to 0.65, which balances precision and recall for class 1. Overall, the tuned model shows improvements across accuracy, precision, recall, and F1-score for class 1 compared to the untuned model.
# Model 5 ;  Supprt Vector Machine Analysis
Precision: Precision measures the proportion of true positive predictions out of all positive predictions made by the classifier. For churn (class 1), precision is 0.00, meaning that out of all instances predicted as churn, none were actually churn. For non-churn (class 0), precision is 0.85, indicating that 85% of the instances predicted as non-churn were actually non-churn.

Recall: Recall, also known as sensitivity or true positive rate, measures the proportion of true positive predictions out of all actual positive instances. For churn (class 1), recall is 0.00, which means that the classifier didn't correctly identify any of the actual churn instances. For non-churn (class 0), recall is 1.00, indicating that all actual non-churn instances were correctly identified by the classifier.

F1-score: The F1-score is the harmonic mean of precision and recall, providing a balance between them. It's a good measure of a model's accuracy. For churn (class 1), the F1-score is 0.00, indicating poor performance due to both precision and recall being low. For non-churn (class 0), the F1-score is 0.92, indicating a good balance between precision and recall.

Support: Support refers to the number of actual occurrences of each class in the dataset.

There are 566 instances of non-churn (class 0) and 101 instances of churn (class 1) in the dataset. Accuracy: Overall accuracy of the classifier.

The overall accuracy of the classifier is 85%, which is calculated as the proportion of correctly classified instances out of all instances. Macro Avg and Weighted Avg: These are the averages of precision, recall, and F1-score across all classes.

Macro average calculates metrics independently for each class and then takes the unweighted mean of the measures.

Weighted average calculates metrics for each class and finds their average weighted by support (the number of true instances for each label).

In summary, the classifier performs well in identifying non-churn instances (class 0) with high precision, recall, and F1-score, but it performs poorly in identifying churn instances (class 1), as indicated by the low precision, recall, and F1-score for class 1. This suggests that the classifier has difficulty correctly identifying churn instances.
 #  Tuned Hyperparameter for Support vector Machine
 The model has improved since the accuracy of the tuned increased.
 Comparison of all models.
 ![image](https://github.com/Essyvio/SYRIATEL-CUSTOMER-CHURN-DATA-ANALYSIS/assets/152212265/750f9fd7-95da-4a68-858b-39338a92e8ba)

# Logistic Regression:
Churn Prediction: The logistic regression model correctly identified some churn cases (recall: 15.84%), but it also misclassified many non-churn cases as churn (low precision: 55.17%). This indicates that while it could predict some churn cases, it also falsely flagged many loyal customers as potential churners. Non-Churn Prediction: The model achieved moderate accuracy in predicting non-churn cases, but it failed to distinguish them effectively from churn cases, leading to a low F1-score.

# Random Forest:
Churn Prediction: The random forest model demonstrated superior performance in identifying churn cases, achieving high recall (72.28%) and precision (92.41%). This suggests that it effectively identified a large portion of churners while minimizing false alarms. Non-Churn Prediction: The model maintained high accuracy and precision in predicting non-churn cases, indicating its ability to accurately identify loyal customers.

# KNN (K-Nearest Neighbors):
Churn Prediction: The KNN model exhibited moderate performance in identifying churn cases, with a relatively high precision (87.50%) but lower recall (34.65%). This indicates that while it correctly identified many churners, it also missed a significant portion of them. Non-Churn Prediction: The model achieved reasonable accuracy in predicting non-churn cases but had a lower F1-score compared to the random forest, suggesting a higher false positive rate.

# SVM (Support Vector Machine):
Churn Prediction: The SVM model performed poorly in churn prediction, with zero recall, precision, and F1-score for churn cases. It failed to identify any churners correctly. Non-Churn Prediction: Similar to logistic regression, the SVM model struggled to differentiate churn from non-churn cases, resulting in low accuracy and poor performance across all metrics. In summary, for churn prediction:

## Best Model: Random Forest exhibited the most balanced performance in identifying churn and non-churn cases, with high accuracy, recall, precision, and F1-score.
Worst Model: SVM failed to effectively predict churn, showing zero performance metrics for churn cases, indicating a lack of predictive power in this context.

# Comparing the ROC AUC Curve
![image](https://github.com/Essyvio/SYRIATEL-CUSTOMER-CHURN-DATA-ANALYSIS/assets/152212265/88a7e095-3437-4a5b-8fbf-326c8e2d1656)

My analysis also suggests that we can accurately predict customer churn using a machine learning model, with the Random Forest Classifier being the recommended model due to its strong overall performance. In terms of evaluating the impact of customer service calls, international plans, and day and night charges on customer churn, our feature importance values suggest that customer_service_calls is an important predictor of churn, while total_day_charge and total_day_minutes are also important. International plans do not appear to be among the most important features for predicting churn based on our analysis.

In terms of strategic recommendations for SyriaTel, we would suggest using the predictions from the Random Forest Classifier to identify customers who are at high risk of churning and proactively target them with retention efforts. These efforts could include personalized offers or discounts on day charges, improved customer service to address any issues or concerns, or changes to international plans to make them more attractive to customers. By implementing cost-effective strategies that address the key factors driving customer churn, SyriaTel can retain customers and minimize revenue loss.

Overall, our recommendation is for SyriaTel to use the Random Forest Classifier as its primary model for predicting customer churn and to take proactive measures to retain customers who are identified as being at high risk of churning. These measures should be informed by our analysis of the key predictors of customer churn and targeted towards addressing these factors in a cost-effective manner.

