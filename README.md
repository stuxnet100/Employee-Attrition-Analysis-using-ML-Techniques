# Employee-Attrition-Analysis-using-ML-Techniques-

## Breif Summary
**1.Preprocessed the data by Removing Skewness & performed dimensionality reduction using Principal Component Analysis**  

**2. Performed Data Augmentation techniques via. SMOTE to reduce models’ biases towards the majority class**

**3.Examined Support Vector Classifier, Logistic Regression, Random Forest, AdaBoost & Gradient Boosting on the dataset** 

**4.Compared the results based on various metrics, including Accuracy, F1 score, Confusion Matrix for validation**

**5.Examined Support Vector Classifier, Logistic Regression, Gaussian Naïve Bayes and Decision Tree classifiers on the dataset** 

**6.Improved performance of models using ensemble techniques i.e., AdaBoost, Gradient Boosting & Random Forest**


##  Project Description
This project's goal is to make predictions about whether or not an employee would resign using certain parameters. Before applying machine learning algorithms, data was analysed, preprocessed, and feature engineering was carried out. On the basis of accuracy, precision, recall, F1 and ROC scores, various ML algorithms are contrasted. These measures are used to choose the final model, and grid search and stratified k-fold cross validation are used to tweak its hyperparameters.

## Data
Data is taken from IBM HR Analytics Employee Attrition & Performance at kaggle https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/download
The downloaded file is present in Data folder. 

## Exploratory Data Analysis
1. To investigate the impact of characteristics on attrition, graphs of features vs. attrition are plotted.
2. A distribution analysis using a histogram of numerical features is performed.
3. The attrition distribution plot reveals extremely imbalanced data.**.

## Preprocessing and feature engineering
1. Categorical features are encoded using **one hot encoding scheme**.
2. Numerical features with **skewness** value greater than 0.8 are transformed using **log transformation.**
3. Three **new features** viz. tenure per job, years without change and compensation ratio were **created** using existing features.
4. **Irrelevant attributes** viz. EmployeeNumber, EmployeeCount, Over18 and StandardHours were **dropped**.
5. **One hot encoding** is used to transform cardinal categorical features.
6. **Label encoding** is used to transform ordinal categorical features and target labels.
7. Sybthetic Minority Oversampling Technique **SMOTE** is used to tackle data imbalance.
8. **Principal Component Analysis** is used for dimensionality reduction of data.

## ALgorithm tried
1. LogisticRegression
2. SVC
3. DecisionTreeClassifier 	    
4. RandomForestClassifier 	    
5. ExtraTreesClassifier 	      
6. GradientBoostingClassifier 	
7. AdaBoostClassifier 	        
8. GaussianNB 	

## Metrics used
Above models are compared on the basis of following metrics.
1. Accuracy
2. Precision
3. Sensitivity
4. F1
5. ROC

## Performance of various models


| Sr. No. |                      Model |  Accuracy | Precision | Sensitivity | Specificity | ROC Score |
--------:|---------------------------:|----------:|----------:|------------:|------------:|----------:|
|0|LogisticRegression|75.42372881355932|47.368421052631575|66.66666666666666|78.02197802197803|0.7234432234432233
|1|SVC|84.7457627118644|69.56521739130434|59.25925925925925|92.3076923076923|0.7578347578347578
|2|DecisionTreeClassifier|74.57627118644068|44.0|40.74074074074074|84.61538461538461|0.6267806267806268
|3|RandomForestClassifier|82.20338983050848|75.0|33.33333333333333|96.7032967032967|0.6501831501831501
|4|ExtraTreesClassifier|82.20338983050848|80.0|29.629629629629626|97.8021978021978|0.6371591371591372
|5|GradientBoostingClassifier|81.35593220338984|64.70588235294117|40.74074074074074|93.4065934065934|0.6707366707366708
|6|AdaBoostClassifier|88.13559322033898|84.21052631578947|59.25925925925925|96.7032967032967|0.7798127798127797
|7|GaussianNB|72.03389830508475|43.75|77.77777777777779|70.32967032967034|0.7405372405372405



Random forest can be employed as the ultimate model for predicting attrition, according to a comparison of models based on these metrics.
Following model completion, models were compared using "stratified k-fold cross validation" and "grid search" over a range of hyperparameters.
For the complete model, feature importance is plotted..

## Observations
1. Based on the importance of the features, it can be noted that our engineered features, such as "TenurePerJob," "CompRatioOverall," and "YearsWithoutChange," have greater importance values than many of the other features offered. Therefore, feature engineering was successful, it may be said.
2. SMOTE oversampling improved the accuracy of the model's predictions. Following the use of SMOTE, earlier models that were skewed towards forecasting the majority class alone no longer exist.
