# Credit_Risk_Analysis

## Overview of Project

### Purpose of the Project

Our primary customer for this project is a peer to peer lending company called LendingClub who wants to use Machine Learning to to predict credit risk with the hopes that it will result in a quicker and more reliable loan experience.  Additionally, they believe that Machine Learning will increase the accuracy of loan decisions, thereby decreasing loan default rates.  They have tasked us to work with Jill, their lead data scientist, to use multiple Machine Learning algorithms and approaches to predict credit risk, and to assess the best algorithm to use.  For this project in particular, Jill wants us to imbalance-learn from SciKit Learn’s libraries to build and evaluate four methodologies for resampling: Random Over Sample and SMOTE for oversampling, Cluster Centroid for under sampling, and SMOTEEN for a combination of over and under sampling.  In addition to sampling, Jill wants us to look at two ensemble Machine Learning approaches to reduce bias:  Balanced Random Forest Classifier and Easy Ensemble Classifier.  After conducting all six different approaches, Jill has asked for us to make a recommendation on which, if any, algorithm LendingClub should use to predict which candidates will be the best for loans.  


### Project Breakdown

To provide Jill and LendingClub the analysis they requested, we used a credit card data set from LendingClub called [LoanStats_2019Q1](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/Challenge.zip) located within the Zip file in this GitHub section. LoanStats_2019Q1 has 85 different columns, most of which are integers and only a handful of which are string values.  The target value for the analysis is loan_status, which breaks out individual applicants into two categories: high risk or low risk.  The remaining columns are features, providing inputs to the algorithms that we will use to predict an applicants loan status.  

For each of the six models, we used a combination of Python, Pandas, Pandas, and SciKit Learn to not only clean but also split the data into test and training sets.  First, to clean the data, we needed to read in the CSV, drop null values, and then conduct some cleaning of the target column as well to decrease the number of options from 6 to only two (remove ‘Issued’; turn ‘Current’ into ‘low_risk’; and then turn 'Late (31-120 days)', 'Late (16-30 days)', 'Default', and’ In Grace Period' into ‘high_risk’).  Once we completed that, we could then officially name the features (X) and target (y) value with the following code.  Of note, because several feature columns were strings and not integers, we used the ‘get_dummies()’ feature in Pandas to turn those columns from strings into integers, using binary encoding.  This results in a greater number of columns than the original as each ‘string’ option within that column gets broken out into their column. 


```
# Create our features
X = df.copy()
X = X.drop("loan_status", axis=1)
X = pd.get_dummies(X)

# Create our target
y = df["loan_status"]
```

To ensure that our X and y values were balanced, we used X.describe() and y.values_count().  This gave us the break out of our X features and also the official break out of our y values.  While the features looked balanced, the target variables did not: high_risk was only 347 as opposed to low_risk which was 68470.  This highlighted the need for rebalancing and a sampling technique prior to conducting the predictive modeling using linear regression.  


![Feature Balance](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/Balanced_1.png)

![Target Balance](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/Balanced_2.png)


Finally, we needed to create our test and training set using SciKit Learn’s train_test_split function.  We used the following code with an assumed 75:25 split train to test, and resulted in the following count for the training target variable: low_risk - 51366, high_risk - 246.
 
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
Counter(y_train)
```


## Results

As identified above, we employed six different Machine Learning algorithms: 4 focused on correcting the sampling imbalance for the target variable and two to deal with the potential bias issue.  Below outlines the approach for each of the six algorithms and a description of their associated accuracy, precision, and recall scores.


### Sampling + Logistic Regression

With the knowledge that there is a distinct imbalance in the target values, we need to identify the best approach to correcting for that imbalance.  We conducted four different tests of sampling approaches, and then looked for the output of both after using the sampling data as an input to building a linear regression model.  

- ***Naive Random Oversampling***
Using SciKit Learn’s imblearn RandomOverSampling function, we created new X and y training variables.  In doing this, we fit the ROS algorithm to the initial data, which resulted in a completely balanced new target variable count: 51366 for low_risk and 51366 for high_risk.  Of note, this type of oversampling approach takes existing data and adds it back into the training data set, thereby increasing the size of the minority class.  Below outlines the various approaches to evaluating how well this balancing approach has worked.

![Random Oversampling](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/ROS.png)

As we can see, the balanced accuracy is fairly low with a 64% probability of accurately predicting the outcomes (i.e. true positives and true negatives).  That said, however, when we dive deeper into the confusion matrix, we can see the there is a significantly high false positive rate - meaning that there are a lot of predictions the model makes that says something is high risk when it is actually low risk.  This is reflected as well in the Classification Report, specifically with the precision score for high_risk which is .01.  Although low precision for high_risk is not necessarily bad, as most lenders would be more sensitive to false negatives in this case (predicted to be low risk but is actually high risk), it does mean that LendingClub is denying a substantial amount of business.  With regards to sensitivity, or recall, however, you see that for high_risk recall, the sampling approach + linear regression is closer to the balanced accuracy score, but still not very high.  A .66 recall for high_risk means that of the people who truly are high risk, the model predicts that only 66% of the time.  Finally, if we look at the F1 score for high_risk at .02, we see consistent results to recall and precision for high_risk.


- ***SMOTE Oversampling***

Using SciKit Learn’s imblearn SMOTE oversampling function, we created new X and y training variables.  In doing this, we fit the SMOTE algorithm to the initial data, which, like the ROS oversampling, resulted in a completely balanced new target variable count: 51366 for low_risk and 51366 for high_risk.  Of note, this type of oversampling approach takes existing data and adds synthetic data into the training data set, thereby increasing the size of the minority class.  Below outlines the various approaches to evaluating how well this balancing approach has worked.

![SMOTE Oversampling](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/SMOTE.png)

As we can see, the balanced accuracy is low and nearly the same as the ROS oversampling method with a 65% probability of accurately predicting the outcomes (i.e. true positives and true negatives).  Likewise, when we look at the confusion matrix, we see the same significantly high false positive rate, and a very low precision rate for high_risk in the Classification Report (.02).  Recall, or sensitivity, for high_risk is only slightly lower with .61 for the SMOTE approach vs. .66 for ROS.  Finally, if we look at the F1 score for high_risk at .02, the exact same as ROS.

*In general SMOTE oversampling performed nearly the same as ROS for precision and recall in predicting high_risk applicants*

- ***Cluster Centroid Understampling***

Using SciKit Learn’s imblearn Cluster Centroids undersampling function, we created new X and y training variables.  In doing this, we decreased the overbalanced target variables class to match the under balanced ones.  This resulted in a completely balanced new target variable count: 246 for low_risk and 246 for high_risk.  Much like SMOTE for oversampling, Cluster Centroid uses synthetic data points called centroids that are representative of clusters.  The model then down selects the major class to the size of the under sampled minority class using those synthetic centroids/clusters.  Below outlines the various approaches to evaluating how well this balancing approach has worked.

![Cluster Centroid Undersampling](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/CC.png)

As we can see, the balanced accuracy is even lower than ROS and SMOTE oversampling techniques with a score of .544 (i.e. it correctly predicts only 54.4% of the time true negatives and true positives).  When we dive deeper into the confusion matrix, we can see that the problems of false positives grows even more with this approach, with total false negatives at 10002  for Cluster Centroid unhderstampling vs. 5317 for SMOTE and 6546 for ROS.  The false negatives stay fairly consistent as do the true positives.  The Classification Report, shows a low precision rate for high_risk predictions (.01) and a slightly better recall rate for high_risk than ROS (.67 vs .66).  The f1 score is only slightly below that of ROS and SMOTE oversampling techniques.  

*In general Cluster Centroid undersampling performed worse than both SMOTE and ROS for precision and recall in predicting high_risk applicants*

- ***SMOTEENN Combination (Over & Under) Sampling***

Using SciKit Learn’s imblearn SMOTEENN combination sampling function, we created new X and y training variables.  In doing this, we decreased the overbalanced target variables class to match the under balanced ones.  This resulted in a more balanced target variable count: 46653 for low_risk and 51361 for high_risk.  SMOTEEN uses a combination of SMOTE for oversampling and Edited Nearest Neighbors (ENN) for undersampling.  Below outlines the various approaches to evaluating how well this balancing approach has worked.

![SMOTEENN Combination Sampling](https://github.com/MaureenFromuth/Credit_Risk_Analysis/blob/main/SMOTEENN.png)

As we can see, the balanced accuracy is the best of the models thus far with a score of .64.  Looking at the confusion matrix, the true positives increase slightly and the false negatives decreased.  The false positives, however, only decreased slightly as opposed to the Cluster Centroid undersampling approach (10002 for Cluster Centroid vs. 7305 for SMOTEENN).  As such, this model still has a significant problem identifying applicants as high risk when they actually are not.  The Classification report shows a slightly better recall for high_risk applicants as opposed to the other sampling techniques (.71 for SMOTEENN vs. .67 for CC vs. .61 for SMOTE vs. .66 for ROS).  Precision and F1 scores for high_risk, however, are the exact same as the two oversampling techniques.  

*In general SMOTEEN combination sampling performed slightly better in terms of accuracy and recall than the other sampling models, but has the second worse score for false positives in predicting high_risk applications*
