# Credit_Risk_Analysis
Apply machine learning to solve a real-world challenge: credit card risk.

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. In this project, we are going to to employ different techniques to train and evaluate models with unbalanced classes using imbalanced-learn and scikit-learn libraries. 

We are going to use the credit card credit dataset from LendingClub, a peer-to-peer lending services company to:
 
- Oversample the data using the RandomOverSampler algorithms.
- Oversample the data using the SMOTE algorithms.
- Undersample the data using the ClusterCentroids algorithm. 
- Use a combinatorial approach of oversampling and undersampling using the SMOTEENN algorithm. 
-  Compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

### Aim
The aim of this project is to evaluate the performance of 6 machine learning models in order to find the most suitable one to predict credit risk.

## Resources 
- Data Source: <a href="https://github.com/MireyNM/Credit_Risk_Analysis/blob/main/Credit_Risk_Code/LoanStats_2019Q1.csv">LoanStats_2019Q1.csv</a></br>
- Software: Python 3.7.13, Jupyter Notebook </br>
- Scripts: <a href="https://github.com/MireyNM/Credit_Risk_Analysis/blob/main/Credit_Risk_Code/credit_risk_resampling.ipynb"> credit_risk_resampling.ipynb </a> </br>
<a href="https://github.com/MireyNM/Credit_Risk_Analysis/blob/main/Credit_Risk_Code/credit_risk_ensemble.ipynb">credit_risk_ensemble.ipynb </a> </br>

## Analysis of Data 
### First Model: Naive Random Oversampling Model 

<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244105-fb9dfade-fa50-40c5-8ab9-04983bc25a9d.png">
</p>
<p align = "center">
Fig. 1 - Naive Random Oversampling Evaluation
</p>

- The total number of low risk (17104) is so high compared to the total number of high risk (101). 
- Model accuracy is 0.66 or 66%, since the data in unbalanced we need to look into precision and sensitivity.
- Precision for high risk is extremely low (0.01) which reflect the low number of True Positive high risk (75) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 75+7016 = 7091). However, the sensitivity of detecting true high risks is 0.74 (TP/TP+FP = 75/101=0.74), which means 26 cases of 101 high risk were not detected. <br/>
- Sensitivity for low risk is 0.59 which means that 10088 are actually low risk while 17104 cases were detected low risk.

### Second Model:  SMOTE Oversampling Model 

<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244310-32ce9a72-758f-4e2d-9337-a5b06d5b12af.png">
</p>
<p align = "center">
Fig. 2 - SMOTE Oversampling Evaluation
</p>

- The total number of low risk (17104) is so high compared to the total number of high risk (101). 
- Model accuracy is 0.65 or 65%,which is slightly less than that of Naive Random Oversampling Model. 
- Since the data in unbalanced we need to look into precision and sensitivity.
- Precision for high risk is still extremely low (0.01) which reflect the low number of True Positive high risk (64) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 64+5523 = 5587). However, the sensitivity of detecting true high risks has decreased to 0.63 (TP/TP+FP = 64/101=0.63), which means 37 cases of 101 high risk were not detected. <br/>
- Sensitivity for low risk has increased to 0.68, which means that 5523 cases are actually low risk while they have been detected as high risk. </br>

**Naive Random Oversampling Model vs SMOTE** </br>
It would better to adapt the Naive Random Oversampling Model as the sensitivity to detect the high-risk cases is bigger in this model that in SMOTE algorithm. Even though, the sensitivity of detecting low risk is larger in the SMOTE model, however it is better not to miss high-risk cases. <br/>

### Third Model: Undersampling Model 

<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244698-e02d5e10-f490-48f8-a909-b277c1f2b186.png">
</p>
<p align = "center">
Fig. 3 - Undersampling Evaluation
</p>

- Model accuracy in the Undersampling model has decreased to 0.53 or 53%,which is less than both Oversampling Models. 
- Since the data in unbalanced we need to look into precision and sensitivity. <br/>
- Precision for high risk is still extremely low (0.01) which reflects the low number of True Positive high risk (68) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 68+ 10500 = 5587). However, the sensitivity of detecting true high risks is 0.67 (TP/TP+FP = 68/101=0.67), which means 33 cases of 101 high risk were not detected in this model. <br/>
- Sensitivity for low risk has decreased to 0.39, which means that 10500 cases are actually low risk while they have been detected as high risk.  <br>

**Undersampling vs Oversampling** <br/> 
Even though detecting high risk is more important than detecting low risk. The undersampling algorithm is predicting 10500 from 17104 as being high risk while there are low risk cases. This will put more work on employees to re-study this huge number of False positive high risk loans. Moreover, the balanced accuracy score of this model is the lower than both oversampling algorithms, which is another reason to rule this model out. 

## Forth Model: Combination (Over and Under) Sampling
<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244844-07b8a5fa-4b29-4b59-baa5-96eb99811b7b.png">
</p>
<p align = "center">
Fig. 4 - Combination Sampling Evaluation
</p>

- Model accuracy in the Combination Sampling model is to 0.64 or 64%, which is so close to that of both Oversampling Models. 
- Since the data in unbalanced we need to look into precision and sensitivity. <br/>
- Precision for high risk is still extremely low (0.01) which reflects the low number of True Positive high risk (75) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 75+ 7570 = 7645). However, the sensitivity of detecting true high risks is 0.74 (TP/TP+FP = 75/101=0.74), which means 26 cases of 101 high risk were not detected in this model. <br/>
- Sensitivity for low risk is 0.56, which means that 7570 cases are actually low risk while they have been detected as high risk.  <br>

**Combination (Over and Under) Sampling vs Oversampling Sampling**  <br>
Precision for high risk in the combination model (0.74) is higher than that in SMOTE model (0.63) which indicated that this model is detecting more high-risk cases. 
However, both combination model and random oversampling model have the same precision for high risk (0.74). In this case, one could look in the precision for low risk. In the random oversampling model, it's 0.59 while its 0.56 in the combination model. Therefore, the random oversampling model is detecting less false positive cases and would be the best algorithm among the four. 

### Fifth Model: Balanced Random Forest Classifier
<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244969-96fac517-5d9b-4e6c-94f5-0b962b34edce.png">
</p>
<p align = "center">
Fig. 5 - Balanced Random Forest Classifier Evaluation
</p>

- Model accuracy in the balanced forest classifier has increased to 0.79 or 79%, which is higher than all previous models.
- Since the data in unbalanced we need to look into precision and sensitivity. <br/>
- Precision for high risk is slightly bigger than previous models but it's still low (0.04) which reflects the low number of True Positive high risk (72) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 72+1979 = 2051). Moreover, the sensitivity of detecting true high risks is 0.71 (TP/TP+FP = 72/101=0.71), which means 29 cases of 101 high risk were not detected in this model. <br/>
- Sensitivity for low risk has increased to 0.88, which means that 1979 cases are actually low risk while they have been detected as high risk. This number is less than what we got in previous models.

### Sixth Model: Easy Ensemble AdaBoost Classifier
<p align = "center">
<img width="499" alt="Outcomes_vs_Goals" src="https://user-images.githubusercontent.com/109363759/209244994-e9cefbae-8162-4774-9019-807aa065e623.png">
</p>
<p align = "center">
Fig. 6- Easy Ensemble AdaBoost Classifier Evaluation
</p>

- Model accuracy in the easy ensemble AdaBoost has increased to 0.93 or 93%, which is the highest value of all models.
- Since the data in unbalanced we need to look into precision and sensitivity. <br/>
- Precision for high risk has increased to 0.09. However, this low value reflects the low number of True Positive high risk (93) corresponding to the total number of positive low and high risk of our model gets (TP+FP = 93+970=1063). Moreover, the sensitivity of detecting true high risks has increased to 0.92 (TP/TP+FP = 93/101=0.92), which means only 8 cases of 101 high risk were not detected in this model. <br/>
- Sensitivity for low risk has also increased to 0.94, which means that only 970 cases are actually low risk while they have been detected as high risk. This number is less than what we got in all previous models.

## Conclusion
Based on all the above, I would recommend using the Easy Ensemble AdaBoost Classifier algorithm because: 
- It has the highest accuracy score (93%)
- It has the highest sensitivity of detecting true high risks (0.92); Which means only 8 cases with high risks were not detected using this model, 
- It has the highest sensitivity for low risk (0.94); Which means only 970 cases of 17104 were detected as high risk while in fact they are low risk. 

