# Predicting Diabetes using EHR

## Introduction
Diabetes affects millions of people around the world. It strains not only the healthcare sector but also the wider global economy due, for example, sick days. 
Thus, there is a need to minimise expenditure where possible. One such means is to automate the process of detecting diabetic patients, where according to 
the NHS, 3% of adults remain undiagnosed with diabetes. 

The given dataset is record of different age groups, either diabetic or non diabetic, their blood glucose level reading with superficial body features like
body temperature, heart rate, blood pressure etc. Using calibrated wearable devices and apparatus, various age groups recorded their readings. A total of 
16,969 patients and 10 features are contained in that dataset.

In this report, I will detail the use of a machine learning (ML) pipeline for predicting whether a patient was diabetic or not using the superficial features
and glucose levels. 

## Methodology

The ideal methodology will be as detailed and as easy to follow as possible. Think of it like writing a recipe for a meal. When it comes to AI, it is best practice to report the version number of the software used (e.g., Python v.3.7.6) and the computer specs/colab specs. 

The data was obtained from the IEEE DataPort. The digital object identifier (DOI) is https://dx.doi.org/10.21227/c4pp-6347

All coding was performed using **Python** version (3.7.6)

## Results

Writing the results section is possibly the easiest of all the sections. A good template is:

when I...changed/compared/pre-processed, I found/saw/observed/discovered...., Therefore/thus/hence....

For example: 

<ins>When I explored</ins> the dataset for the ratio of diabetic to non-diabetic, <ins>I discovered</ins> the ratio to be 98:2. <ins>Therefore</ins>, the dataset was heavily imbalanced.

<ins>When I trained</ins> 4 models after splitting the data into training and testing instances, <ins>I found</ins> that all 4 models achieved high accuracies. <ins>Hence</ins>, all models were able to model the relationship between the features and the output. 


### Exploratory Data Analysis



<img width="250" alt="practical 3 - EDA - pie" src="https://github.com/Dr-M-ELBA/BIO729P/assets/158515515/1bd5b6c1-8ab4-410a-8fc4-9a6c63197064">

When I explored the dataset for the ratio of diabetic to non-diabetic, <ins>I discovered</ins> the ratio to be 98:2. <ins>Therefore</ins>, the dataset was heavily imbalanced.

<img width="399" alt="Practical 3 - EDA - boxplot" src="https://github.com/Dr-M-ELBA/BIO729P/assets/158515515/8fcc3b7c-d4dd-47a3-9c8d-033dd7b9f4a4">



When I examined the 'Age' feature, I found the data to be distributed within the minimum and maximum ranges, and no outliers were observed.


### Model Training & Evaluation
I trained 4 different learners where their accuracy, f1 score and matthews coefficient correlation (MCC) were:

| Model | Accuracy (%) | F1 (%) | MCC |
| ------------- | ------------- | ------------- | ------------- |
| Random Forest | 100  | 90 | 0.90 |
| Neural Net | 99 | 60 | 0.65 |
| Logistic Regression | 99 | 56 | 0.62 |
| Decision Tree | 99 | 87 | 0.87 |

The results show that despite the imbalance, all four models were able to accurately predict the patients clinical condition. The high MCC score confirmed
that the models were performing much better than random guessing. I then removed the feature 'Blood Glucose Level' and discovered...

## Discussion & Conclusion

Based on these findings, it can be concluded that ML is more than capable of predicting whether a patient is diabetic. 

The expected impact of this...

A possible reason for the high performance was....

A limitation of this study was....

future work will involve...
