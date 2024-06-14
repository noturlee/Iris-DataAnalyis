<img src="Images/banner.gif"/>

# Iris Flower Classification Model

# Table of Contents
1. [Overview](#overview)
2. [Models Used](#models-used)
3. [Data Preprocessing](#data-preprocessing)
   - [Data Loading](#data-loading)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Models Training and Evaluation](#models-training-and-evaluation)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
   - [Interpretation of Classification Report](#interpretation-of-classification-report)
5. [Data Visualization](#data-visualization)
6. [Findings](#findings)
   - [Data Exploration](#data-exploration)
   - [Model Performance](#model-performance)
7. [Ouput](#ouput)
8. [Conclusion](#conclusion)

## Overview

This project aims to classify Iris flowers into three species—setosa, versicolor, and virginica—based on their sepal and petal measurements using machine learning techniques. The dataset comprises 150 samples evenly distributed among these species, making it a standard benchmark for introductory classification tasks.

## Models Used

Two primary models were employed:

- **Logistic Regression:** A linear model suitable for binary and multi-class classification tasks.
- **Random Forest Classifier:** An ensemble learning method effective for handling complex classification problems.

## Data Preprocessing

### Data Loading

The Iris dataset was loaded from a CSV file containing 150 records and 5 attributes: sepal length, sepal width, petal length, petal width, and species.

<img src ="https://i.pinimg.com/originals/d4/5e/47/d45e47ee0ea7e9e285de7d10f34d2f6e.gif" width="300"/>

### Exploratory Data Analysis (EDA)

- **Summary Statistics:** Provided insights into the distribution and variation of sepal and petal measurements.
- **Pair Plot:** Visualized relationships between features across different species.
- **Correlation Heatmap:** Showed feature correlations, aiding in feature selection.

## Models Training and Evaluation

### Model Training

- **Splitting Data:** The dataset was split into training (80%) and testing (20%) sets.
- **Logistic Regression:** Trained a linear model for classification.
- **Random Forest Classifier:** Trained an ensemble model to handle complex relationships.

### Model Evaluation

- **Best Parameters:** The optimal parameters found for the Random Forest Classifier were {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}. These parameters were selected based on cross-validation to maximize accuracy.
  
- **Best Random Forest Accuracy:** The model achieved an accuracy of 100% on the test dataset, indicating that it correctly classified all Iris flowers.

### Interpretation of Classification Report

The classification report provides a detailed breakdown of how well the model performed for each species:

- **Precision:** Measures the accuracy of positive predictions.
- **Recall:** Indicates how well the model captures instances of a class.
- **F1-score:** Harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.
- **Support:** Number of samples in each class.

For example:
- **Iris-setosa:** The model correctly classified all 10 samples of Iris-setosa, achieving perfect precision, recall, and F1-score.
- **Iris-versicolor:** Similarly, all 9 samples of Iris-versicolor were correctly classified.
- **Iris-virginica:** All 11 samples of Iris-virginica were also classified correctly.

The overall accuracy of 100% indicates that the model successfully learned the patterns in the data and accurately classified the Iris flowers into their respective species.

## Data Visualization

- **Pair Plot:** Visualizes relationships between sepal length, sepal width, petal length, and petal width across different species.
- **Correlation Heatmap:** Shows the correlation coefficients between these features, aiding in feature selection and understanding feature importance.
  
<p float="left">
<img width="400" alt="Screenshot 2024-06-15 at 00 38 42" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/0ce8011d-7aca-4dce-b161-63f967f4cfd0">
<img width="400" alt="Screenshot 2024-06-15 at 00 39 10" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/7fec5f5c-bfb0-4541-abbc-28ced0f10c80">

   
<p float="left">
<img width="400" alt="Screenshot 2024-06-15 at 00 39 22" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/fe91c64d-10d1-481c-8dcf-28c18c8f606c">
<img width="400" alt="Screenshot 2024-06-15 at 00 40 09" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/169ab481-cd25-4eb2-8873-2c5122494fac">

## Findings

### Data Exploration:

- Summary statistics provided insights into the distribution and variation of sepal and petal measurements.
- Pair plots visually represented the clustering of different species based on their measurements.
- The correlation heatmap highlighted significant relationships between certain features, influencing classification accuracy.

### Model Performance:

- Both Logistic Regression and Random Forest Classifier achieved perfect accuracy of 100% on the test dataset.
- Precision, recall, and F1-score metrics confirmed the models' ability to effectively distinguish between Iris species.

## Output

The output from the models includes:
- Best Parameters for Random Forest Classifier: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}
- Best Random Forest Accuracy: 1.0
- Classification Reports for Logistic Regression and Random Forest Classifier, showing precision, recall, and F1-score metrics for each Iris species.

<img width="1277" alt="Screenshot 2024-06-15 at 00 47 36" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/6dc4586e-7699-44c0-ba4d-1834822cfc45">

<br/>
<br/>

<img width="1271" alt="Screenshot 2024-06-15 at 00 48 03" src="https://github.com/noturlee/Iris-DataAnalyis-CODSOFT/assets/100778149/3af5f49e-185d-4bf9-98aa-9bc966efaab6">

## Conclusion

This project demonstrated the application of machine learning models to classify Iris flowers based on their morphological measurements with high accuracy. The selected models, Logistic Regression and Random Forest Classifier, performed exceptionally well, showcasing their effectiveness for such classification tasks. By leveraging data preprocessing, visualization, and thorough evaluation techniques, this project provides a robust framework for introductory classification tasks.

