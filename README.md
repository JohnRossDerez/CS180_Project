# CS 180 Project Implementation
Carl David Ragunton  
John Ross Derez

# Google Drive Link
https://drive.google.com/drive/folders/1W6fTsMYLWXVTZR2ZM6PBjNh-AY3dRgnu  
All the trained models and datasets used in this project are found in this google drive folder.  

# Project Summary
The objective of this project is to determine the highest and lowest region that contribute to the total game sales given different features in the dataset. A 'region' can be classified as NA_Sales, EU_Sales, JP_Sales, and Other_Sales. To achieve this, DesicionTreeClassifier is used and its results will be compared to MLPClassifier, another machine learning classifier algorithms. As an attempt to extend our model from 6825 records with complete attributes to all of the available 16450 records in the dataset, we used HistGradientBoostingClassifier. It is an ensemble machine learning classifier algorithm that utilizes decision trees and is capable of handling missing values. Two models are trained for each of these machine learning models; one to determine the highest region and one to determine the lowest region. Thus, there are a total of 6 final models. Several comments will state which code parts are for the highest and lowest region in their specific notebooks. 

# Code Summary
The code is divided into 6 notebooks: 1 for the data preprocessing, 3 for the different models used, 1 for saving the models, and 1 for Model Checking and Demo. The machine learning models used are DecisionTreeClassifier, MLPClassifier, and HistGradientBoostingClassifier. A random_state of 427 was applied to every randomized parts of the code to make the results consistent.

# Note
Hyperparameter tuning for MLPClassifier and HistGradientBoostingClassifier is conducted in Jupyter to avoid runtime disconnections. The models trained in Jupyter are not compatible with Google Colab. Thus, hyperparameters are manually inputted and the model is retrained to make it compatible with Google Colab. The final implementation of the models are expected to run in Google Colab.

This is a copy of the folder stored in https://drive.google.com/drive/folders/1W6fTsMYLWXVTZR2ZM6PBjNh-AY3dRgnu?usp=drive_link  
Some models in the MODELS folder are removed because they exceed the 10MB limit mentioned in the project guidelines. Also, some models in the MODELS folder are removed for the same reason. Please view the orginal folder stored in the link above. Here is a list of the removed files from DATASETS and MODELS folder:
- DATASETS Folder
  - processed_noNull.csv
  - processed_withNull.csv
  - X1_test.csv
  - X1_train.csv
  - X2_test.csv
  - X2_train.csv
- MODELS folder
  - Highest_Region_HistGradientBoostingClassifierClassifier.pkl
  - Lowest_Region_HistGradientBoostingClassifierClassifier.pkl

# Folder Items
- readme.md: This file

- DATASETS Folder: Contains datasets before and after preprocessing
  - raw_data.csv: raw dataset from kaggle
  - processed_withNull.csv: preprocessed data without removing null values
  - processed_noNull.csv: preprocessed data without null values
  - X1_train.csv, X1_test.csv, hi1_train.csv, hi1_test.csv: data splitting for Highest Region for DecisionTreeClassifier and MLPClassifier
  - X1_train.csv, X1_test.csv, lo1_train.csv, lo1_test.csv: data splitting for Lowest Region for DecisionTreeClassifier and MLPClassifier
  - X2_train.csv, X2_test.csv, hi2_train.csv, hi2_test.csv: data splitting for Highest Region for HistGradientBoostingClassifier
  - X2_train.csv, X2_test.csv, lo2_train.csv, lo2_test.csv: data splitting for Lowest Region for HistGradientBoostingClassifier

- MODELS Folder: contains saved models from training (pickle was used to save the models)
  - Highest_Region_DecisionTreeClassifier.pkl
  - Highest_Region_MLPClassifier.pkl
  - Highest_Region_HistGradientBoostingClassifier.pkl
  - Lowest_Region_DecisionTreeClassifier.pkl
  - Lowest_Region_MLPClassifier.pkl
  - Lowest_Region_HistGradientBoostingClassifier.pkl

- NOTEBOOKS Folder: contains python notebooks used to train the models
  - Data_Preprocessing.ipynb: Python Notebook that contains codes for data loading and preprocessing
  - Hyperparameter Tuning Folder
    - Decision Tree Training.ipynb: Training and Hyperparameter Tuning of DecisionTreeClassifier Model
    - MLP_Training.ipynb: Training and Hyperparameter Tuning of MLPClassifier Model
    - HistGradientBoosting.ipynb: Training and Hyperparameter Tuning of HistGradientBoostingClassifier Model
  - Model_Implementation.ipynb: python notebook used to save the models with the results of hyperparameter tuning manually inputted to them (models saved from Jupyter does not work properly on google colab)
  - Model Checking and Demo Code.ipynb: python note book to check if the saved models are working as intended and serves as the demo code
  

# How to Run
In order to run the implementation and fully understand it, run every code snippet. Note that the hyperparameter tuning for the MLPClassifier and HistGradientBoostingClassifier will take a few hours.  

The python notebooks are recommended to be visited in the following order. Please check Folder Items to see where the files are in the Gdrive. In the Uvle submission, datasets and models that are more than 10MB are removed so, please check the google drive for the complete collection of files. We do not recommend running notebooks 3 and 4 as it takes several hours to finish. 
1. Data_Preprocessing.ipynb
2. Decision Tree Training.ipynb
3. MLP_Training.ipynb
4. HistGradientBoosting.ipynb
5. Model_Implementation.ipynb
6. Model Checking and Demo Code.ipynb

# Data Splitting
Data are divided into Training and Testing Sets. Training set will be used to train the models. Crossfold Validation is used in the hyperparameter tuning algorithm from sklearn. Testing set will be used for testing the results of the trained models. Training Set is 80%, and Testing Set is 20%.

# Hyperparameter Tuning
The algorithm used for hyperparameter tuning is grid search cv and randomized search cv. The grid search was used on the decision trees because it runs fast compared to the other models. Randomized search was applied to MLPClassifier and HistGradientBoostingClassifier as we are forced budget our computing power.

# Data Preprocessing
The implementation starts with importing some libraries and loading the dataset gathered from kaggle about video game sales with ratings. Then, rows of the dataset with and without null values are identified. Note that they are not immediately removed.  

For the whole dataset, a 'highest' and 'lowest' columns are created to indicate which region (NA_Sales, EU_Sales, JP_Sales, and Other_Sales) has the highest and lowest game sales. These 2 new columns are created by evaluating values from the columns: NA_Sales, EU_Sales, JP_Sales, and Other_Sales. These columns are then, removed from the dataset.  

The next step is one-hot encoding because several of the data in the dataset are categorical and can't be accepted as it is by the ml algorithms. The columns that are one-hot encoded are Platform,	Year_of_Release,	Genre,	Publisher,	Developer, and	Rating. These original columns are then dropped from the dataset.  

The last part of the data preprocessing is creating a dataset for each of the models later. Both DecisionTreeClassifier and MLPClassifier does not accept null values and so, df1 will be the dataset of these 2 models and rows with null values are removed from it. On the other hand, HistGradientBoostingClassifier is designed to accept datasets with null values and so, its dataset preserves the rows with null values. df1 contains more than 6000 rows and HistGradientBoostingClassifier has more than 16000 rows.  

# Model 1: Decision Tree Classifier
The DecisionTreeClassifier used is imported from sklearn. This specific algorithm is used to easily visualize the results of the model and analyze the tree which is the one of the main objective on why this specific dataset is chosen. The data (df1) are split into the Training and Testing Sets and are used on the model to produce preliminary results. Hyperparameter tuning is then done and the model is reimplemented with the results of the tuning. Running the model implementations should take a few seconds to complete and a few minutes for the hyperparameter tuning. Finally, the decision trees are visualized. Note that running the visualization code for the lowest region will take around a minute because its max_depth is none.

# Model 2: Multi-Layer Perceptron Classifier
The MLPClassifier used is imported from sklearn. This algorithm was selected to compare the performance of decision trees against an algorithm that uses backpropagation and neural networks. The data (df1) are split into the Training and Testing Sets and are used on the model to produce preliminary results. Hyperparameter tuning is then done and the model is reimplemented with the results of the tuning. Running the model implementations should take a few minutes (around 5 minutes in google colab) to complete and the hyperparameter tuning will take a few hours to finish.

# Model 3: Hist Gradient Boosting Classifier
The HistGradientBoostingClassifier used is imported from sklearn. The algorithm utilizes a decision tree but it allows the use of data with missing values. The data are split into the Training and Testing Sets and are used on the model to produce preliminary results. Hyperparameter tuning is then done and the model is reimplemented with the results of the tuning. Running the model implementations should take a few minutes (around 5 minutes in google colab) to complete and the hyperparameter tuning will take a few hours to finish.

# Saved Datasets and Models
The following are the different dataset versions throughout the code implementation:
- raw_data.csv
- processed_noNull.csv (df1, for DecisionTreeClassifier and MLPClassifier)
- processed_withNull.csv (for HistGradientBoostingClassifier)
- data splits are also saved in the DATASETS folder

The following are the different trained models for the highest regions:
- Highest_Region_DecisionTreeClassifier.pkl
- Highest_Region_MLPClassifier.pkl
- Highest_Region_HistGradientBoostingClassifier.pkl

The following are the different trained models for the lowest regions:
- Lowest_Region_DecisionTreeClassifier.pkl
- Lowest_Region_MLPClassifier.pkl
- Lowest_Region_HistGradientBoostingClassifier.pkl

The saved models can be used by using "saved_model = joblib.load('saved_model.pkl')".

## References
https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn  
https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/  
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html  
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
https://scikit-learn.org/stable/modules/grid_search.html
https://scikit-learn.org/dev/auto_examples/ensemble/plot_gradient_boosting_categorical.html