# Fatigue Level Classification Using GRF

## Project Overview
This project aims to develop a machine learning model for classifying fatigue levels based on the time-series data analysis of ground reaction forces (GRF). Fatigue is an important factor to consider in various fields, such as sports performance, workplace safety, and healthcare. By studying the patterns and characteristics of GRF data, we can train a model that accurately classifies fatigue levels, providing valuable insights and potentially aiding in fatigue management.

## Main Stages
*  Data Collection
*  Exploratory Data Analysis (EDA)
*  Data Consolidation & Feature Extraction
*  Fatigue Level Labeling
*  Feature Selection
*  Model Building

## Data Collection
To accomplish this objective, we collected GRF data from individuals performing physical activities of varying intensities and durations. This data will be captured using appropriate sensors or wearable devices capable of measuring the forces exerted on the ground during the activities. The collected data will serve as the foundation for training and evaluating our machine learning model.

## Exploratory Data Analysis (EDA)
Before developing our machine learning model, we will conduct exploratory data analysis (EDA) on the GRF dataset. EDA techniques will help us understand the characteristics of the data, identify any outliers or missing values, visualize the distribution and correlations among variables, and gain insights into the relationships between GRF patterns and fatigue levels. EDA will guide our feature engineering process and inform the subsequent model development.

First Univariate Analysis was made, such as: Descriptive Statistics, Box-plots & Outliers, Skewness & Kurtosis, Line Plots & Variation. Secondly, Multivariate Analysis, such as: Features Correlation to get rid of any multicollinearity issue, Scatter Plots to study the kind of correlation between features.

Notebook: EDA.ipynb

## Data Consolidation & Feature Extraction
Once the GRF data is collected, we will apply feature extraction techniques to derive meaningful characteristics from the raw data. These features may include statistical measures, frequency domain analysis, or time-domain parameters that capture important aspects of the GRF patterns. Additionally, feature selection methods will be employed to identify the most relevant features that contribute to fatigue level classification.

Functions developed:

### load_grf:
This function is designed to load ground reaction force (GRF) data from a specified directory path. It traverses through the subject  folders within the given path and extracts the desired features to build ten dataframes, one for each subject.

#### Parameters:
* `path`: a string representing the directory path where the GRF data is stored.

#### Returns:
* `subjects`: a list of each subject's data frame.

Please make sure to provide the correct directory path containing the GRF data as an argument when calling the load_grf function.

### rms_calculation:
This function calculates the root mean square (RMS) of a given set of values. RMS is a measure of the average magnitude of a set of values and is commonly used in signal processing and data analysis. 

#### Parameters:
* `values` (array-like): an array or list of numerical values for which the RMS needs to be calculated.

#### Returns:
* `rms` (float): the root mean square value of the input values.

### combine_subjects:
This function combines all the subjects data into one dataframe and calculates various statistical measures for each feature. The statistical measures include mean, standard deviation, root mean square (RMS), skewness, kurtosis, interquartile range (IQR), mean absolute deviation (MAD), maximum value, zero-crossing count (ZCC), and range.

#### Parameters:
* `subjects` (list): a list of each subject's data frame.

#### Returns:
* `stats_vals` (list): a list of dictionaries containing the calculated statistical measures for each feature.

Notebook: data_organizing.ipynb

## Fatigue Level Labeling
Labeling the fatigue levels for the subjects based on specific records chosen for each level. It takes in a list of subjects statistical data and assigns a fatigue level label to specific records. Records 1 and 2 of each subject are assigned as level 1 fatigue, records 5 and 6 as level 2, and records 9 and 10 as level 3.

Function developed:

### fatigue_level_labeling:

#### Parameters:
* `subjects_stats` (list): a list of pandas DataFrames containing the statistical data for each subject.

#### Returns:
* `labeled_subjects` (list): a list of pandas DataFrames with fatigue level labels assigned to specific records.

Notebook: data_organizing.ipynb

## Feature Selection
In this step, we applied two methods for selcting features that will best serve our machine learning model:

* Feature Importance
* Recursive Feature Elimination

### Feature Importance
The Feature Importances with a Forest of Trees in scikit-learn is used to evaluate the importance of features on an artificial classification task, the output will be a diagram showing the blue bars which are the feature importances of the forest, along with their inter-trees variability represented by the error bars. 

### Recursive Feature Elimination (RFE)
The RFE is a feature selection method that recursively eliminates less important features based on their ranking, allowing for the selection of the most relevant features for a given task. RFE is useful for reducing the dimensionality of datasets, improving model performance, and enhancing interpretability.

At the end we will get:
* ```support_``` (array-like): a boolean mask indicating the selected features.
* ```ranking_``` (array-like): the feature ranking, where a lower value indicates higher ranking.

Both methods gave roughly same values, the selected features to complete with were: ```['VGRF_skewness', 'VGRF_kurtosis', 'HGRF_skewness', 'HGRF_mad', 'LGRF_kurtosis', 'fatigue_level']```

Notebook: data_organizing.ipynb

##  Model Building
We will then proceed with the development of a machine learning model that can effectively classify fatigue levels based on the selected features from the GRF data. Several classification algorithms such as RandomForestClassifier, Decision Tree, Naive Bayes, KNN, and lazypredict to try many different models on our data and output different metrics (Accuracy, F1 Score) will be explored and evaluated.

Notebook: model_building.ipynb

## Conclusion

## Supervision
* [@Ramzi Halabi](https://github.com/RamziHalabi) 

## Contact
* You can email me on: amjad.baba91@gmail.com.  
* Get in touch with my blog posts on [medium](https://medium.com/@amjadelbaba), and don't forget to drop your comments!
