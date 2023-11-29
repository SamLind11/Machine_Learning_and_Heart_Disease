# Machine Learning and Heart Disease

## Authors
- [Kaija Gregory](https://github.com/kaijaygregory)
- [Sam Lind](https://github.com/SamLind11)
- [Julianne Lo](https://github.com/julo123456)
- [Elijah Reid](https://github.com/EliReid222)
## Summary
Using data from a 2022 CDC health survey, we aimed to create several machine learning models which might assist medical professionals in predicting heart disease in patients.  The original data set contained over 40 features comprised of largely categorical data and over 246,000 rows.  After preprocessing the data and selecting a subset of relevant features, several machine learning algorithms were trained and evaluated on the data.  Models developed include Logistic Regression, Decision Tree, Random Forest, and Perceptron Neural Networks.

## Data Visualizations
INSERT VISUALIZATIONS

## Data Preprocessing
The `data_cleaning.ipynb` contains code for processing data from the csv file.  The data was first loaded into a Pandas Dataframe.  Next, fifteen relevant features were selected from the original 40.  Continuous values such as Height and Weight were scaled using `StandardScaler` and categorical data was one-hot encoded into separate columns using the `get_dummies` method in Pandas.  

The resulting Dataframe was then randomly sampled without replacement to generate data subsets with 25,000, 50,000, and 100,000 rows.  Each of these data sets were then exported as csv files, which can be found in the `preprocessed_data` folder.

## Machine Learning Models

### Logistic Regression

### Decision Tree and Random Forest Algorithms

Three decision tree models were generated, one using the basic `DecisionTreeClassifier()` model and two using the `RandomForestClassifier`, both from the `scikit-learn` library.  

INSERT RESULTS TABLE

While each model attained a high accuracy score, the low precision and recall scores make the models unfit for predicting heart disease.

### Neural Networks

Several perceptron neural network models were developed using different architectures and different amounts of training data.
Unlike the previous models, these models output a continuous value between 0 and 1.
- 0 indicates a near 0% chance that the person has heart disease.
- 0.50 indicates a 50% chance.
- 1 indicates a near 100% chance.
  
A total of 8 models were developed.  Progressive changes made to the models included altering the: 
- Size of input data (25,000 rows vs. 100,000) rows.
- Number of hidden layers (0, 2, and 4 layers).
- Number of epochs (50, 100, and 300 epochs).
- Randomly oversampled training data vs. not oversampled training data.

Several models that were initially developed showed very poor performance, as indicated by looking at the predictions made on test data.  Often the predictions for patients known to have heart disease were largely predicted to have near 0 probability.  This indicated that the models were likely overfitting to the data, and so new models were developed with resampled data using `RandomOverSampler`.  

Simple neural networks with no hidden layers, trained on randomly oversampled (ROS) data, showed much improved performance.  While these models saw some loss in accuracy when assessing patients without heart disease, the majority of patients with heart disease were assigned probabilities higher than 90%. 

## Conclusions

Many of the Machine Learning models reported having very high accuracy, but their low precision and recall scores make them unsuitable for predicting heart failure.

The neural network (specifically NN Model #2) stands out as the strongest model for two reasons: 
- Outputting a continuous probability between 0 and 1 is more realistic to the complex nature of preventative medicine.
- While its accuracy score is lower, it is better able to assign high probabilities to those with HD, and lower probabilities to those with no HD.

### Limitations
- Survey data was self-reported.
- Most of our data was categorical, which may limit the applicability of machine learning.

### Areas for Future Improvements
- More features from the survey might be used to train future models.
- There are many additional hyperparameters of the NN which could be tuned further.
- There may be more robust data sets which could be used to develop similar models.
