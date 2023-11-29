# Machine Learning and Heart Disease

## Authors
- [Kaija Gregory](https://github.com/kaijaygregory)
- [Sam Lind](https://github.com/SamLind11)
- [Julianne Lo](https://github.com/julo123456)
- [Elijah Reid](https://github.com/EliReid222)
## Summary
Using data from a 2022 CDC health survey, we aimed to create several machine learning models which might assist medical professionals in predicting heart disease in patients.  The original data set contained over 40 features comprised of largely categorical data and over 246,000 rows.  After preprocessing the data and selecting a subset of relevant features, several machine learning algorithms were trained and evaluated on the data.  Models developed include Logistic Regression, Decision Tree, Random Forest, and Perceptron Neural Networks.

## Data Visualizations
Before developing any machine learning models, several data visualizations were developed using Tableau to better understand the 2022 CDC survey data.

<img src='https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/d41ce71d-a3c9-4da1-a390-233c8d1763a8' width=600/>

<img src='https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/7c7fca1a-db9e-4a97-b0ae-8349456c706e' width=600/>

<img src='https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/dbd5f673-c5d6-42cd-8aeb-0484502d6dd4' width=600/>

<img src='https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/1a9cca0e-ed99-4a3a-8250-e6e13359943c' width=600/>

## Data Preprocessing
The `data_cleaning.ipynb` contains code for processing data from the csv file.  The data was first loaded into a Pandas Dataframe.  Next, fifteen relevant features were selected from the original 40.  Continuous values such as Height and Weight were scaled using `StandardScaler` and categorical data was one-hot encoded into separate columns using the `get_dummies` method in Pandas.  

The resulting Dataframe was then randomly sampled without replacement to generate data subsets with 25,000, 50,000, and 100,000 rows.  Each of these data sets were then exported as csv files, which can be found in the `preprocessed_data` folder.

## Machine Learning Models

### Logistic Regression

Logistic regression models were trained on two different datasets: one was the preprocessed data subset containing 25,000 rows, and the other was a randomly oversampled version of that data.  Random oversampling (ROS) was used to balance the classes (those with heart disease and those without).  The model trained on ROS data attained an accuracy = 0.85, precision = 0.22, and recall = 0.73.  This is compared to the non-ROS trained data which attained an accuracy = 0.95, precision = 0.48, and recall = 0.18.  While both models achieved relatively high accuracy scores, their usability is undermined by their low precision and recall scores, indicating their inability to actually identify heart disease in patients.  In the first model, the slightly lower accuracy is likely a fair trade-off for a higher recall score, which represents the proportion of test patients with heart disease that were actually identified as having heart disease.

### Decision Tree and Random Forest Algorithms

Three decision tree models were generated, one using the basic `DecisionTreeClassifier()` model and two using the `RandomForestClassifier`, both from the `scikit-learn` library.  

![image](https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/99b1cf53-248f-48aa-87fc-7992390bb111)

The results for these models are unfortunately similar to those for the Logistic Regression models.  While each model attained a high accuracy score, the low precision and recall scores make the models unfit for predicting heart disease.

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

**Figure**: Model responses for people with HD

![image](https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/e17e4dfb-aedc-4d85-86d2-6b49c6176108)

**Figure**: Model responses for people without HD

![image](https://github.com/SamLind11/Machine_Learning_and_Heart_Disease/assets/131621692/ca4df459-0fd8-42e7-aa7c-005769a9ecac)

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
