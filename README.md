# REGRESSION I: Simple-Linear-Regression
Section 6 of Machine Learning A-Z: AI, Python &amp; R. We now start diving deep into the models of machine learning. We'll learn how to use it and how to analize it!

# Important

Everything we see and code is used with the help with the scikit learn library, mainly, which is our most valuable object of study for machine learning in python. Learn more about it here: https://scikit-learn.org/stable/
## Why use it?

The objective of simple linear regression in the machine learning area is to model the relationship between a dependent variable (also called the target) and a single independent variable (also called the predictor or feature). The goal is to find a linear relationship that can be used to make predictions or understand the correlation between the variables.
## Using the Google Collab Template

### Import the libraries

1.  As a standard, we always import three libraries: numpy, matplotlib.pyplot and pandas.
   ```
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   ```
### Import the Dataset

1. second of all, as always, we specify the .csv file that we will be analizing, using the pandas function, which gives us fast shortcuts to importing it and analizing it (dividing into two sections: dependent and independent variables)

   ```
   dataset = pd.read_csv('Salary_Data.csv')
   X = dataset.iloc[:, :-1].values
   y = dataset.iloc[:, -1].values
   ```
### Splitting the dataset into the Training set and Test set

1. After we divide the sections, we have to divide our data into two, the training and test sets, which we'll first learn the data and in the test (about 20% of data) we will make our predictions.
   ```
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
   ```
3. What we are doing in the image above is: create two variables for the X (dependent variables) and y (independent variables), one for the training set and another for the test set.
   
4. The train_test_split function will make all the subsets for the dataset. The test_size will specify the proportion of the dataset to include in the test split, so 1/3 (the less data in the test split, the more data in the training set will be received and learnt). And the random_state is an optional parameter that allows you to set a seed for the random number generator, which ensures reproducibility.

### Training the Simple Linear Regression model on the Training set
   ```
   from sklearn.linear_model import LinearRegression
   regressor = LinearRegression()
   regressor.fit(X_train, y_train)
   ```
After we make the variables, we have to train them to learn the dataset and eventually make predictions. To do so we call the linearRegression function from the scikit learn linear model library and we use the fit method, which will allow the object to obtain the data from the .csv file.


### Predicting the Test set results

   ```
   y_pred = regressor.predict(X_test)
   ```
1. Finally, to predict the results received, we have to use the scikit learn method predict on the regressor object the taught the training data, which will need one parameter: the x_test (that stands for the test set of features).

### Visualising the Training set results
   ```
   plt.scatter(X_train, y_train, color = 'red')
   plt.plot(X_train, regressor.predict(X_train), color = 'blue')
   plt.title('Salary vs Experience (Training set)')
   plt.xlabel('Years of Experience')
   plt.ylabel('Salary')
   plt.show()
   ```
![image](https://github.com/Yudiaramos/REGRESSION-I---Simple-Linear-Regression/assets/71808184/71ab7e6f-7663-45c5-90a8-39538d6bcbc3)

1. Using the library matplot, we to these simple coding steps that will show the graph obtained by the training set
2. The code to plot the regression is almost identical everytime, only needing to change the variables.
#### Analizing the graph
We can see that the regression line obtained is very optimal, but we still see a lot of dots outside the regressor, and that can cause a percentage of errors.

### Visualising the training set results
   ```
   plt.scatter(X_test, y_test, color = 'red')
   plt.plot(X_train, regressor.predict(X_train), color = 'blue')
   plt.title('Salary vs Experience (Test set)')
   plt.xlabel('Years of Experience')
   plt.ylabel('Salary')
   plt.show()
   ```
![image](https://github.com/Yudiaramos/REGRESSION-I---Simple-Linear-Regression/assets/71808184/5c863706-955a-4ae4-b60f-5dc2054b8caa)

1. Again, we use the same code but instead of using the training variables, we use the test variables to obtain the predicted results.

#### Analizing the Test set results
We now see that the new graph with the test results shows the dots very close to the regression line, which shows that the predictions are very optimal and are in harmony with the results we expected (dots as close to the line as possible).
