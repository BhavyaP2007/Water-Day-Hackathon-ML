I have used scikit library of python to predict the test data.
I used the training data to train my Random Forest Regression Model. It creates multiple decision trees to find accurate results. I have tuned the number of estimators for highest accuracy.
Using my model, I am able to perform regression and classification on the test data at the same time with a 97.9% accuracy on my training data.
There were some inaccurate values in residents and humidity columns. Training values which can cause overfitting were dropped. Some of these values were interpolated using pandas library. The irregular values in Income column were given a separate values.
To perform classification on categorical values, I converted the categories into separate integers using Label Encoder.
I am submitting a ZIP file with the provided dataset, readme file and my prediction.
LIBRARIES USED :
1. Scikit
2. Pandas
3. NumPy
PROGRAMMING LANGUAGE : Python

I am saving the prediction.csv file in the main folder and the dataset folder.