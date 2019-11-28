# Credit Card Transaction Fraud Detection
### Trained and compared 10 models to detect whether a credit card transaction is fraudulent or not. 
#### Dataset contains 284,807 examples each with 31 features resulting in over 8,829,017 instances parsed.
#### ** Note ** The dataset is too large to upload to github so it is left out of the repo. It can be found at: https://www.kaggle.com/mlg-ulb/creditcardfraud

### Some initial data visualization

#### Transaction amount and time distribution
![image](https://user-images.githubusercontent.com/41659296/69772383-c69c8a80-115d-11ea-9ee2-0b203bac2930.png)

#### Box plot to visualize outliers in each class based on amount.
![image](https://user-images.githubusercontent.com/41659296/69772525-2abf4e80-115e-11ea-9f4c-0e0930e78bbf.png)


## In this kernel, I train and analyze a variety of models with different pre-processing techiques. The first technique for the unbalanced dataset is Random Undersampling

#### Class distributions before and after random undersampling.
##### Before
![image](https://user-images.githubusercontent.com/41659296/69772662-930e3000-115e-11ea-907a-396d0f90483c.png)

##### After
![image](https://user-images.githubusercontent.com/41659296/69772673-9bff0180-115e-11ea-9625-fc9c0b88dac2.png)


### Analyzing the features using a correlation matrix to see which ones are likely to be important.
![image](https://user-images.githubusercontent.com/41659296/69772826-fe580200-115e-11ea-99b4-171ff0ac0b3d.png)

#### From this correlation matrix, we can see that features V2, V4, V11, and V19 are correlated positvely and that features V10, V12, V14, and V16 are correlated negatively.

### Check out the boxplots of these features.
![image](https://user-images.githubusercontent.com/41659296/69772944-6c9cc480-115f-11ea-85a1-1a6a48b6ec66.png)

#### We can see that these features have a great number of outliers which can inhibit our models accuracy. Some of these outliers will be removed. After calculating the interquartile range (statistical dispersion) by subtracting the 25th lower percentiles from the 75th upper percentiles (quartile75 - quartile25) I add an outlier cutoff value of 1.5 to the range. If any point is lower than the (lower quartile * cutoff), it will be removed. Similarly, if any point is greater than the (upper quartile * 1.5) it will also be removed.

#### Feature V2 contained the highest number of outliers at 46.
![image](https://user-images.githubusercontent.com/41659296/69773244-4af00d00-1160-11ea-8667-a30b9c24a336.png)


### It is a good idea to use some clustering algorithms to indicate whether future predictive models will be accurate.
#### Here are 3 clustering algorithms fit onto the data
![image](https://user-images.githubusercontent.com/41659296/69773366-ac17e080-1160-11ea-8463-8a171c1172a3.png)

#### We see that the T-distributed stochastic neighbor embedding performs the best.

## These are the learning curves of the models after optimizing their hyperparameters.
![image](https://user-images.githubusercontent.com/41659296/69773607-4a0bab00-1161-11ea-850f-ef2b5ec73186.png)

![image](https://user-images.githubusercontent.com/41659296/69773636-5b54b780-1161-11ea-9c07-fcfb72996a8b.png)

![image](https://user-images.githubusercontent.com/41659296/69773577-3e1fe900-1161-11ea-9831-7834d402049f.png)

#### ** Note ** how badly the Random Forest and K Nearest Neighbors classifiers overfit the data.

### Displaying the ROC AUC Curves after cross validation
![image](https://user-images.githubusercontent.com/41659296/69773667-70314b00-1161-11ea-9c25-9ebee78dd742.png)
##### Here we see that Logistic Regression is performing best on the test data.

## I will now implement the second technique - SMOTE Oversampling

#### After training the best logistic regression model from the previous section on the oversampled data, I obtained these results.
![image](https://user-images.githubusercontent.com/41659296/69773870-2137e580-1162-11ea-8582-661e305d67cb.png)


## I now used TensorFlow as a backend to implement two neural networks, each with one hidden layer. The neural nets will be used to see what dataset provides better accuracy (SMOTE Oversampled vs. Random Undersampled)

#### Here is the accuracy on the last few epochs:
##### Random Undersampling
![image](https://user-images.githubusercontent.com/41659296/69774096-e84c4080-1162-11ea-9708-df7913985896.png)
##### SMOTE Oversampling
![image](https://user-images.githubusercontent.com/41659296/69774107-f4380280-1162-11ea-855a-6df633c3bc52.png)

#### We see that the SMOTE Oversampled neural network has a greater accuracy BUT also takes longer to train.

### The results of the neural networks in a confusion matrix form. From the results we see that the SMOTE oversampling technique has much better results than random undersampling. The oversampled model misclassified 40 cases whereas the random undersampling misclassified 2823 cases.
![image](https://user-images.githubusercontent.com/41659296/69774219-4f69f500-1163-11ea-8ecb-abd6d732b52b.png)

## As a final technique, I implemented a Voting Classifier along with Bagging and Pasting Ensemble Classifiers.

#### Here are the results, respectively:
![image](https://user-images.githubusercontent.com/41659296/69774320-a079e900-1163-11ea-9939-03a14c207c1d.png)

![image](https://user-images.githubusercontent.com/41659296/69774325-a66fca00-1163-11ea-839b-7fef0b0bf3f8.png)

![image](https://user-images.githubusercontent.com/41659296/69774332-abcd1480-1163-11ea-9d66-50515354698c.png)

### Please Note that these models were originally created by myself but then used Janio Martinez's work as reference for many of the visualization techniques and process.
