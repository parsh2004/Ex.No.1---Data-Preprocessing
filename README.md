# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
# STEP 1:
 Importing the libraries.
 # STEP 2:
 Importing the dataset.
 # STEP 3:
 Taking care of missing data.
 # STEP 4:
 Encoding categorical data.
 # STEP 5:
 Normalizing the data.
 # STEP 6:
 Splitting the data into test and train.

## PROGRAM:
```
NAME: M PARSHWANATH
REG NO: 212221230073
```
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```
## OUTPUT:
(![image](https://user-images.githubusercontent.com/94222288/228867625-925940c2-568f-4741-9fc8-c3cc881d1747.png)
)
![image](https://user-images.githubusercontent.com/94222288/228867864-1a4d1279-361d-4af2-9504-b32e809f259a.png)
![image](https://user-images.githubusercontent.com/94222288/228867959-0f4ffa0d-5db9-4030-849c-b8886a1df393.png)
![image](https://user-images.githubusercontent.com/94222288/228867998-5b7abb44-3371-4b44-a749-115ae88e9c21.png)
![image](https://user-images.githubusercontent.com/94222288/228868208-f8a0dc3e-fb21-4eaa-a6d7-e7425222e9da.png)
![image](https://user-images.githubusercontent.com/94222288/228868262-53149daa-630d-4cfc-8ebe-eb4a80f29119.png)
![image](https://user-images.githubusercontent.com/94222288/228868316-987bc63d-1d0f-411e-a51f-af85432f68ce.png)
![image](https://user-images.githubusercontent.com/94222288/228868393-f19ae194-52ec-4ac6-829f-4d0e1d9727f0.png)
![image](https://user-images.githubusercontent.com/94222288/228868448-fcf84473-2398-4b05-bd80-0f709b5f1f72.png)

## RESULT
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully..
