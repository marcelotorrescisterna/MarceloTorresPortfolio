# Marcelo Torres: Data Scientist Portfolio
Hello and welcome to my personal Data Scientist Portfolio. My name is **Marcelo Torres** and I am a Data Scientist, Industrial Engineer with a Master´s Degree in Finance. I began my journey in Data Science when I wrote my thesis when I was finishing my engineering career back in 2019. I had to work with a dataset of 7 million entries, therefore I was forced to learn Python and its most common libraries for working with data such as Pandas and Numpy. Since then and during all 2020 I started my path on Data Science, focusing my effort on learning Machine Learning and Deep Learning. During the whole year I did a great amount of courses from Coursera and Udemy which gave me a great amount of knowledge. I also did several projects in those courses and some of my own which I intend to show in this portfolio.
## Project Number 1: Titanic Kaggle Competition Project
The Titanic Project is a classic project from Kaggle specially for beginners in the data science world. THe goal of this project is to classify a passenger as a survivor or not. As it´s usual in the world of Data Science, there are several approaches that can lead to a possible solution. For this particular task I decided to work with Neural Networks. As a personal opinion, Neural Networks are one of my favorite tools.  

### Exploratory Data Analysis
Before starting any project, it´s very useful to take a look at the data, this is a very common task known as Exploratory Data Analysis.  

![](/images/Describe.PNG)  ![](/images/DfTitanic.PNG)  

The first thing we can identify is that it´s a very small dataset (as mentioned before, beginners project), with only 891 entries. Some variables are categorical, others only are an id of the instance. Therefore, those features were the first ones to be deleted: Passenger ID and Name. We can also see that Age, Cabin and Embarked have missing values for some instances.

Next, I decided to use some plots to visualize the data in a better way. The first plot shows the target class: 0 Didn´t Survive , 1 Survived. As we can see below, the difference between survivors and non survivors isn´t quite big.  
![](/images/TitanicSurvived.PNG) 

Digging deeper into the survivors, we can identify that people belonging to Pclass 3 (Third Class Ticket) were the ones who mostly died.  
![](/images/TitanicByClass.PNG)

Next up is the correlation matrix. We can see that the most strong positive correlation is between Parch (# of parents / children aboard the Titanic) and SibSp (# of siblings / spouses aboard the Titanic). Also that the most negative correlation is between Pclass and Fare which is quite obvious (Third class ticket is the cheapest).  
![](/images/TitanicCorr.PNG)


### Model Number 1

For this project I decided to run two Neural Network Models. Both of them were build up on features which hadn´t missing values. The features selected where: **Pclass , Sex , SibSp, Parch and Embarked.** Since features Sex, Embarked and Pclass are categorical, they were converted into different features using **Get Dummies** from Pandas. I didn´t use Features Scaling since most of the values were more or less in a similar range.  
 
Using **Keras** the first model was built using a 4 layer network. The first one containing **64 Neurons** , the second one **128 Neurons** , the third **64 Neurons** and one final layer with only **One Neuron** which outputs the class to which the instance belongs. The code to this project is available on my github page. For the training set I used 80% of the data, 10% for validation and 10% for testing. **NOTE: Since Training Test Split uses a random seed to generate this random split, we aren´t going to get the same results**. The results are shown below.  
![](/images/TitanicG11.PNG)![](/images/TitanicG12.PNG)  
![](/images/CorrMat11.PNG) 

As you can see it gets almos 80% accuracy, however when we compare the learning curves, we can see that the model is overfitting the data. To fix this, **Dropout Regularization** was applied. The results are shown below. As you can see, the learning curve is way more smooth thanks to regularization. Confusion matrices are very similar as so as the accuracy obtained. With this model my score was 77% , giving me the ranking in the leaderboard of 4360/16579.  
![](/images/TitanicG21.PNG)![](/images/TitanicG22.PNG)  

![](/images/CorrMat12.PNG)


### Model Number 2

I tried a second model. This time I changed the structure: 3 layers of **32 Neurons** and one output neuron. Regularization was also applied. The results are shown below. Both models gave the same score on the Kaggle Leaderboard.  


## Project Number 2: Itaú Binnario Competition  
This was actually **my first own project and my first competition**. It took place last year during October. This challenge was about predicting if a customer would buy one, more than one or none of a list of five different bank products (A-A, B-B, C-D, D-E , E-E). The competition was sponsored by Binnario.ai and Banco Itaú. Several databases were given: The first one containing 24 million transactions belonging to 79539 differente customers. The second one containing details of different campaigns that were made. The third one containing information about communications with clients and the fourth database information of each customer was detailed. According to the platform, more than 70 different models were submitted. I ended up number 35 on the overall leaderboard.  

### Exploratory Data Analysis
First of all I began with the **Campaign** Database. The plots below show that campaings were focused on 4 out of 5 products, mainly on product A-A. Also, the month in which most of the campaings were made was March 2020.   
![](/images/Campañas1.PNG)![](/images/Campañas2.PNG)  

However, eventhough there was quite a good effort, results show that campaings weren´t effective enough.  
![](/images/Campañas3.PNG)

**Communications** Database was targeted towards all five products and only a small part of customers read the communication.  
![](/images/Comu1.png)![](/images/Comu2.png) 


**Client Information** was a very large database. Therefore as a summary I´m only going to show a couple of features. Starting with the **Age** of customers, there was a great variety. The largest amount of clients belonged to group which had an age between 25 and 30 years old. Customers also have mostly medium-high income levels (R4 to R1) and have been members from the bank between 8 and 9 years.  
![](/images/Consum1.jpg)![](/images/Consum2.jpg) ![](/images/Consumidores3.PNG) 

### Exploratory Data Analysis: Transactions Database  
This was the biggest database of the whole set. As shown in the figure below, most of the transactions involved products A-K and G-K .  
![](/images/Transac1.jpg)

Also, a correlation matrix was built (only with no missing sign values) with the amounts of all the transactions, showing important correlations. For example, product F-H and F-J are perfectly negative correlated. Products, E-F and E-E have a negative correlation of -0.9, similar to products D-F and D-E (-0.96) and products C-D and C-C (-0.9).  

![](/images/CorrMat.JPG)

An important feature from this dataset is the one called **Sign** which indicates if the transaction was positive or negative. However in this feature, 1.439.192 values were missing all belonging to product A-G. As an assumption I decided to consider all these missing values as Positive.


### Model  
For this challenge I decided to try again with a Neural Network (by that time I didn´t know algorithms such as XGBoost which is currently one of the best algorithms). I´m not going to dig deeper into the preprocessing I did with all the data (creating dummy variables, dealing with missing values, etc), but the code is available on my github page. So basically, I tried a whole amount of different combinations trying to find the best model. I also tried with different combinations of features to find the optimal combination. In total I used **52 features**. First of all I added up all the transactions for each product for each user. For example, lets suppose that some user invested 100 in product A-A on January 2020, then withdraw 100 from this product on February 2020 and invested 500 in this product on March 2020. So for this user feature **A-A** would be 500 (100-100+500). The same procedure was used for all the available products (11 products). Also if there was any tipe of communication, the age of the user nad his income category. **NOTE: you can see all the feature engineering and procedures on my github page**. Since the transactions amount was considerable different to other features (for example de categorical ones), **Features Scaling** was necessary to optimize the gradient propagation method.

The best combination I could find was a 3 layer model: The First Layer containing **200 Neurons** , The Second Layer **800 Neurons** and The Third Layer with **5 Neurons** , were each of them outputs 1 if the client buys that product or 0 if it doesn´t. The model was trained on 300 epochs using mini batches of 1000 instances each and a validation split of 20%. The learning curves are shown below.  
![](/images/ModelItau.JPG)


So, eventhough I didn´t reach the top 10, as a first competition I was pretty happy with my own performance. Also because it was the first time that I was facing a real life challenge. In here you can check more information about the competition:  
[Competition Leaderboard](https://binnario-prod.netlify.app/challenge/-MMDsMov6MVyOl3gDuOB)  
[Competition News](https://mundoenlinea.cl/2021/01/13/itau-desarrollara-modelo-para-anticiparse-a-las-proximas-transacciones-de-clientes-bancarios/)


## Project Number 3: Diabetes Prediction Project  
This project is quite simple, but useful to practice algorithms. It´s a small dataset (768 instances) in which 6 features are given for each patient and the target is to predict if it has diabetes or not. This dataset was obtained from [OpenML.com](https://www.openml.org/). The features that were in the dataset were the following: Number of times pregnant, Plasma glucose concentration a 2 hours in an oral glucose tolerance test, Diastolic blood pressure (mm Hg), Triceps skin fold thickness (mm), 2-Hour serum insulin (mu U/ml), Body mass index (weight in kg/(height in m)^2), Diabetes pedigree function and Age (years).  

### Exploratory Data Analysis  
As usual, I started with an EDA using the describe tool from pandas. There were no missing values in the dataset.  
![](/images/DescribeDiab.PNG)


Using Seaborn I managed to visualize other aspects of the dataset. For example that nearly almost a third of the patients had diabetes and also that the proportion of patients which had the disease was bigger in those patients who´s age was in the range 31-33 or 41-45 as shown below.  
![](/images/DiabClass.PNG)![](/images/DiabClass2.PNG)  

Also I created a Correlation Matrix to try and see the behavior between features:  
![](/images/DiabCorr.PNG)  

So, the strongest positive correlation was between Age and number of times pregnant (which it kind of makes sense) , followed by the correlation between skin thickness and insuline levels. 


### Model 1: Logistic Regression  
Since it´s a classification problem, one of the most common models is Logistic Regression. As usual, some preprocessing of the data including One Hot transformation and feature scaling was applied. So, the target value is 1 in case the patient has the disease and 0 if it´s healthy. The results of the model are shown below. The model has a precision of 80% nearly in both classes, however it has 93% recall on class 0 and only 54% on class 1.  
![](/images/ClasMatDiab1.PNG)  
![](/images/ClasRepDiab1.PNG) 

### Model 2: Random Forest  
Another commonly used algorithm is Random Forest which is an ensemble of Decision Trees. The results are shown below:  
![](/images/ClasMatDiab2.PNG)  
![](/images/ClasRepDiab2.PNG)  

After comparing both models, we can see that both of them have a pretty similar accuracy score, however the Logistic Regression model has a more accurate precision score than the Random Forest Classifier. Most of the models for this set have 70% accuracy, you can visit the global models [here](https://www.openml.org/t/37).



## Project Number 4: Implementing LeNet5 For Image Classification  
Coming soon...
