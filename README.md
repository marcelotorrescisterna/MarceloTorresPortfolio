# Marcelo Torres: Data Scientist Portfolio
Hello and welcome to my personal Data Scientist Portfolio. My name is **Marcelo Torres** and I am a Data Scientist, Industrial Engineer with a Master´s Degree in Finance. I began my journey in Data Science when I wrote my thesis when I was finishing my engineering career back in 2019. I had to work with a dataset of 7 million entries, therefore I was forced to learn Python and it´s most common libraries for working with data such as Pandas and Numpy. Since then and during all 2020 I started my path on Data Science, focusing my effort on learning Machine Learning and Deep Learning. During the whole year I did a great amount of courses from Coursera and Udemy which gave me a great amount of knowledge. I also did several projects in those courses and some of my own which I intend to show in this portfolio.
## Project Number 1: Titanic Kaggle Competition Project
The Titanic Project is a classic project from Kaggle specially for beginners in the data science world. THe goal of this project is to classify if a person as a survivor or not. As it´s usual in the world of Data Science, there are several approaches that can lead to a possible solution. For this particular task I decided to work with Neural Networks. As a personal opinion, Neural Networks are one of my favorite tools.  

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
![](/images/TitanicG11.PNG)![](/images/TitanicG12.PNG) 

Using **Keras** the first model was built using a 4 layer network. The first one containing **64 Neurons** , the second one **128 Neurons** , the third **64 Neurons** and one final layer with only **One Neuron** which outputs the class to which the instance belongs. The code to this project is available on my github page. For the training set I used 80% of the data, 10% for validation and 10% for testing. **NOTE: Since Training Test Split uses a random seed to generate this random split, we aren´t going to get the same results**. The results are shown below. As you can see it gets almos 80% accuracy, however when we compare the learning curves, we can see that the model is overfitting the data. To fix this, **Dropout Regularization** was applied. The results are shown below. As you can see, the learning curve is way more smooth thanks to regularization. Confusion matrices are very similar as so as the accuracy obtained. With this model my score was 77% , giving me the ranking in the leaderboard of 4360/16579. 

### Model Number 2

I tried a second model. This time I changed the structure: 3 layer of **32 Neurons** and one output neuron. Regularization was also applied. The results are shown below. Both models gave the same score on the Kaggle Leaderboard.

## Project Number 2: Diabetes Prediction Project
## Project Number 3: Itau Binnario Competition 
## Project Number 4: Implementing LeNet5 For Image Classification
