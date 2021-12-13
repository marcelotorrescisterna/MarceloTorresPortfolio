# Disaster Response

The motivation of this proyect was to create two pipelines: one for ETL and one for ML Training and build a disaster message clasifier. The ML model outputs the probability of the message belonging to one or more of 36 different categories. Finally, results are displayed in different dashboards using Flask. The dataset used was a series of disaster messages provided by Figure Eight, a company specialized in working with nlp and disaster tasks.

## Files

There is a winrar file containing all the files used in this proyect. First of all unzip the winrar file. You´ll find 3 folders:

1) __App Folder :__ contains all the necessary scripts to build de web application and deploy the results from the ML model.
2) __Data__ : Contains the dataset used to train the ML Algorithm which are Messages and Categories (to which the messages belong to). Also it has a __process_data__ file in which the preprocessing can be automaticaly done.
3) __Models__: Contains the script used to ingest data and train an ML Model

## Installations
Everything is specified in the .py files. A detailed explanation is shown below. Once you extract all the files:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements 

* Figure Eight: For Providing the Dataset
* Udacity Nanodegree Team: For explaining Pipelines with pandas, nlp and flask

