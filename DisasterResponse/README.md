# Disaster Response

The motivation of this proyect was to create two pipelines: one for ETL and one for ML Training and build a disaster message clasifier. The ML model outputs the probability of the message belonging to one or more of 36 different categories. Finally, results are displayed in different dashboards using Flask. The dataset used was a series of disaster messages provided by Figure Eight, a company specialized in working with nlp and disaster tasks.

## Files

There is a winrar file containing all the files used in this proyect. First of all unzip the winrar file. You´ll find 3 folders:

1) __App Folder :__ contains all the necessary scripts to build de web application and deploy the results from the ML model.
2) __Data__ : Contains the dataset used to train the ML Algorithm which are Messages and Categories (to which the messages belong to). Also it has a __process_data__ file in which the preprocessing can be automaticaly done.
3) __Models__: Contains the script used to ingest data and train an ML Model

