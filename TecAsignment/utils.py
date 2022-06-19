"""
Utils Notebook for Technical Assignment

Marcelo Torres Cisterna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def Countplot(data , feature , title , palette = "autumn" , figsize = (15,4)):
    """
    Plots a countplot for a particular features given a dataset
    
    ARGS :
        data : dataframe containing the data
        feature : desired feature
        title : plot title
        palette : desired palette
        figsize = size of the plot
    
    """
    plt.figure(figsize = figsize)
    plt.title(title)
    sns.countplot(x = feature , data = data , palette = palette);
    
def ComparativePlot(data1, data2 , feature ,title1 ,title2, globaltitle = " ", palette1 = "autumn" , palette2 = "winter" , order = None):
    """ 
    Plots a side by side comparison of countplots for a particular feature
    
    ARGS :
        data1 : first dataset to compare
        data2 : second dataset to compare
        feature : feature to compare
        title1 : title for first countplot
        title2 : title for second countplot
        globaltitle : title for whole plot
        
    """
    fig, axes = plt.subplots(1,2 , figsize = (20,4))
    fig.suptitle(globaltitle, fontsize=14)
    axes[0].title.set_text(title1)
    axes[1].title.set_text(title2)
    sns.countplot(x = feature , data = data1 , palette = palette1 , ax = axes[0] , order = order);
    sns.countplot(x = feature , data = data2 , palette = palette2 , ax = axes[1] , order = order);
    
def FeatureRatio(data , feature , horizon , target = "y" , posclass = "yes" , negclass = "no" , plot = True , plot_title = " " , returndata = False):
    """
    Calculate Ratio for a particular class for a particular feature
    
    ARGS:
        data : Dataframe with the required data
        feature : feature in which we will calculate the ratio
        horizon : horizon of the feature. Example for days we have [monday, tuesday...]
        target : target feature which separates the classes
        posclass : positive class
        negclass : negative class
        plot : True if you want to plot de answers
        plot_title : Title for the plot
        returndata : Return ratio dataframe
    """
    grouped_data = pd.DataFrame(data.groupby([feature , target]).size())
    grouped_data.columns = ["total"]
    
    ratiodf = []
    for i in horizon:
        pos = grouped_data.loc[i].loc[posclass]
        total = grouped_data.loc[i].loc[posclass] + grouped_data.loc[i].loc[negclass]
        ratio = np.round((pos/total).values * 100 , 2)[0]
        d = {
            feature : i,
            'ratio' : ratio
        }
        ratiodf.append(d)
    ratiodf = pd.DataFrame(ratiodf)
    if plot:
        fig, ax = plt.subplots(figsize = (15,4))
        ax.set_title(plot_title)
        fig.canvas.draw()
        tick_spacing = 1
        labels = [horizon[0]] + horizon
        ax.plot(ratiodf.ratio , '-bo')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Ratio")
        ax.set_xlabel(f"{feature}")
        plt.show()
    if returndata:
        return ratiodf
    
def make_bins(df , label_names , cut_points , binnedcol , maincol):
    """
    Used for making bins of different features
    
    ARGS:
        df : DataFrame containing the data
        label_names : Desired Bins
        cut_points : Intervals for splitting the data
        binnedcol : name of the new binned column
        maincol : main column to bin
    """
    label_names = label_names
    cut_points = cut_points
    df[binnedcol] = pd.cut(df[maincol], cut_points, labels=label_names)
    return df
    
def make_bins_categorical(data , category_range , feature , category_name):
    """
    Used for making bins of different features with string name
    
    ARGS:
        data: DataFrame containing the data
        category_range : Categories in which the feature must be 
        feature: Specific feature
        category_name : New name for the category
    """
    binned = []
    for i in data[feature] :
        if i in category_range:
            binned.append(1)
        else:
            binned.append(0)
    data[category_name] = binned
    return data

def fillerFunction(data , feature):
    unknownset = data[data[feature] == "unknown"]
    knownset = data[data[feature] != "unknown"]
    filtervec = []
    for i in unknownset.itertuples():
        edad = i.age
        job = i.job
        temp_data = knownset[(knownset["age"] == edad)]
        filtervec.append(temp_data.groupby(feature).size().sort_values(ascending = False).index[0])    
    newFeature = f"new{feature}"
    unknownset[newFeature] = filtervec
    knownset[newFeature] = knownset[feature]
    data = pd.concat([unknownset , knownset])
    data.drop(feature, axis = 1 , inplace = True)
    return data
