import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_forecast(model, x_test, y_test, index, history , train_sigma , train_mean):
    fig, ax = plt.subplots(3, 1 , figsize = (20,20))
    (pd.Series(history.history['loss'])
                      .plot(style='k',alpha=0.50, title='Loss by Epoch',
                            ax = ax[0], label='loss'))
    (pd.Series(history.history['val_loss'])
                      .plot(style='k',ax=ax[0],label='val_loss'))
    ax[0].legend()
    predicted = model.predict(x_test)
    pd.Series(y_test.reshape(-1), 
              index=index).plot(ax=ax[1], 
                                title='Forecast vs Actual Scaled',
                                label='actual', linestyle = "dashed" , color = "blue")
    pd.Series(predicted.reshape(-1), 
              index=index).plot(label='Forecast', ax=ax[1] , color = "red")
    fig.tight_layout()
    ax[1].legend()
    ################ SCALING BACK ##############
    concat_df = pd.concat([inverse_y(pd.DataFrame(np.squeeze(predicted) , index = index , columns = ["Predicted"]) , train_sigma , train_mean) ,
          inverse_y(pd.DataFrame(np.squeeze(y_test) , index = index , columns = ["Real"]), train_sigma , train_mean) ] , axis = 1)
    
    
    concat_df['Real'].plot(ax=ax[2], 
                            title='Forecast vs Real',
                            label='actual', linestyle = "dashed" , color = "blue")
    concat_df['Predicted'].plot(label='Forecast', ax=ax[2] , color = "red")
    fig.tight_layout()
    ax[2].legend();plt.show()
    return concat_df

def inverse(data , train_sigma , train_mean):
    return (data * train_sigma)+train_mean

def inverse_y(data , train_sigma , train_mean):
    return (data * train_sigma[-1])+train_mean[-1]

def splitter(df):
    columns = df.columns
    X = df[columns[:-1]]
    y = df[columns[-1]]
    X = np.expand_dims(X , axis = -1)
    y = np.expand_dims(y , axis = -1)
    return X,y

def one_step_forecast(df, window):
    d = df.values
    x = []
    n = len(df)
    idx = df.index[:-window]
    for start in range(n-window):
        end = start + window
        x.append(d[start:end])
    cols = [f'x_{i}' for i in range(1, window+1)]
    x = np.array(x).reshape(n-window, -1)
    y = df.iloc[window:].values
    df_xs = pd.DataFrame(x, columns=cols, index=idx)
    df_y = pd.DataFrame(y.reshape(-1), columns=['y'], index=idx)
    return pd.concat([df_xs, df_y], axis=1).dropna()

def plot_outliers(outliers, data, title = "" , figsize = (20,5)):
    
    ax = data.plot(figsize = figsize)

    data.loc[outliers.index].plot(ax=ax, style='rx')
        
    plt.title(f'{title}')
    plt.xlabel('date'); plt.ylabel('Sales')
    plt.legend(['sales','outliers'])
    plt.show()