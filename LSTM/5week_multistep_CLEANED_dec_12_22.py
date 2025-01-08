# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:06:19 2022
@author: zerb

Description:
- This is the cleaned version of the 5 week multistep forecast using LSTM
- This code is used in tandem with "split_sequences_multi" to split the data to feed into the LSTM model

Actions:
- Parameters that can be changed before script run
- Look in sections "Variables" and "Load Data" for all actions
"""


#IMPORT LIBRARIES##############################################################
import numpy as np
import pandas as pd
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import split_sequences_multi as ss
np.random.seed(1234)
random.seed(1234)

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(1234)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


################################################################################
# SET PARAMETERS
################################################################################

# Set date
date = datetime.now().strftime("%Y_%m_%d-%I%M%p")

#STUDY REGION
#ACTIONS: Set hexel number, confusion matrix threshold, and number of epochs
study_unit = "hexel" # hexel / ecoregion
hexel = "17" # which hexel to use (see map)
#ecoregion = "17" # specify what ecoregion to use
response="BUI"
perc_test=0.95 #confusion matrix threshold.
n_epochs =500
save_output = True
normalization = True

#MODEL INPUT SELECTION
#ACTIONS: Select model inputs to run. Change "modelnum" according to selected model inputs.
modelnum='1'
inputs = ["BUI", "week","BUI_climatology"] #MODEL 1
#inputs = ["BUI", "week","BUI_climatology","ENSOMEIv2_fft6M_weekly"] #MODEL 2
#inputs = ["BUI", "week","BUI_climatology","AMO_fft6M_weekly"] #MODEL 3
#inputs = ["BUI", "week","BUI_climatology","ENSOMEIv2_fft6M_weekly","AMO_fft6M_weekly"] #MODEL 4

#MODEL TIME PERIOD
weeks=list(range(13,43)) # only test fire season weeks.
n_steps_in = 12
n_steps_out = 5

#MODEL TRAINING AND TESTING INTERVALS
train_year_start = 1981 #based on 80-20 ration
train_year_end = 2012 #based on 80-20 ratio
test_year_start = train_year_end+1
test_year_end = 2020

# CHANGE PATHS AS NEEDED:

# paths on macbook
#path1 = "/Users/piyush/RESEARCH/PROJECTS/Seasonal forecasting/AMII test/data/era5land_BUI_DC_monthly_ez/" #PATH TO ERA5 DATA
#path2 = "/Users/piyush/RESEARCH/PROJECTS/Seasonal forecasting/AMII test/data/updated_teleconnection/" #PATH TO WEEKLY (INTERPOLATED) TELECONNECTION
#image_output_path = "/Users/piyush/RESEARCH/PROJECTS/Seasonal forecasting/AMII test/data/output_data/images/" #PATH TO STORE PLOTS
#table_output_path = "/Users/piyush/RESEARCH/PROJECTS/Seasonal forecasting/AMII test/data/output_data/" #PATH TO STORE RESULTS TABLES

path1 = r'C:\Users\kzammit\Documents\LSTM-JZ\data\era5land_BUI_DC_monthly_ez'
path2 = r'C:\Users\kzammit\Documents\LSTM-JZ\data\updated_teleconnection'
image_output_path = r'C:\Users\kzammit\Documents\LSTM-JZ\data\output_data\images'
table_output_path = r'C:\Users\kzammit\Documents\LSTM-JZ\data\output_data\tables'

################################################################################
# DO NOT EDIT BELOW HERE
################################################################################

#PARAMETERS FOR REPORTING
param = {}
if study_unit == "ecoregion":
    param["ecoregion"] = ecoregion
elif study_unit == "hexel":
    param["hexel"] = hexel
param["n_steps_in"] = n_steps_in
param["n_steps_out"] = n_steps_out
param["n_epochs"] = n_epochs
param["response"] = response
param["inputs"] = inputs
param["train_year_start"] = train_year_start
param["train_year_end"] = train_year_end
param["test_year_start"] = test_year_start
param["test_year_end"] = test_year_end

#LOAD DATA

if study_unit == "ecoregion":
    data = pd.read_csv(path2 + '\\' + ''"BUI_covariates_merged_weekly_ez_" + str(hexel) + "_smoothed" + ".csv")
elif study_unit == "hexel":
    data = pd.read_csv(path2 + '\\' + "BUI_covariates_merged_weekly_hx_" + str(hexel) + "_smoothed" + ".csv")

data = data.replace(np.nan, 0)
data = data[data['year'] >= train_year_start]
data = data[data['year'] <= test_year_end]

train_start = data.index[(data['year'] == train_year_start) & (data['week'] == 1)][0]
train_end = data.index[(data['year'] == train_year_end) & (data['week'] == 52)][0]
test_start = data.index[(data['year'] == test_year_start) & (data['week'] ==1)][0]
test_end = data.index[(data['year'] == test_year_end) & (data['week'] ==52)][0]

################################################################################
#TRAINING AND MODEL BUILDING
################################################################################


bui_clim = data.iloc[train_start:train_end+1].groupby("week").mean().BUI
bui_clim=np.tile(np.array(bui_clim), test_year_end-train_year_start+1)
data["BUI_climatology"] = bui_clim
data["BUI_anom"] = data["BUI"] - data["BUI_climatology"]

#SOME PLOTS TO TEST/LOOK AT DATA
plt.figure(figsize=(15,5))
data['date'] = "1-" + data['week'].map(str)+ '-' +data['year'].map(str)
plt.plot(data['date'], data["BUI_climatology"],color='green', label="BUI climatology")
plt.plot(data['date'][train_start:train_end+1], data["BUI"][train_start:train_end+1],'c',label="BUI train")
plt.plot(data['date'][test_start:test_end+1], data["BUI"][test_start:test_end+1], color="red",label="BUI test")
plt.title(str(train_year_start) + "-" + str(test_year_end),size=12)
plt.xlabel("Year",size=12)
plt.ylabel("BUI",size=12)
yearsFmt = mdates.DateFormatter('%Y')
yearsLoc = mdates.YearLocator(5)
monthsLoc = mdates.MonthLocator()
ax = plt.gca()
ax.xaxis.set_major_locator(yearsLoc)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(monthsLoc)
plt.legend()
plt.show(block=False)
if save_output:
    plt.savefig(image_output_path + "/" + "BUI_timeseries_all.pdf", dpi=600, bbox_inches='tight')
    plt.close()

#BUILD DATASET ARRAY
dataset = np.asarray(data[[*inputs, response]])

if normalization:
    #NORMALIZATION OF DATA
    scaler_inputs = MinMaxScaler()
    scaler_response = MinMaxScaler()

    #TRANSFORM DATA
    dataset_scaled = dataset.copy()
    dataset_scaled[:,0:len(inputs)] = scaler_inputs.fit_transform(dataset_scaled[:,0:len(inputs)])
    dataset_scaled[:,-1] = scaler_response.fit_transform(dataset_scaled[:,-1][:,np.newaxis]).flatten()
    dataset_train = dataset_scaled[train_start:train_end+1]
    dataset_test = dataset_scaled[test_start:test_end+1]

else:
    dataset_train = dataset[train_start:train_end+1]
    dataset_test = dataset[test_start:test_end+1]

dataset_train = dataset[train_start:train_end+1]
dataset_test = dataset[test_start:test_end+1]

#SPLIT INTO SAMPLES
X, y = ss.split_sequences_multi(dataset_train, n_steps_in, n_steps_out)
n_features = X.shape[2] # reshape from [samples, timesteps] into [samples, timesteps, features]
Xtest, ytest = ss.split_sequences_multi(dataset_test, n_steps_in, n_steps_out)

################################################################################
# MAIN MODEL
################################################################################

#DEFINE MODEL ARCHITECTURE
model = Sequential()
model.add(LSTM(300, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.10))
model.add(Dense(20)) # Piyush - I changed this from Dense(n_steps_out) as seemed to give lower loss.
model.add(Dense(n_steps_out,activation='relu'))
optimizer = tf.keras.optimizers.Adam(0.00005)
model.compile(optimizer=optimizer, loss='mse')

#FIT MODEL
history = model.fit(X, y, epochs=n_epochs, verbose=1, validation_data=(Xtest, ytest))

#TRAINING LOSS PLOTS
n_epochs = len(history.history['loss'])
loss_train = history.history['loss']
epochs = range(1,n_epochs+1)
plt.figure()
plt.plot(epochs, loss_train, 'g', label='Training Loss')
if "val_loss" in history.history.keys():
    loss_val = history.history['val_loss']
    plt.plot(epochs, loss_val, 'b', label='Validation Loss')
    plt.title('Model '+str(modelnum)+ ' Training and Validation Loss Curves')
else:
    plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
if save_output:
    plt.savefig(image_output_path + "/" + 'hexel' + str(hexel) + 'model' + str(modelnum) + "_traininglosscurves.png", bbox_inches="tight", dpi=100)
    plt.close()
else:
    plt.show(block=True)

#PREDICTION: RUN THE MODEL WITH TESTING DATA
yhat_bui = model.predict(Xtest, verbose=0)
if normalization:
    ypredict = scaler_response.inverse_transform(yhat_bui)
print(model.summary())

################################################################################
#PLOTTING
################################################################################

#SUBSET TO FIRE SEASON FOR TESTING
indices_test = [True if i in weeks else False for i in data[test_start:test_end+1].week]
indices_test[0:n_steps_in] = [*np.repeat(False, n_steps_in)]
df_results = data.iloc[test_start:test_end+1,:].copy()

results_to_add1 = np.array([*np.repeat(np.nan, n_steps_in), *yhat_bui[:,0].flatten()])
results_to_add2 = np.array([*np.repeat(np.nan, n_steps_in), *yhat_bui[:,1].flatten()])  #THIS ONE TURNS ON AND OFF
results_to_add3 = np.array([*np.repeat(np.nan, n_steps_in), *yhat_bui[:,2].flatten()]) #THIS ONE TURNS ON AND OFF
results_to_add4 = np.array([*np.repeat(np.nan, n_steps_in), *yhat_bui[:,3].flatten()]) #THIS ONE TURNS ON AND OFF
results_to_add5 = np.array([*np.repeat(np.nan, n_steps_in), *yhat_bui[:,4].flatten()]) #THIS ONE TURNS ON AND OFF

results_to_add1 = np.append(results_to_add1, [0,0,0,0]) #for 1 week prediction
results_to_add2 = np.append(results_to_add2, [0,0,0,0]) #for 2 weeks prediction   #THIS ONE TURNS ON AND OFF
results_to_add3 = np.append(results_to_add3, [0,0,0,0]) #for 3 weeks prediction   #THIS ONE TURNS ON AND OFF
results_to_add4 = np.append(results_to_add4, [0,0,0,0]) #for 4 weeks prediction   #THIS ONE TURNS ON AND OFF
results_to_add5 = np.append(results_to_add5, [0,0,0,0]) #for 5 weeks prediction   #THIS ONE TURNS ON AND OFF

df_results['BUI_predict_1week'] =  results_to_add1
df_results['BUI_predict_2week'] =  results_to_add2    #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_3week'] =  results_to_add3   #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_4week'] =  results_to_add4   #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_5week'] =  results_to_add5   #THIS ONE TURNS ON AND OFF


BUI_predict_2week =  df_results['BUI_predict_2week'].shift(+1)
df_results['BUI_predict_2week'] =  BUI_predict_2week
BUI_predict_3week =  df_results['BUI_predict_3week'].shift(+2)   #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_3week'] =  BUI_predict_3week  #THIS ONE TURNS ON AND OFF
BUI_predict_4week =  df_results['BUI_predict_4week'].shift(+3)   #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_4week'] =  BUI_predict_4week  #THIS ONE TURNS ON AND OFF
BUI_predict_5week =  df_results['BUI_predict_5week'].shift(+4)   #THIS ONE TURNS ON AND OFF
df_results['BUI_predict_5week'] =  BUI_predict_5week  #THIS ONE TURNS ON AND OFF

bui_persistance = df_results['BUI'].shift(+1).replace(np.nan,0)
bui_persistance2 = df_results['BUI'].shift(+2).replace(np.nan,0)
bui_persistance3 = df_results['BUI'].shift(+3).replace(np.nan,0)
bui_persistance4 = df_results['BUI'].shift(+4).replace(np.nan,0)
bui_persistance5 = df_results['BUI'].shift(+5).replace(np.nan,0)

df_results['bui_persistance']=bui_persistance
df_results['bui_persistance2']=bui_persistance2
df_results['bui_persistance3']=bui_persistance3
df_results['bui_persistance4']=bui_persistance4
df_results['bui_persistance5']=bui_persistance5


#PLOT #2 FOR THESIS "MODEL VS PERSISTENCE FORECAST"#############################################################

for lead in np.arange(1,n_steps_out+1):

    #PLOT: CLIMATOLOGY 1 WEEK FORECAST
    plt.figure()
    plt.plot(df_results["BUI"], df_results["BUI_climatology"],"gx",label="Climatology Forecast")
    plt.plot(df_results["BUI"], df_results["BUI_predict_" + str(lead) + "week"],".",color='red',label="Model Forecast")
    plt.title(str(lead) + " Week: Model Versus Climatology Forecast",size=12)
    #plt.title(str(test_year_start) + "-" + str(test_year_end)+ " Climatological and Predicted BUI Versus Observations",size=12)
    plt.xlabel("Observed BUI",size=12)
    plt.ylabel("Forecast BUI",size=12)
    xscale = np.linspace(0, df_results["BUI"].max(), 101)
    plt.plot(xscale,xscale,'c',label="Observed")
    plt.legend()
    if save_output:
        plt.savefig(image_output_path + "/" + 'hexel' + str(hexel) + 'model' + str(modelnum) + "week" + str(lead) + "_clim_graph2.png", bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show(block=True)

    #PLOT: PERSISTENCE 1 WEEK FORECAST
    plt.figure()
    plt.plot(df_results["BUI"], df_results["bui_persistance"],"gx",label="Persistence Forecast")
    plt.plot(df_results["BUI"], df_results["BUI_predict_" + str(lead) + "week"],".",color='red',label="Model Forecast")
    plt.title(str(lead) + " Week: Model Versus Persistence Forecast",size=12)
    #plt.title(str(test_year_start) + "-" + str(test_year_end)+ " Climatological and Predicted BUI Versus Observations",size=12)
    plt.xlabel("Observed BUI",size=12)
    plt.ylabel("Forecast BUI",size=12)
    xscale = np.linspace(0, df_results["BUI"].max(), 101)
    plt.plot(xscale,xscale,'c',label="Observed")
    plt.legend()
    if save_output:
        plt.savefig(image_output_path + "/" + 'hexel' + str(hexel) + 'model' + str(modelnum) + "week" + str(lead) + "_pers_graph2.png", bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show(block=True)


#PLOT #3 FOR THESIS "MODEL AND PERSISTENCE FORECAST ANOMALIES" #############################################################

for lead in np.arange(1,n_steps_out+1):

    #PLOT: CLIMATOLOGY 1 WEEK FORECAST
    plt.figure()
    plt.plot(df_results["week"], df_results["BUI_climatology"]-df_results["BUI"],"gx",label="Climatology Forecast")
    plt.plot(df_results["week"],df_results["BUI_predict_" + str(lead) + "week"]-df_results["BUI"],".",color='red',label="Model Forecast")
    plt.plot(df_results["week"], df_results["BUI"]-df_results["BUI"],"c",label="Observed")
    plt.title(str(lead) + " Week: Model and Climatology Forecast Anomalies",size=12)
    plt.xlabel("Week of Year",size=12)
    plt.ylabel("Forecast Anomalies",size=12)
    xscale = np.linspace(0, df_results["BUI"].max(), 101)
    plt.legend()
    if save_output:
        plt.savefig(image_output_path + "/" + 'hexel' + str(hexel) + 'model' + str(modelnum) + "week" + str(lead) + "_clim_graph3.png", bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show(block=True)

    #PLOT: PERSISTENCE 1 WEEK FORECAST
    plt.figure()
    plt.plot(df_results["week"], df_results["bui_persistance"]-df_results["BUI"],"gx",label="Persistence Forecast")
    plt.plot(df_results["week"],df_results["BUI_predict_" + str(lead) + "week"]-df_results["BUI"],".",color='red',label="Model Forecast")
    plt.plot(df_results["week"], df_results["BUI"]-df_results["BUI"],"c",label="Observed")
    plt.title(str(lead) + " Week: Model and Persistence Forecast Anomalies",size=12)
    plt.xlabel("Week of Year",size=12)
    plt.ylabel("Forecast Anomalies",size=12)
    xscale = np.linspace(0, df_results["BUI"].max(), 101)
    plt.legend()
    if save_output:
        plt.savefig(image_output_path + "/" + 'hexel' + str(hexel) + 'model' + str(modelnum) + "week" + str(lead) + "_pers_graph3.png", bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show(block=True)


################################################################################
#ERROR METRICS
################################################################################

#MSE CALCULATIONS
climatology_MSE1 = np.nanmean(np.power(df_results["BUI_climatology"]- df_results["BUI"],2))
multivariate_MSE1 = np.nanmean(np.power(df_results["BUI_predict_1week"]- df_results["BUI"],2))
multivariate_MSE2 = np.nanmean(np.power(df_results["BUI_predict_2week"]- df_results["BUI"],2))
multivariate_MSE3 = np.nanmean(np.power(df_results["BUI_predict_3week"]- df_results["BUI"],2))
multivariate_MSE4 = np.nanmean(np.power(df_results["BUI_predict_4week"]- df_results["BUI"],2))
multivariate_MSE5 = np.nanmean(np.power(df_results["BUI_predict_5week"]- df_results["BUI"],2))

persistance_MSE1 = np.nanmean(np.power(df_results["bui_persistance"]- df_results["BUI"],2))
persistance_MSE2 = np.nanmean(np.power(df_results["bui_persistance2"]- df_results["BUI"],2))
persistance_MSE3 = np.nanmean(np.power(df_results["bui_persistance3"]- df_results["BUI"],2))
persistance_MSE4 = np.nanmean(np.power(df_results["bui_persistance4"]- df_results["BUI"],2))
persistance_MSE5 = np.nanmean(np.power(df_results["bui_persistance5"]- df_results["BUI"],2))

print("Multivariate MSE1:",multivariate_MSE1)
print("Multivariate MSE2:",multivariate_MSE2)
print("Multivariate MSE3:",multivariate_MSE3)
print("Multivariate MSE4:",multivariate_MSE4)
print("Multivariate MSE5:",multivariate_MSE5)

print("Climatology MSE:",climatology_MSE1)
print("Persistence MSE1:",persistance_MSE1)
print("Persistence MSE2:",persistance_MSE2)
print("Persistence MSE3:",persistance_MSE3)
print("Persistence MSE4:",persistance_MSE4)
print("Persistence MSE5:",persistance_MSE5)

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#MAE CALCULATIONS
climatology_MAE1 = np.nanmean(np.abs(df_results["BUI_climatology"]- df_results["BUI"]))
multivariate_MAE1 = np.nanmean(np.abs(df_results["BUI_predict_1week"]- df_results["BUI"]))
multivariate_MAE2 = np.nanmean(np.abs(df_results["BUI_predict_2week"]- df_results["BUI"]))
multivariate_MAE3 = np.nanmean(np.abs(df_results["BUI_predict_3week"]- df_results["BUI"]))
multivariate_MAE4 = np.nanmean(np.abs(df_results["BUI_predict_4week"]- df_results["BUI"]))
multivariate_MAE5 = np.nanmean(np.abs(df_results["BUI_predict_5week"]- df_results["BUI"]))

persistance_MAE1 = np.nanmean(np.abs(df_results["bui_persistance"]- df_results["BUI"]))
persistance_MAE2 = np.nanmean(np.abs(df_results["bui_persistance2"]- df_results["BUI"]))
persistance_MAE3 = np.nanmean(np.abs(df_results["bui_persistance3"]- df_results["BUI"]))
persistance_MAE4 = np.nanmean(np.abs(df_results["bui_persistance4"]- df_results["BUI"]))
persistance_MAE5 = np.nanmean(np.abs(df_results["bui_persistance5"]- df_results["BUI"]))

print("Multivariate MAE1:",multivariate_MAE1)
print("Multivariate MAE2:",multivariate_MAE2)
print("Multivariate MAE3:",multivariate_MAE3)
print("Multivariate MAE4:",multivariate_MAE4)
print("Multivariate MAE5:",multivariate_MAE5)

print("Climatology MAE:",climatology_MAE1)
print("Persistence MAE1:",persistance_MAE1)
print("Persistence MAE2:",persistance_MAE2)
print("Persistence MAE3:",persistance_MAE3)
print("Persistence MAE4:",persistance_MAE4)
print("Persistence MAE5:",persistance_MAE5)

#MBE CALCULATIONS
climatology_MBE1 = np.nanmean(df_results["BUI_climatology"]-df_results["BUI"])
multivariate_MBE1 = np.nanmean(df_results["BUI_predict_1week"]-df_results["BUI"])
multivariate_MBE2 = np.nanmean(df_results["BUI_predict_2week"]-df_results["BUI"])
multivariate_MBE3 = np.nanmean(df_results["BUI_predict_3week"]-df_results["BUI"])
multivariate_MBE4 = np.nanmean(df_results["BUI_predict_4week"]-df_results["BUI"])
multivariate_MBE5 = np.nanmean(df_results["BUI_predict_5week"]-df_results["BUI"])

persistance_MBE1 = np.nanmean(df_results["bui_persistance"]-df_results["BUI"])
persistance_MBE2 = np.nanmean(df_results["bui_persistance2"]-df_results["BUI"])
persistance_MBE3 = np.nanmean(df_results["bui_persistance3"]-df_results["BUI"])
persistance_MBE4 = np.nanmean(df_results["bui_persistance4"]-df_results["BUI"])
persistance_MBE5 = np.nanmean(df_results["bui_persistance5"]-df_results["BUI"])

print("Multivariate MBE1:",multivariate_MBE1)
print("Multivariate MBE2:",multivariate_MBE2)
print("Multivariate MBE3:",multivariate_MBE3)
print("Multivariate MBE4:",multivariate_MBE4)
print("Multivariate MBE5:",multivariate_MBE5)

print("Climatology MBE:",climatology_MBE1)
print("Persistence MBE1:",persistance_MBE1)
print("Persistence MBE2:",persistance_MBE2)
print("Persistence MBE3:",persistance_MBE3)
print("Persistence MBE4:",persistance_MBE4)
print("Persistence MBE5:",persistance_MBE5)

#CONFUSTION MATRICES###########################################################

# use 75th percentile of BUI as threshold for categories
perc_75=df_results.BUI.quantile(perc_test)
print(perc_75)

df_copy = df_results.copy()
df_copy["BUI_predict_1week"] = df_copy["BUI_predict_1week"].fillna(0)
df_copy["BUI_predict_2week"] = df_copy["BUI_predict_2week"].fillna(0)
df_copy["BUI_predict_3week"] = df_copy["BUI_predict_3week"].fillna(0)
df_copy["BUI_predict_4week"] = df_copy["BUI_predict_4week"].fillna(0)
df_copy["BUI_predict_5week"] = df_copy["BUI_predict_5week"].fillna(0)

df_copy.loc[df_copy['BUI'] < perc_75, 'BUI'] = 0
df_copy.loc[df_copy['BUI'] >= perc_75, 'BUI'] = 1
df_copy.loc[df_copy['BUI_predict_1week'] < perc_75, 'BUI_predict_1week'] = 0
df_copy.loc[df_copy['BUI_predict_1week'] >= perc_75, 'BUI_predict_1week'] = 1
df_copy.loc[df_copy['BUI_predict_2week'] < perc_75, 'BUI_predict_2week'] = 0
df_copy.loc[df_copy['BUI_predict_2week'] >= perc_75, 'BUI_predict_2week'] = 1
df_copy.loc[df_copy['BUI_predict_3week'] < perc_75, 'BUI_predict_3week'] = 0
df_copy.loc[df_copy['BUI_predict_3week'] >= perc_75, 'BUI_predict_3week'] = 1
df_copy.loc[df_copy['BUI_predict_4week'] < perc_75, 'BUI_predict_4week'] = 0
df_copy.loc[df_copy['BUI_predict_4week'] >= perc_75, 'BUI_predict_4week'] = 1
df_copy.loc[df_copy['BUI_predict_5week'] < perc_75, 'BUI_predict_5week'] = 0
df_copy.loc[df_copy['BUI_predict_5week'] >= perc_75, 'BUI_predict_5week'] = 1

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

conf_mat1 = confusion_matrix(df_copy['BUI'], df_copy['BUI_predict_1week'])
conf_mat2 = confusion_matrix(df_copy['BUI'], df_copy['BUI_predict_2week'])
conf_mat3 = confusion_matrix(df_copy['BUI'], df_copy['BUI_predict_3week'])
conf_mat4 = confusion_matrix(df_copy['BUI'], df_copy['BUI_predict_4week'])
conf_mat5 = confusion_matrix(df_copy['BUI'], df_copy['BUI_predict_5week'])

#PLOTTING THE CONFUSION MATRICES
# Week1 Confusion Matrix Forecast
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat1.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat1.flatten()/np.sum(conf_mat1)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
import seaborn as sns
ax = sns.heatmap(conf_mat1, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix 1 Week Forecast\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_ylim([0,2])
if save_output:
    plt.savefig(image_output_path + '\\' + "1weekimage_confmatrix_hexel" + str(hexel) + "_" + date + ".png")
    plt.close()
else:
    plt.show(block=True)

#Classification Report of the Confusion Matrix Test
class_rep1 = classification_report(df_copy['BUI'], df_copy['BUI_predict_1week'],labels=[1,0],output_dict=True)
classrep1 = pd.DataFrame(class_rep1).transpose()
print('Classification report : \n',classrep1)
classrep1.to_csv(table_output_path + '\\' + "1week_forecast_confmatrix_hexel" + str(hexel) + "_" + date + ".csv")

classrep1_f1score=classrep1['f1-score']
classrep1_accuracy=classrep1_f1score[2]

#SENSITIVITY, SPECIFICITY, PPV, NPV
sen1=conf_mat1[1,1]/(conf_mat1[1,1]+conf_mat1[1,0]) *100
spec1=conf_mat1[0,0]/(conf_mat1[0,0]+conf_mat1[0,1]) *100
ppv1=conf_mat1[1,1]/(conf_mat1[1,1]+conf_mat1[0,1]) *100
npv1=conf_mat1[0,0]/(conf_mat1[0,0]+conf_mat1[1,0]) *100
print("Sensitivity/Recall =",sen1,"Specificity/Selecitiviy =",spec1,"PPV/Precision =",ppv1,"NPV =",npv1)

# Week2 Confusion Matrix Forecast
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat2.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat2.flatten()/np.sum(conf_mat2)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
import seaborn as sns
ax = sns.heatmap(conf_mat2, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix 2 Week Forecast\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_ylim([0,2])
if save_output:
    plt.savefig(image_output_path + '\\' + "2weekimage_confmatrix_hexel" + str(hexel) + "_" + date + ".png")
    plt.close()
else:
    plt.show(block=True)

#Classification Report of the Confusion Matrix Test
class_rep2 = classification_report(df_copy['BUI'], df_copy['BUI_predict_2week'],labels=[1,0],output_dict=True)
classrep2 = pd.DataFrame(class_rep2).transpose()
print('Classification report : \n',classrep2)
classrep2.to_csv(table_output_path + '\\' + "2week_forecast_confmatrix_hexel" + str(hexel) + "_" + date + ".csv")

classrep2_f1score=classrep2['f1-score']
classrep2_accuracy=classrep2_f1score[2]

#SENSITIVITY, SPECIFICITY, PPV, NPV
sen2=conf_mat2[1,1]/(conf_mat2[1,1]+conf_mat2[1,0])*100
spec2=conf_mat2[0,0]/(conf_mat2[0,0]+conf_mat2[0,1])*100
ppv2=conf_mat2[1,1]/(conf_mat2[1,1]+conf_mat2[0,1])*100
npv2=conf_mat2[0,0]/(conf_mat2[0,0]+conf_mat2[1,0])*100
print("Sensitivity/Recall =",sen2,"Specificity/Selecitiviy =",spec2,"PPV/Precision =",ppv2,"NPV =",npv2)

# Week3 Confusion Matrix Forecast
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat3.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat3.flatten()/np.sum(conf_mat3)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
import seaborn as sns
ax = sns.heatmap(conf_mat3, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix 3 Week Forecast\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_ylim([0,2])
if save_output:
    plt.savefig(image_output_path + '\\' + "3weekimage_confmatrix_hexel" + str(hexel) + "_" + date + ".png")
    plt.close()
else:
    plt.show()

#Classification Report of the Confusion Matrix Test
class_rep3 = classification_report(df_copy['BUI'], df_copy['BUI_predict_3week'],labels=[1,0],output_dict=True)
classrep3 = pd.DataFrame(class_rep3).transpose()
print('Classification report : \n',classrep3)
classrep3.to_csv(table_output_path + '\\' + "3week_forecast_confmatrix_hexel" + str(hexel) + "_" + date + ".csv")

classrep3_f1score=classrep3['f1-score']
classrep3_accuracy=classrep3_f1score[2]

#SENSITIVITY, SPECIFICITY, PPV, NPV
sen3=conf_mat3[1,1]/(conf_mat3[1,1]+conf_mat3[1,0])*100
spec3=conf_mat3[0,0]/(conf_mat3[0,0]+conf_mat3[0,1])*100
ppv3=conf_mat3[1,1]/(conf_mat3[1,1]+conf_mat3[0,1])*100
npv3=conf_mat3[0,0]/(conf_mat3[0,0]+conf_mat3[1,0])*100
print("Sensitivity/Recall =",sen3,"Specificity/Selecitiviy =",spec3,"PPV/Precision =",ppv3,"NPV =",npv3)

# Week4 Confusion Matrix Forecast
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat4.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat4.flatten()/np.sum(conf_mat4)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
import seaborn as sns
ax = sns.heatmap(conf_mat4, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix 4 Week Forecast\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_ylim([0,2])
if save_output:
    plt.savefig(image_output_path + '\\' + "4weekimage_confmatrix_hexel" + str(hexel) + "_" + date + ".png")
    plt.close()
else:
    plt.show(block=True)

#Classification Report of the Confusion Matrix Test
class_rep4 = classification_report(df_copy['BUI'], df_copy['BUI_predict_4week'],labels=[1,0],output_dict=True)
classrep4 = pd.DataFrame(class_rep4).transpose()
print('Classification report : \n',classrep4)
classrep4.to_csv(table_output_path + '\\' + "4week_forecast_confmatrix_hexel" + str(hexel) + "_" + date + ".csv")

classrep4_f1score=classrep4['f1-score']
classrep4_accuracy=classrep4_f1score[2]

#SENSITIVITY, SPECIFICITY, PPV, NPV
sen4=conf_mat4[1,1]/(conf_mat4[1,1]+conf_mat4[1,0])*100
spec4=conf_mat4[0,0]/(conf_mat4[0,0]+conf_mat4[0,1])*100
ppv4=conf_mat4[1,1]/(conf_mat4[1,1]+conf_mat4[0,1])*100
npv4=conf_mat4[0,0]/(conf_mat4[0,0]+conf_mat4[1,0])*100
print("Sensitivity/Recall =",sen4,"Specificity/Selecitiviy =",spec4,"PPV/Precision =",ppv4,"NPV =",npv4)

# Week5 Confusion Matrix Forecast
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat5.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat5.flatten()/np.sum(conf_mat5)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
import seaborn as sns
ax = sns.heatmap(conf_mat5, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix 5 Week Forecast\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_ylim([0,2])
if save_output:
    plt.savefig(image_output_path + '\\' + "5weekimage_confmatrix_hexel" + str(hexel) + "_" + date + ".png")
    plt.close()
else:
    plt.show(block=True)

#Classification Report of the Confusion Matrix Test
class_rep5 = classification_report(df_copy['BUI'], df_copy['BUI_predict_5week'],labels=[1,0],output_dict=True)
classrep5 = pd.DataFrame(class_rep5).transpose()
print('Classification report : \n',classrep5)
classrep5.to_csv(table_output_path + '\\'+ "5week_forecast_confmatrix_hexel" + str(hexel) + "_" + date + ".csv")

classrep5_f1score=classrep5['f1-score']
classrep5_accuracy=classrep5_f1score[2]

#SENSITIVITY, SPECIFICITY, PPV, NPV
sen5=conf_mat5[1,1]/(conf_mat5[1,1]+conf_mat5[1,0])*100
spec5=conf_mat5[0,0]/(conf_mat5[0,0]+conf_mat5[0,1])*100
ppv5=conf_mat5[1,1]/(conf_mat5[1,1]+conf_mat5[0,1])*100
npv5=conf_mat5[0,0]/(conf_mat5[0,0]+conf_mat5[1,0])*100
print("Sensitivity/Recall =",sen5,"Specificity/Selecitiviy =",spec5,"PPV/Precision =",ppv5,"NPV =",npv5)

###############################################################################

#MAE DIFFERENCES
MAEclim_diff1 = multivariate_MAE1 - climatology_MAE1
MAEclim_diff2 = multivariate_MAE2 - climatology_MAE1
MAEclim_diff3 = multivariate_MAE3 - climatology_MAE1
MAEclim_diff4 = multivariate_MAE4 - climatology_MAE1
MAEclim_diff5 = multivariate_MAE5 - climatology_MAE1

MAEpers_diff1 = multivariate_MAE1 - persistance_MAE1
MAEpers_diff2 = multivariate_MAE2 - persistance_MAE2
MAEpers_diff3 = multivariate_MAE3 - persistance_MAE3
MAEpers_diff4 = multivariate_MAE4 - persistance_MAE4
MAEpers_diff5 = multivariate_MAE5 - persistance_MAE5


#SAVE RESULTS##################################################################

#Saving the Results in DataFrames
errors_df = pd.DataFrame(np.array([[1, multivariate_MSE1,climatology_MSE1,persistance_MSE1,multivariate_MAE1,climatology_MAE1,persistance_MAE1,multivariate_MBE1,climatology_MBE1,persistance_MBE1,MAEclim_diff1,MAEpers_diff1,classrep1_accuracy],
                                   [2, multivariate_MSE2,climatology_MSE1,persistance_MSE2,multivariate_MAE2,climatology_MAE1,persistance_MAE2,multivariate_MBE2,climatology_MBE1,persistance_MBE2,MAEclim_diff2,MAEpers_diff2,classrep2_accuracy],
                                   [3, multivariate_MSE3,climatology_MSE1,persistance_MSE3,multivariate_MAE3,climatology_MAE1,persistance_MAE3,multivariate_MBE3,climatology_MBE1,persistance_MBE3,MAEclim_diff3,MAEpers_diff3,classrep3_accuracy],
                                   [4, multivariate_MSE4,climatology_MSE1,persistance_MSE4,multivariate_MAE4,climatology_MAE1,persistance_MAE4,multivariate_MBE4,climatology_MBE1,persistance_MBE4,MAEclim_diff4,MAEpers_diff4,classrep4_accuracy],
                                   [5, multivariate_MSE5,climatology_MSE1,persistance_MSE5,multivariate_MAE5,climatology_MAE1,persistance_MAE5,multivariate_MBE5,climatology_MBE1,persistance_MBE5,MAEclim_diff5,MAEpers_diff5,classrep5_accuracy]]),
                                   columns=['week', 'multivariate MSE','climatology MSE', 'persistance MSE','multivariate MAE','climatology MAE', 'persistance MAE','multivariate MBE','climatology MBE', 'persistance MBE','clim diff MAE', 'pers diff MAE','accuracy'])


errors_df.to_csv(table_output_path + "result_errors_hexel" + str(hexel) + 'model' + str(modelnum) + "_" + date + ".csv")


perc_75=df_results.BUI.quantile(perc_test)
percentile_value_df = pd.DataFrame(np.array([1,perc_75]),columns=['95th perc'])
percentile_value_df.to_csv(table_output_path + "percentile_value_hexel" + str(hexel) + 'model' + str(modelnum) + "_" + date + ".csv")
print(perc_75)


#END###########################################################################
