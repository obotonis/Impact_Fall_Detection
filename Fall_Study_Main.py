#!/usr/bin/env python
# coding: utf-8

# # Data Analysis and Feature Selection

#  **Defined Code Functions** 

# In[1]:


# Code objective:(1) Import subject airbag data and .xls data sheet with fall event indices to segment data, 
# (2) determine the impact time of the fall within the segment, (3) create a 1s wide window 75ms before impact,
# (4) calculate features within the window space, #(5) export features onto .xls sheet and re-loop for each fall,
# (6) determine best features for multiple types ML based on trial and error.


# **Libraries**

# In[29]:


# Open library functions
import xlrd #to open and import excel data
from pathlib import Path #to concatenate file name
import pandas as pd #to perforn dataframe calulcations
import numpy as np #to perform mathematical equations
import scipy.io as sio #to load matrix data from .mat file
import math
import sklearn # to process data, access ML algorithms

# ML Required Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# SVM
from sklearn.svm import SVC 
# KNN
from sklearn.neighbors import KNeighborsClassifier


# **Import Data**

# In[3]:


# Get subject information
subject = input('Subject ID: ')
dataFolder = Path("Y:/Fall Catcher/Exp Data/")
subjectFolder = dataFolder/subject

# Import .xls combined data sheet per subject
eventFile = subject + '_CombinedData.xls'
eventDataLoc = subjectFolder/eventFile
eventdf = pd.read_excel(eventDataLoc)

# Import .mat airbag data file per subject
airbagFolder = subjectFolder/'.MAT files'
airbagWS = airbagFolder/'Airbag.mat'
Airbag = sio.loadmat(airbagWS)
airbagMat = Airbag.get('Airbag','Entry not found.')
airbagHeaders = ['t','bAx','bAy','bAz','bGx','bGy','bGz','rAx','rAy','rAz','rGx','rGy','rGz','lAx','lAy','lAz','lGx','lGy','lGz','output']
airbagdf = pd.DataFrame(airbagMat,columns = airbagHeaders)


# In[6]:


# Raw data import of airbag mat file and combined subject data sheet
print(eventdf)
print(airbagdf)


# **Pre-Processing Data**

# In[7]:


from sklearn import preprocessing

cols_normalized = ['bAx','bAy','bAz','bGx','bGy','bGz','rAx','rAy','rAz','rGx','rGy','rGz','lAx','lAy','lAz','lGx','lGy','lGz']
# Turn to units of G
airbagdf[cols_normalized] = airbagdf[cols_normalized].apply(lambda x: (x/(-2048)))
# Normalize airbag values between 0 and 1
airbagdf[cols_normalized] = airbagdf[cols_normalized].apply(lambda x: (x - x.min())/(x.max()-x.min()))


# In[8]:


# updated airbag file after processing
print(airbagdf)


# **Identify Theoretical Impact Times and Create Fall Segment Files**

# In[9]:


# For each fall, determine the true time of the impact. 
theorImpactTime = []
impactError = []

# Based on impact time, create segmented data file and append to appropriate list.
allFall_clipList = []
sideFall_clipList = []
NF_clipList = [] #non-fall.
fbFall_clipList = [] #forward or backward fall.

# Clip size in frames.
windowSize = 2000
halfWin = windowSize/2

for event in range(eventdf.shape[0]):
    # For each loop, initialize the fall data frame to be empty.
    falldf = pd.DataFrame()
    
    # Get start and stop times.
    if eventdf.eventLabel[event] == 500: # No range times documented- use time window between documented falls.
        start_range = eventdf.eventEnd_ms[event-1]
        end_range = eventdf.eventStart_ms[event+1]
    else:
        start_range = eventdf.eventStart_ms[event]
        end_range = eventdf.eventEnd_ms[event]
    
    # Get the index of the airbag time closely aligning to the start and stop marked timepoints.
    airbagStartIndex = airbagdf[airbagdf['t']>=start_range].index.values.astype(int)[0]
    airbagEndIndex =airbagdf[airbagdf['t']>= end_range].index.values.astype(int)[0]
    
    # Create a sub-data frame for the fall data.
    falldf = airbagdf.loc[airbagStartIndex:airbagEndIndex]
    falldf.columns = airbagHeaders

    
    # ATTEMPTS TO FIND TRUE / THEORETICAL IMPACT TIMES:
    # Attempt 1: Calculate difference with previous row to get largest slope change during task event.
    
    # ...
    
    # Attempt 2:  Calculate magnitude of acceleration for each sensor
    # Subset dataframe to find magnitude of acceleration for each sensor.
    bAdf = falldf[['bAx','bAy','bAz']]
    rAdf = falldf[['rAx','rAy','rAz']]
    lAdf = falldf[['lAx','lAy','lAz']]
    mag_bAdf = []
    mag_rAdf = []
    mag_lAdf = []
    
    # Get magnitude vector for each sensor.
    mag_bAdf= bAdf.apply(np.linalg.norm, axis = 1)
    mag_rAdf= rAdf.apply(np.linalg.norm, axis = 1)
    mag_lAdf= lAdf. apply(np.linalg.norm, axis = 1)

    # Get the maximum magnitude for each sensor data frame.
    maxA = []
    maxA.append(mag_bAdf.max())
    maxA.append(mag_rAdf.max())
    maxA.append(mag_lAdf.max())

    # Overall maximum magnitude value.
    valMaxA = max(maxA)
    sensorMaxA = maxA.index(max(maxA))
    if sensorMaxA == 0:
        sensorType = "base"
        indexMaxA = mag_bAdf[mag_bAdf == valMaxA].index.values.astype(int)[0]
    elif sensorMaxA == 1:
        sensorType = "right"
        indexMaxA = mag_rAdf[mag_rAdf == valMaxA].index.values.astype(int)[0]
    elif sensorMaxA == 2:
        sensorType = "left"
        indexMaxA = mag_lAdf[mag_lAdf == valMaxA].index.values.astype(int)[0]

    timeMaxA = airbagdf.t[indexMaxA]
    theorImpactTime.append(timeMaxA)
    #print('The maximum acceleration value for the fall event was',valMaxA,' given by the',sensorType,'sensor at time t = ',timeMaxA, ' ms')

    # Calculate percent error for the theoretical impact time if experimental impact time marked during session. 
    if eventdf.impactTime_ms[event] != 0:
        expImpactTime = eventdf.impactTime_ms[event]
        perError = ((abs(expImpactTime-timeMaxA))/timeMaxA)*100
        impactError.append(perError)
    else:
        perError = float("NaN")
        impactError.append(perError)
    
    # Attempt #N.....
    
    # Compare and select the method with the lowest error when compared to the manual impact time point. 
    # (if a manual time point exists)

    
    
    
    # Create the Segmented Clip for the Fall
    # Get index of airbag time for the pre-selected window size.
    event_index = airbagdf[airbagdf['t']>= timeMaxA].index.values.astype(int)[0]
    startInd = event_index - halfWin
    endInd = event_index + halfWin
    clipdf = airbagdf.loc[startInd:endInd]
    clipdf.columns = airbagHeaders
    
    # Append to fall/NF list.
    if eventdf.actualFallType[event]==2:
        sideFall_clipList.append(clipdf)
        allFall_clipList.append(clipdf)
    elif eventdf.actualFallType[event]==1:
        allFall_clipList.append(clipdf)
        fbFall_clipList.append(clipdf)
    else:
        NF_clipList.append(clipdf)        
            


# In[10]:


print(theorImpactTime)


# **Define Feature Calculations**

# In[11]:


# Define feature calculations here if necessary... 


# **Feature Calculations on Pre-Impact Window**

# In[12]:


# Create empty dictionary for calculated features to be appended to. Fall Label (Key) : Feature Vector (Value)
fallFeaturesDict = dict()

# Set parameters for window size.
leadTime = 75
windowSize = 1000
# Set parameters for feature selection.
chosenVars = ['bAx','bAy','bAz','rAx','rAy','rAz','lAx','lAy','lAz']
calculations = ['mean()','max()','min()','std()']


for impact in range(len(theorImpactTime)):
    # Create window.
    endWindow = theorImpactTime[impact] - leadTime 
    startWindow = endWindow - windowSize
    
    # Assign airbag data index for window size.
    airbagStartWin = airbagdf[airbagdf['t']>=startWindow].index.values.astype(int)[0]
    airbagEndWin =airbagdf[airbagdf['t']>= endWindow].index.values.astype(int)[0]
    
    # Create sub data frame of airbag data within the window.
    dfWindow = airbagdf[chosenVars].loc[airbagStartWin:airbagEndWin]

    
    
    
    # PERFORM CALCULATIONS ON WINDOW OF TIME
    # Pre-allocate temporary dataframe for the window of feature data calculated.
    featureCol = [] 
    # Get and assign proper column names based on variables and functions chosen.
    for var in chosenVars:
        for func in calculations:
            featureCol.append(var + '_' + func)
    dfWinFeatures = pd.DataFrame(columns = featureCol) # Temporary DF row will be appended to this. 
    
    # Use chosen variables and functions to calculate feature statistics over the window. Temporarily store in dfWinFeatures.
    for var in chosenVars:
        for func in calculations:
            evalStatement = ('dfWindow["'+var+'"].'+func)
            dfWinFeatures.at[0,var + '_' + func] = (eval(evalStatement))
   

    
    # ASSIGN WINDOW FEATURES TO FALL FEATURE DICTIONARY
    featureValues = []
    eventID = str(eventdf.eventLabel[impact])
    nextLetter = 'A'
    
    # 'If' statement to append a letter to eventID if already exists in the dictionary.
    while eventID in fallFeaturesDict.keys():
        if len(eventID) == 4:
            eventID = eventID[:-1]
        eventID = eventID + nextLetter
        nextLetter = chr(ord(nextLetter)+1) 
        
    # Assign dictionary key and value pair for the fall    
    for i in range(dfWinFeatures.shape[1]): # Column wise
        for j in range (dfWinFeatures.shape[0]): # Row wise
            featureValues.append(dfWinFeatures.iloc[j,i])
   # Update the dictionary.
    fallFeaturesDict.update({eventID:featureValues})
    
            


# In[ ]:


print(dfWinFeatures)
print(fallFeaturesDict)


# **Get Pre-Impact Feature Dataframe for Whole Session**

# In[13]:


# Assign features per fall from dictionary to dataframe.
fallfeaturesdf = pd.DataFrame.from_dict(fallFeaturesDict)
fallfeaturesdf = fallfeaturesdf.transpose()
fallfeaturesdf.columns = dfWinFeatures.columns

# Append a final column based on whether the event IS a right/left fall (1) or is not a right/left fall (0)
rlFall = ['103','108','111','115','117','119','104','109','112','116','118','120']
outputLabel = []
for index in fallfeaturesdf.index:
    if index in rlFall:
        outputLabel.append(1)
    else:
        outputLabel.append(0)
        
fallfeaturesdf['Output'] = outputLabel
fallfeaturesdf


# **Create Separate DataFrame of Pre-Impact Features based on Fall Type**

# In[22]:


allFall_preFeatures = []
sideFall_preFeatures = []
NF_preFeatures = []
fbFall_preFeatures = []

for i in range((fallfeaturesdf.shape[0])):
    if eventdf.actualFallType[i]==2:
        sideFall_preFeatures.append(list(fallfeaturesdf.iloc[i,:]))
        allFall_preFeatures.append(list(fallfeaturesdf.iloc[i,:]))
    elif eventdf.actualFallType[i]==1:
        fbFall_preFeatures.append(list(fallfeaturesdf.iloc[i,:]))
        allFall_preFeatures.append(list(fallfeaturesdf.iloc[i,:]))
    else:
        NF_preFeatures.append(list(fallfeaturesdf.iloc[i,:]))

# Turn all lists into data frame with same column headings. 
allFall_preFeatDF = pd.DataFrame(allFall_preFeatures, columns = fallfeaturesdf.columns) 
sideFall_preFeatDF = pd.DataFrame(sideFall_preFeatures, columns = fallfeaturesdf.columns)
NF_preFeatDF = pd.DataFrame(NF_preFeatures, columns = fallfeaturesdf.columns)
fbFall_preFeatDF = pd.DataFrame(fbFall_preFeatures, columns = fallfeaturesdf.columns)


# In[ ]:


# Output the feature dataframe as a document in excel.


# # Import machine learning algorithms from sklearn

# In[ ]:


# Set parameters.
percent_test = 0.40 # Test size for x data.

# Select the feature data: all rows, all columns except the last (label).
x = fallfeaturesdf.iloc[:,:-1].values
# Select the label column as the output label.
y = fallfeaturesdf['Output']

# Split the data into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = percent_test, random_state = 0)

#print(x_train)
#print(y_train)


# In[27]:


# SVM: Side Fall (1) versus Not Side Fall (0) For Single Subject Session

SVC_model = SVC()
SVC_model.fit(x_train,y_train)
SVC_prediction = SVC_model.predict(x_test)

print(accuracy_score(SVC_prediction, y_test))


# In[34]:


# KNN: Side Fall (1) versus Not Side Fall (0) for Single Subject Session

KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model. fit(x_train, y_train)
KNN_prediction = KNN_model.predict(x_test)

print(accuracy_score(KNN_prediction, y_test))


# In[ ]:



        # *** This is easier: dfWindow.describe() gives direct statistics on every column.
        df.describe().transpose()
    
        # OR pip install pandas-profiling
        df.profile_report() # Essentials, quantile statistics, descriptive statistics, most frequent values, histogram, correlations, etc. EVERYTHING.
        # This is column wise. We would want some of these to be between sets of data/columns.

    # Assign key:value pair to the fallFeaturesDict
 
    
    # Allow program to re-loop and perform again for another fall.


# In[ ]:




