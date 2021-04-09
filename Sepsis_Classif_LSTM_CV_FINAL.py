"""
Kinematics Approach with Neural Networks for Early De-tection of Sepsis (KANNEDS) 
Márcio Freire Cruz2,1*, Naoaki Ono3,1, Ming Huang1, Md. Altaf-Ul-Amin1, 
Shigehiko Kana-ya1, Carlos Arthur Mattos Teixeira Cavalcante2

1Graduate School of Science and Technology, Nara Institute of Science and Technology, Ikoma, Takayama, 8916-5 Nara, Japan.
2Graduate Program in Mechatronics, Federal University of Bahia, Salvador, Bahia, 40170-110, Brazil.
3Data Science Center, Nara Institute of Science and Technology, Ikoma, Takayama, 8916-5 Nara, Japan.
*Corresponding author (marciofreire@gmail.com)


Algorithm for sepsis classification using LSTM NN
"""

import os
# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
# Disable other warnings
warnings.simplefilter("ignore", category=FutureWarning)

import tensorflow as tf
import timeit
import numpy as np
import random as rn

import pandas as pd

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adagrad



# Get the correct number of patient data in sequencial form
def getSequences(PacsPositives, PacsNegatives, numTS):

   
    # Get the patients data values
    xPacsPositives = PacsPositives.values[:,:]
    xPacsNegatives = PacsNegatives.values[:,:]
    
    # Format data in lists deleting the number of registers before the sepsis
    sequencesPos = list()
    for i in range(0,len(xPacsPositives),numTS):        
        sequencesPos.append(xPacsPositives[i:i+numTS,:])
    
    sequencesNeg = list()
    for i in range(0,len(xPacsNegatives),numTS):
        sequencesNeg.append(xPacsNegatives[i:i+numTS,:])

    return sequencesPos, sequencesNeg

# Get trainning and target data
def getTrainData(PacsPositives, PacsNegatives, numTSInitial):

    # Get the correct number TS and variables in sequencial form
    
    sequencesPos,sequencesNeg = getSequences(PacsPositives, PacsNegatives, numTSInitial)

    lenTrainPos = len(sequencesPos)
    lenTrainNeg = len(sequencesNeg)
    
    train = sequencesPos[0:lenTrainPos] + sequencesNeg[0:lenTrainNeg]
    
    train_target = np.concatenate((np.repeat(1,lenTrainPos), np.repeat(0,lenTrainNeg))) 
       
    # Array transformation
    train = np.array(train)
    
    # Shuffles the train data
    inds = np.random.choice(len(train),len(train),replace=False)
    train = train[inds]
    train_target = train_target[inds]

    return train,train_target

# Create the LSTM model
def create_model(numTS, numVariables, numNodes, percDropout):

    # Create a one layer LSTM model
    model = Sequential()
    model.add(LSTM(numNodes, input_shape=(numTS, numVariables))) #Layer LSTM nodes, shape(#timestamps,#variables)
    model.add(Dropout(percDropout)) # Layer dropout to randomly not consider 20% of the nodes
    model.add(Dense(1, activation='sigmoid')) # Output layer with one unit (0,1) and sigmoid activation function
    opt = Adagrad() 
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])    
    return model


# Timestamp data
numTSInitial = 49 # 1 Initial + 48 Timestamps per patient 

print('\n**************Classification of Sepsis using LSTM Model**************')
print("\nDate and time =", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
strDate = datetime.now().strftime("_%Y%m%d_%H%M%S")

# Read the original files
filesPath = 'Input/'
fileSufix = 'SAMPLE_DATA_FINAL' #'SAMPLE_DATA'
fileDescription = 'Database Sample for sepsis early detection'
pacsPositives = pd.read_csv(filesPath +'KANNEDS_POSITIVES_' + fileSufix + '.csv')
pacsNegatives = pd.read_csv(filesPath + 'KANNEDS_NEGATIVES_' + fileSufix  + '.csv')
variables = pd.read_csv(filesPath + 'KANNEDS_VARIABLES.csv')                                                           

print("Source file: " + fileSufix)
print(fileDescription)

# Variables data
numVarsGroups =  np.shape(variables)[1] # Number of groups
varsNicks = list(variables.columns) # Nicknames of variables
columns = pacsPositives.columns

# Number of patients
numPacsPos =  np.shape(pacsPositives)[0]/numTSInitial #total registers/numTS
numPacsNeg = np.shape(pacsNegatives)[0]/numTSInitial #total registers/numTS


# Model Params
numNodes = 128
percDropout = 0.2
numEpochs = 100
batch_size = 64  

# Sets the environment for reproducibility
numCores = 4  
numCPU = 1
numGPU = 0

os.environ['PYTHONHASHSEED'] = '0'       
os.environ['CUDA_VISIBLE_DEVICES'] = str(numGPU)             
np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)        
session_conf = tf.ConfigProto(intra_op_parallelism_threads=numCores,
                                  inter_op_parallelism_threads=numCores, 
                                  allow_soft_placement=True,
                                  device_count = {'CPU' : numCPU, 'GPU' : numGPU})    
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


print('%i patients (%i positives, %i negatives)' %((numPacsPos+numPacsNeg), numPacsPos, numPacsNeg))   
print("Model with %i Nodes, %.2f Dropout, %i Epochs and %i Bach Size." %(numNodes, percDropout, numEpochs,\
                                                                         batch_size))

# Gets data for training mixing positive and negative patients, and considering the num TS defined  
pacsTrain,pacsTrainTarget = getTrainData(pacsPositives, pacsNegatives, numTSInitial)
lenTrain = np.shape(pacsTrain) # train set dimension        
yTrain = pacsTrainTarget       
       
# Gets the folds for cross-validation
numFolds = 5
folds = list(StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=1).split(pacsTrain,pacsTrainTarget))
lenFoldTrain = np.shape(folds[0][0])
lenFoldValid = np.shape(folds[0][1]) 

print("%i-fold Cross-validation. Each fold with %i patients for train and %i for validation. " \
      %(numFolds,lenFoldTrain[0], lenFoldValid[0]))
print('\n*******************************************************************')

# Number of hours before sepsis
numHRBef = 6    
# New number of timestamps
numTS = numTSInitial - numHRBef 

print('\nCLASSIFICATION AT %i HOURS BEFORE SEPSIS ONSET (%i TIMESTAMPS).' %(numHRBef, numTS))       

# Repeats cross-validation for each group of variables
summary = []
for indVar in range(1,numVarsGroups):
     
    # Gets the indexes of the considered variables     
    importVars = [variables.values[i,0] for i in range(len(variables)) if variables.values[i,indVar] == 1]
    indsImportVars = [i for i in range(len(columns)) if columns[i] in importVars]
    numVariables = len(indsImportVars)
    
    print('\nVariables: %s' %(varsNicks[indVar]))

    # Gets the data for input according to numHRBef and indsImportVars
    xTrain = pacsTrain[:,0:numTS,indsImportVars] 
    
    # Initial cross-validation time
    initialTime = timeit.default_timer() 

    print("\nInitiating cross-validation...")     
    results = []
    for indFold, (trainInds, valInds) in enumerate(folds):    

        # Get folder data
        train = xTrain[trainInds]
        trainTarget = yTrain[trainInds]
        validation = xTrain[valInds]
        validationTarget = yTrain[valInds]        
        
        #Creates the model
        model = create_model(numTS, numVariables, numNodes, percDropout)
 
        # Trains the model and evaluate   
        model.fit(train, trainTarget, epochs=numEpochs, verbose=0, batch_size=batch_size)
        [modelLoss,modelAcc] = model.evaluate(validation,validationTarget,batch_size=batch_size,verbose=0)

        print("Fold %i: Acc = %.4f.; loss = %.4f." %((indFold+1),modelAcc,modelLoss)) 

        results.append(modelAcc)  

    # Saves the summary
    summary.append((numHRBef, varsNicks[indVar], np.mean(results), np.std(results),results))
    
    # Final cross-validation time
    processTime = timeit.default_timer() - initialTime 
    minutes = processTime / 60
    seconds = processTime % 60        
    print("Finished in %i minutes and %i seconds." %(minutes, seconds))
    print("Accuracy mean = {:.4f}".format(np.mean(results)), end='')
    print("(Std. = {:.4f}).".format(np.std(results)))
    print("--------------------------------------------------------------")  

print('\n**********THE END**********')
K.clear_session()


