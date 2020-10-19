 import base64

import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import requests

from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn import metrics,linear_model

from sklearn.metrics import recall_score, classification_report, auc, roc_curve

 

 

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

 

 

# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1

# at every location where the original column (name) matches each of the target_values.  One column is added for

# each target value.

def encode_text_single_dummy(df, name, target_values):

    for tv in target_values:

        l = list(df[name].astype(str))

        l = [1 if str(x) == str(tv) else 0 for x in l]

        name2 = f"{name}-{tv}"

        df[name2] = l

 

 

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).

def encode_text_index(df, name):

    le = preprocessing.LabelEncoder()

    df[name] = le.fit_transform(df[name])

    return le.classes_

 

 

# Encode a numeric column as zscores

def encode_numeric_zscore(df, name, mean=None, sd=None):

    if mean is None:

        mean = df[name].mean()

 

    if sd is None:

        sd = df[name].std()

 

    df[name] = (df[name] - mean) / sd

 

 

# Convert all missing values in the specified column to the median

def missing_median(df, name):

    med = df[name].median()

    df[name] = df[name].fillna(med)

 

 

# Convert all missing values in the specified column to the default

def missing_default(df, name, default_value):

    df[name] = df[name].fillna(default_value)

 

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs

def to_xy(df, target):

    result = []

    for x in df.columns:

        if x != target:

            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(

    target_type = df[target].dtypes

    target_type = target_type[0] if hasattr(

        target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.

    if target_type in (np.int64, np.int32):

        # Classification

        dummies = pd.get_dummies(df[target])

        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)

    # Regression

    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

 

 

# Nicely formatted time string

def hms_string(sec_elapsed):

    h = int(sec_elapsed / (60 * 60))

    m = int((sec_elapsed % (60 * 60)) / 60)

    s = sec_elapsed % 60

    return f"{h}:{m:>02}:{s:>05.2f}"

 

 

# Regression chart.

def chart_regression(pred, y, sort=True):

    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})

    if sort:

        t.sort_values(by=['y'], inplace=True)

    plt.plot(t['y'].tolist(), label='expected')

    plt.plot(t['pred'].tolist(), label='prediction')

    plt.ylabel('output')

    plt.legend()

    plt.show()

 

# Remove all rows where the specified column is +/- sd standard deviations

def remove_outliers(df, name, sd):

    drop_rows = df.index[(np.abs(df[name] - df[name].mean())

                          >= (sd * df[name].std()))]

    df.drop(drop_rows, axis=0, inplace=True)

 

 

# Encode a column to a range between normalized_low and normalized_high.

def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,

                         data_low=None, data_high=None):

    if data_low is None:

        data_low = min(df[name])

        data_high = max(df[name])

 

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \

        * (normalized_high - normalized_low) + normalized_low

 

def read_main_data(path):

    filename_read = os.path.join(path,"DLInput_FilteredKob_THA_Test_Rot_Flip_Scaled.csv")

    filename_write = os.path.join(path,"Result_DLInput_FilteredKob_THA_Test_Rot_Flip_Scaled.csv")

    df_main = pd.read_csv(filename_read,na_values=['NA','?'])

    return df_main

    

    

def read_holdout_data(path):

    filename_read = os.path.join(path,"DLInput_FilteredKob_THA_TSMQCN6F.csv")

    filename_write = os.path.join(path,"Result_DLInput_FilteredKob_THA_TSMQCN6F.csv")

    df_valid = pd.read_csv(filename_read,na_values=['NA','?'])

    return df_valid   

 

# Split the main data into train and test 

def gen_train_test (X, y):

  # split 80 20

  X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42) 

 

  folder = "train_test_data"

  if not os.path.exists(folder):

    os.makedirs(folder)

 

  # save to npy file

  np.save(os.path.join(folder, "X_train.npy"), X_train)

  np.save(os.path.join(folder, "X_test.npy"), X_test)  

  np.save(os.path.join(folder, "y_train.npy"), y_train)

  np.save(os.path.join(folder, "y_test.npy"), y_test)

 

 

  #print("Shape of Main Set: {}".format(X_train.shape))

  #print("Shape of Hold-out Set: {}".format(X_test.shape)) 

  print ("{} train and {} test was generated!".format(len(X_train), len(X_test)))

 

# Read the Hold-out (validation) data 

def gen_valid (X_valid, y_valid):

  

  folder = "train_test_data"

  if not os.path.exists(folder):

    os.makedirs(folder)

 

  # save to npy file

  np.save(os.path.join(folder, "X_valid.npy"), X_valid)

  np.save(os.path.join(folder, "y_valid.npy"), y_valid)

 

def model_NN():

            

    model_NN = Sequential()        

    model_NN.add(Dense(300, input_dim=X.shape[1], activation='relu'))

    model_NN.add(Dense(400, activation='relu'))

    model_NN.add(Dense(500, activation='relu'))

    model_NN.add(Dense(300, activation='relu'))

    model_NN.add(Dense(200, activation='relu'))

    model_NN.add(Dense(100, activation='relu'))

    output = model_NN.add(Dense(1))

    

    return model_NN

 

def model_LSTM():

    input_shape = (291, 4)

    input = Input(shape=input_shape)

 

    x = LSTM(128, return_sequences=True)(input)

    x = Dropout(0.5)(x)

    x = LSTM(128, return_sequences=True)(x)

    x = Dropout(0.5)(x)

 

    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input, outputs=output)

    return model

 

def model_CNN_1D():

    

    input_shape = (1164,1)

    input = Input(shape=input_shape)

 

    # Conv block 1

    x = Conv1D(128, 4, padding='same', name='conv1')(input)

    x = BatchNormalization(name='bn1')(x)

    x = ReLU()(x)

    x = MaxPool1D(pool_size=4, name='pool1')(x)

    

    # Conv block 2

    x = Conv1D(128, 4, padding='same', name='conv2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = ReLU()(x)

    x = MaxPool1D(pool_size=4, name='pool2')(x)

 

    # Conv block 3

    x = Conv1D(128, 4, padding='same', name='conv3')(x)

    x = BatchNormalization(name='bn3')(x)

    x = ReLU()(x)

    x = MaxPool1D(pool_size=4, name='pool3')(x)   

    

    # Conv block 4

    x = Conv1D(128, 4, padding='same', name='conv4')(x)

    x = BatchNormalization(name='bn4')(x)

    x = ReLU()(x)

    x = MaxPool1D(pool_size=4, name='pool4')(x)   

    

#      # Conv block 5

#     x = Conv1D(64, 4, padding='same', name='conv5')(x)

#     x = BatchNormalization(name='bn5')(x)

#     x = ReLU()(x)

#     x = MaxPool1D(pool_size=4, name='pool5')(x)   

 

    x = Flatten()(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input, outputs=output)

    return model

 

def model_CNN_2D():

    

    input_shape = (291, 4,1)

    input = Input(shape=input_shape)

 

    # Conv block 1

    x = Conv2D(128, (4,4), padding='same', name='conv1')(input)

   x = BatchNormalization(name='bn1')(x)

    x = ReLU()(x)

    x = MaxPool1D(pool_size=(4,4), name='pool1')(x)

 

#     # Conv block 2

#     x = Conv2D(128, (4,4), padding='same', name='conv2')(x)

#     x = BatchNormalization(name='bn2')(x)

#     x = ReLU()(x)

#     x = MaxPool1D(pool_size=(4,4), name='pool2')(x)

 

#     # Conv block 3

#     x = Conv2D(128, (4,4), padding='same', name='conv3')(x)

#     x = BatchNormalization(name='bn3')(x)

#     x = ReLU()(x)

#     x = MaxPool1D(pool_size=(4,4), name='pool3')(x)

    

    x = Flatten()(x)

    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input, outputs=output)

    return model

 

def train (data_path, weight_name):

  

    X_train = np.load(os.path.join(data_path, "X_train.npy"))

    X_test  = np.load(os.path.join(data_path, "X_test.npy"))

    y_train = np.load(os.path.join(data_path, "y_train.npy"))

    y_test  = np.load(os.path.join(data_path, "y_test.npy"))

    

    model=model_CNN_1D()

    model.summary()

    model.fit

    

    weight_path = 'weights'

    if not os.path.exists(weight_path):

        os.makedirs(weight_path)

    

    model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=['mean_absolute_percentage_error'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=1, mode='auto')

   

    weight = os.path.join(weight_path, weight_name)

    checkpointer = ModelCheckpoint(weight, verbose=2, save_best_only=True, monitor='val_loss')

    hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor, checkpointer],verbose=2, batch_size=32, epochs=20)

             

    # Plot the chart for Loss

    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

 

    # Plot the chart for accuracy

    plt.plot(hist.history['mean_absolute_percentage_error'])

    plt.plot(hist.history['val_mean_absolute_percentage_error'])

    plt.title('model accuracy')

    plt.ylabel('MSE_error (%)')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

 

    # Plot the chart for 

def plot_LinReg(X,y, model):

        

    y_pred = model.predict(X)

        

    regr = linear_model.LinearRegression()

    y_p = y.reshape(-1, 1)

    regr.fit(y_p, y_pred)    

    pred_y_pred = regr.predict(y_p)

    

    plt.scatter(y, y_pred,  color='darkcyan', alpha = 0.6)

    plt.plot(y_p, pred_y_pred, color='olive', linewidth=3)

    plt.title('Actual vs. Prediction')

    plt.ylabel('Prediction')

    plt.xlabel('Actual')

    plt.show()

    plt.close()

    

def evaluation (data_path, weight_name):

    

    X_test  = np.load(os.path.join(data_path, "X_test.npy"))

    X_valid = np.load(os.path.join(data_path, "X_valid.npy"))

    y_test  = np.load(os.path.join(data_path, "y_test.npy"))

    y_valid = np.load(os.path.join(data_path, "y_valid.npy"))

    

    model=model_CNN_1D()

    weight_path = os.path.join("weights", weight_name)

    model.load_weights(weight_path) 

    

    y_pred = model.predict(X_test)

    rsquared_main=r2_score(y_pred,y_test)

    

    y_pred_valid = model.predict(X_valid)

    rsquared_valid=r2_score(y_pred_valid,y_valid)

             

    

     # Plot the chart for test dataset

    chart_regression(y_pred.flatten(),y_test)    

    plot_LinReg(X_test,y_test, model)

      

    # Calculate RMSE for test dataset

    score = np.sqrt(metrics.mean_squared_error(y_pred,y_test))

    print("Score (RMSE) for Main Data is: {}".format(score))

    

    # Calculate R-sq for teh Main dataset    

    print("The R-sq for Main Data is: {}".format(rsquared_main)) 

         

    # Plot the chart for Hold-out dataset

    chart_regression(y_pred_valid.flatten(),y_valid)  

    plot_LinReg(X_valid,y_valid, model)

        

    # Calculate RMSE for Hold-out dataset

    score_valid = np.sqrt(metrics.mean_squared_error(y_pred_valid,y_valid))

    print("Score_valid (RMSE) for Hold-out Data is: {}".format(score_valid))

    

    # Calculate R-sq for Hold-out dataset    

    print("The R-sq for Hold-out Data is: {}".format(rsquared_valid)) 

 

    

def re_arrange_row (row):

 

    ord_1 = np.reshape(row, (291, 4))

    ord_1_first_half = ord_1[:145]

    ord_1_socond_half = ord_1[146:]

    ord_1_socond_half = ord_1_socond_half[::-1] #reverse the array

    ord_2 = np.concatenate((ord_1_first_half, ord_1_socond_half), axis = 1)

 

    return ord_2

 

def re_shape_1(X):              

       

    #X_processed = np.array([np.moveaxis(np.reshape(x, (291, 4)), 0, 1).flatten() for x in X])

    X_processed = np.array([np.reshape(x, (291, 4)).flatten() for x in X])

        

    return X_processed

 

def re_shape_4(X):              

       

    #X_processed=np.array([np.moveaxis(np.reshape(x, (291,4)),0,1) for x in X])

    X_processed=np.array([np.reshape(x, (291,4)) for x in X])

        

    return X_processed

 

def re_shape_8(X):              

       

    #X_processed = np.array([np.moveaxis(re_arrange_row(x), 0, 1) for x in X])

    X_processed = np.array([re_arrange_row(x) for x in X])

        

    return X_processed   

 

 

%matplotlib inline

from matplotlib.pyplot import figure, show

from sklearn.model_selection import train_test_split

import pandas as pd

import os

import numpy as np

from sklearn import metrics,linear_model

from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import zscore

import tensorflow as tf

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.externals.joblib import dump, load

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D

from tensorflow.keras.layers import Conv1D,Conv2D, MaxPool1D, ReLU, Input, BatchNormalization, MaxPool1D, Dense, Flatten, Embedding

from keras.layers import MaxPooling1D, LSTM

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Model

import numpy as np

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import recall_score, classification_report, auc, roc_curve

 

preprocess = False

 

# Read  data 

df_main  = read_main_data('./data/')

 

df_valid = read_holdout_data('./data/')

 

# Encode to a 2D matrix for training

X_raw,y = to_xy(df_main,'Column1')

#X= re_shape_1(X_raw)

 

X_valid_raw,y_valid = to_xy(df_valid,'Column1')

#X_valid= re_shape_1(X_valid_raw)

 

 

# Split the Main set into train/test

gen_train_test(X,y)



# Split the Validation(Hold-out) set into train/test

gen_valid(X_valid,y_valid)

 

# Train the model

train ("train_test_data", 'NameXXX.hdf5')

 

# Evaluation the model

evaluation ("train_test_data", 'NameXXX.hdf5')
