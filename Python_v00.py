from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import glob, os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_file(path):
  list_file = glob.glob(os.path.join(path, '*.jpg'))
  return list_file

def label_gen (string):
  lst = str.split(string, "/")
  if lst[-1].startswith('c'):
    return 0
  elif lst[-1].startswith('d'):
    return 1

def data_gen (files):
  X = [] #list of inputs x's
  y = [] # list of labels y's
  for file in files:
    lb = label_gen(file)
    img = cv2.imread(file)
    img = cv2.resize(img, (384,384))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #plt.show()
    X.append(img)
    y.append(lb)
  return np.array(X), np.array(y)

def gen_npy (X, y):
  # split 80 10 10
  X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42) 
  X_test, X_valid, y_test, y_valid = train_test_split (X_test, y_test, test_size=0.5, random_state=42) 

  folder = "npy_data"
  if not os.path.exists(folder):
    os.makedirs(folder)

  # save to npy file
  np.save(os.path.join(folder, "X_train.npy"), X_train)
  np.save(os.path.join(folder, "X_test.npy"), X_test)
  np.save(os.path.join(folder, "X_valid.npy"), X_valid)
  np.save(os.path.join(folder, "y_train.npy"), y_train)
  np.save(os.path.join(folder, "y_test.npy"), y_test)
  np.save(os.path.join(folder, "y_valid.npy"), y_valid)

  print ("{} train and {} test/valid was generated!".format(len(X_train), len(X_test)))
  
  
  import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score


#from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.resnet50 import preprocess_input

def model ():
  base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(384,384,3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(units=1, activation='sigmoid')(x)

  model = Model(inputs = base_model.input, outputs = predictions)
  #model.summary()

  return model
  
  
  def train (data_path, weight_name):

  X_train = np.load(os.path.join(data_path, "X_train.npy"))
  X_valid = np.load(os.path.join(data_path, "X_valid.npy"))
  y_train = np.load(os.path.join(data_path, "y_train.npy"))
  y_valid = np.load(os.path.join(data_path, "y_valid.npy"))
  
  cls_model = model()

  # VAHID I added this
  weight_path = 'weights'
  if not os.path.exists(weight_path):
    os.makedirs(weight_path)

  weight = os.path.join(weight_path, weight_name)
  model_checkpoint = ModelCheckpoint(weight, monitor='val_loss',verbose=2, save_best_only=True)

  cls_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics = ['accuracy'])
  hist = cls_model.fit(X_train, y_train, epochs=10, batch_size = 32, validation_data = (X_valid, y_valid), callbacks=[model_checkpoint])  
  
  
  def test (data_path, weight_name):

  X_test = np.load(os.path.join(data_path, "X_test.npy"))
  y_test = np.load(os.path.join(data_path, "y_test.npy"))
  
  cls_model = model()
  weight_path = os.path.join("weights", weight_name)
  cls_model.load_weights(weight_path)
  
  y_pred = cls_model.predict(X_test)
  false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred)
  
  # Ploting ROC curves
  plt.subplots(1, figsize=(10,10))
  plt.title('Receiver Operating Characteristic - DecisionTree')
  plt.plot(false_positive_rate, true_positive_rate,label="auc="+str(auc))
  plt.legend(loc=4)
  plt.plot([0, 1], ls="--")
  plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()


files = read_file('/content/drive/My Drive/Vahid/Data/')
X, y = data_gen (files)
print (X.shape, y.shape)
gen_npy(X, y)

from tensorflow.keras.layers import Conv1D, MaxPool1D, ReLU, Input, BatchNormalization, MaxPool1D, Dense, Flatten
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.optimizers import Adam



def new_model():
    input_shape = (1164, 1)
    input = Input(shape=input_shape)

    # Conv block 1
    x = Conv1D(128, 3, padding='same', name='conv1')(input)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2, name='pool1')(x)

    # Conv block 2
    x = Conv1D(128, 3, padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2, name='pool2')(x)

    # Conv block 3
    x = Conv1D(128, 3, padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2, name='pool3')(x)

    x = Flatten()(x)
    output = Dense(1, activation='linear', name='output')(x)
    model = Model(inputs=input, outputs=output)
    return model
  
  def train ():

  myModel = new_model()
  myModel.summary()

  X_train = np.ones((20, 1164))
  y_train = np.array([1]*20)


  X_valid = np.ones((20, 1164))
  y_valid = np.array([1]*20)

  myModel.fit

  model_checkpoint = ModelCheckpoint("weight.h5", monitor='val_loss',verbose=2, save_best_only=True)

  myModel.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics = ['accuracy'])
  hist = myModel.fit(X_train, y_train, epochs=10, batch_size = 32, validation_data = (X_valid, y_valid), callbacks=[model_checkpoint])

  
  from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def validation (X_test, y_test, weights, name):
    model_cls = cls_testing_model()
    model_cls.load_weights(weights)
    #model_cls.summary()
    X_test = preprocess_input(X_test)
    y_pred = model_cls.predict(X_test)    r2 = r2_score(y_test, y_pred)
    regr = linear_model.LinearRegression()    y_test_p = y_test.reshape(-1, 1)
    regr.fit(y_test_p, y_pred)
    pred_y_pred = regr.predict(y_test_p)    plt.scatter(y_test, y_pred,  color='darkcyan', alpha = 0.6)
    plt.plot(y_test_p, pred_y_pred, color='olive', linewidth=3)
    plt.savefig(name+'.png')
    plt.close()    return r2, y_test, y_pred

  def LSTM_model():
    input_shape = (1164, 1)
    input = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(input)
    x = Dropout(0.5)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model
  
  temp = np.array([1,2,3,4]*291)
X_raw = np.array([temp, temp])

# this is for 8 grouping and uses re_arrange function
X_8grouped = np.array([np.moveaxis(re_arrange(x), 0, 1) for x in X_raw])

# this if for 4 grouping 
X_4grouped = np.array([np.moveaxis(np.reshape(x, (291, 4)), 0, 1) for x in X_raw])

# this is for 1 grouping
X_1grouped = np.array([np.moveaxis(np.reshape(x, (291, 4)), 0, 1).flatten() for x in X_raw])


print (X_8grouped.shape, X_4grouped.shape, X_the_boss.shape)

def re_arrange (data):

  ord_1 = np.reshape(data, (291, 4))

  ord_1_first_half = ord_1[:145]
  ord_1_socond_half = ord_1[146:]
  ord_1_socond_half = ord_1_socond_half[::-1] #reverse the array
  ord_2 = np.concatenate((ord_1_first_half, ord_1_socond_half), axis = 1)

  return ord_2


import numpy as np
import math
import matplotlib.pyplot as plt

def gen_file (file_name, data):

  X = [int(x) for [x,y,val] in data]
  Y = [int(y) for [x,y,val] in data]
  val_dict = {"{}:{}".format(int(x), int(y)):val for [x, y, val] in data}

  n = max(X)-min(X)+1
  m = max(Y)-min(Y)+1
  print (val_dict)
  result = np.zeros((n,m))

  for i in range(min(X), max(X)+1):
    for j in range(min(Y), max(Y)+1):
      temp = "{}:{}".format(i, j)
      if temp in val_dict:
        result[i, j] = val_dict[temp]

  new_name = "vis" + file_name

  plt.pcolormesh(result)
  plt.savefig(new_name)
  #plt.show()

  return result
