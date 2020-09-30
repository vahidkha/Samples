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
