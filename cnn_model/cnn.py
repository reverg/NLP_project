# %%
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import tensorflow as tf
from keras import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D, MaxPooling2D, Conv2D, LSTM, GRU, Bidirectional
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
import keras
# %%
df = pd.read_csv('NonPromoterSequence.txt', sep = '>', )
df.dropna(subset=['Unnamed: 0'], how='all', inplace=True)
df.reset_index(inplace = True)
df.drop(['EP 1 (+) mt:CoI_1; range -400 to -100.', 'index'], axis = 1, inplace=True) #data cleaning after error found
df.rename(columns={'Unnamed: 0': "sequence"}, inplace = True)
df['label'] = 0
print(df)
print(df.shape)
# %%
df2 = pd.read_csv('PromoterSequence.txt', sep = '>', )
df2.dropna(subset=['Unnamed: 0'], how='all', inplace=True)
df2.reset_index(inplace = True)
df2.drop(['EP 1 (+) mt:CoI_1; range -100 to 200.', 'index'], axis = 1, inplace=True)
df2.rename(columns={'Unnamed: 0': "sequence"}, inplace = True)
df2['label'] = 1

print(df2)
print(df2.shape)
# %%
df = pd.concat([df, df2], axis = 0 )
print(df.shape)
"""
for seq in df['sequence']:
    if 'N' in seq:
        print(df.loc[df['sequence'] == seq])
"""     
df.drop([1822], inplace = True)   
# %%
sequence = list(df.loc[:, 'sequence'])
encoded_list = []
# %%
def encode_seq(s):
    Encode = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
    return [Encode[x] for x in s]

for i in sequence:
    x = encode_seq(i)
    encoded_list.append(x)

X = np.array(encoded_list)
y = df['label'] # 22598 * 1
# %%
print(y.shape)
print(X.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

params = {
    'first_node': [128, 64],
    'second_node': [32, 64],
    'alpha': [0.001, 0.01],
    'first_filter': [9, 16, 32], 
    'dropout': [0.1, 0.2, 0.5]
}
#used for GridSearchCV

# %%
cnn_model = Sequential()
cnn_model.add(Conv1D(filters = 60, kernel_size = (4), activation = 'relu', input_shape = (301, 4))) # 298 * 60
cnn_model.add(MaxPooling1D(pool_size = (3))) # 99 * 60
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv1D(filters = 60, kernel_size = (2), activation = 'relu', padding = 'same')) # 99 * 30
cnn_model.add(MaxPooling1D(pool_size = (3))) # 33 * 15
cnn_model.add(Dropout(0.2))

cnn_model.add(Conv1D(filters = 30, kernel_size = (2), activation = 'relu', padding = 'same')) # 33 * 30
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation = 'relu'))
cnn_model.add(Dense(256, activation = 'relu'))
cnn_model.add(Dense(128, activation = 'relu'))
cnn_model.add(Dense(64, activation = 'relu'))
cnn_model.add(Dense(64, activation = 'relu'))
cnn_model.add(Dense(16, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
cnn_model.add(Dense(2, activation = 'sigmoid'))

cnn_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0003, patience=8, 
                                           restore_best_weights=True )
history = cnn_model.fit(X_train, y_train, batch_size = 128, validation_data=(X_test, y_test), 
                        epochs=115)
# %%
gru_model = Sequential()

gru_model.add(Conv1D(filters = 27, kernel_size = (3), activation = 'relu', input_shape = (301, 4))) # 298 * 27
gru_model.add(MaxPooling1D(pool_size = (3))) # 99 * 27
gru_model.add(Dropout(0.2))

gru_model.add(Conv1D(filters = 14, kernel_size = (2), activation = 'relu', padding = 'same')) # 99 * 14
#cnn_model.add(MaxPooling1D(pool_size= (2)))
#cnn_model.add(Dropout(0.2))


# gru_model.add(Bidirectional(GRU(128, activation = 'relu'))) # 256
gru_model.add(Dropout(0.2))
gru_model.add(Dense(128, activation = 'relu'))
gru_model.add(Dense(64, activation = 'relu'))
gru_model.add(Dense(64, activation = 'relu'))
gru_model.add(Dense(16, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
gru_model.add(Dense(2, activation = 'sigmoid'))

gru_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0005, patience=8, 
                                           restore_best_weights=True )
history = gru_model.fit(X_train, y_train, batch_size = 128, validation_data=(X_test, y_test), 
                        epochs=115)
# %%
pred = gru_model.predict
df = pd.read_csv('../input/testing/test.csv', sep = '\n', ) #loading full test set
df.head()

# %%
input_shape = (16948, 301, 4)
x = tf.random.normal(input_shape)
y = keras.layers.Conv1D(filters = 27, kernel_size = (4), activation = 'relu', input_shape = (301, 4))(x)
y = keras.layers.MaxPooling1D(pool_size = (3))(y)
y = keras.layers.Conv1D(filters = 14, kernel_size = (2), activation = 'relu', padding = 'same')(y)
y = keras.layers.Bidirectional(GRU(128, activation = 'relu'))(y)
print(y.shape)
# %%
input_shape = (16948, 301, 4)
x = tf.random.normal(input_shape)
y = Conv1D(filters = 30, kernel_size = (4), activation = 'relu', input_shape = (301, 4))(x)
y = MaxPooling1D(pool_size = (3))(y) # 99 * 60
y = Conv1D(filters = 15, kernel_size = (2), activation = 'relu', padding = 'same')(y) # 99 * 15
y = MaxPooling1D(pool_size = (3))(y) # 33 * 15
y = Conv1D(filters = 10, kernel_size = (2), activation = 'relu', padding = 'same')(y) # 33 * 10
print(y.shape)
y = Flatten()(y)
print(y.shape)
# %%
