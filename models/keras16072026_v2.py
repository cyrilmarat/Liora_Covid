# %%
# Pour la manipulation de tableaux
import numpy as np

# Imports nécessaires pour construire un modèle CNN avec l'API fonctionnelle de Keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Pour encoder les labels
from tensorflow.keras.utils import to_categorical 

# Pour évaluer les performances 
from sklearn import metrics

# Pour visualiser les performances
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns




# %%
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%
train_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/train_augmented/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    seed=42,
    color_mode='grayscale'
    
)


# %%
val_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/validation/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    shuffle=False,
    color_mode='grayscale'
)



# %%
test_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/test/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    seed = 42,
    shuffle=False,
    color_mode='grayscale'
)

# %%
from tensorflow.keras.callbacks import Callback
from timeit import default_timer as timer

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(
                                patience=5, # Attendre 5 epochs avant application
                                min_delta=0.01, # si au bout de 5 epochs la fonction de perte ne varie pas de 1%, 
    # que ce soit à la hausse ou à la baisse, on arrête
                                verbose=1, # Afficher à quel epoch on s'arrête
                                mode='min',
                                monitor='val_loss')

reduce_learning_rate = ReduceLROnPlateau(
                                    monitor="val_loss",
                                    patience=3, # si val_loss stagne sur 3 epochs consécutives selon la valeur min_delta
                                    min_delta=0.01,
                                    factor=0.1,  # On réduit le learning rate d'un facteur 0.1
                                    cooldown=4,  # On attend 4 epochs avant de réitérer 
                                    verbose=1)

time_callback = TimingCallback()

# %%
# Instanciation des couches
inputs = Input(shape=(299, 299, 1), name="Input")

normalization_layer=Rescaling(1./255)

#layer 1 à 3
layer1 = Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    name='conv_layer1',
)

layer2=BatchNormalization(name="BatchNormalization1")

layer3 = MaxPooling2D(
    pool_size=(2, 2),
    name='max_pooling_layer1'
)
#layer 4 à 6
layer4 = Conv2D(
    filters=64,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    name='conv_layer2',
)

layer5=BatchNormalization(name="BatchNormalization2")

layer6 = MaxPooling2D(
    pool_size=(2, 2),
    name='max_pooling_layer2'
)

#layer 7 à 9
layer7 = Conv2D(
    filters=128,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    name='conv_layer3',
)

layer8=BatchNormalization(name="BatchNormalization3")

layer9 = MaxPooling2D(
    pool_size=(2, 2),
    name='max_pooling_layer3'
)

layer10 = GlobalAveragePooling2D()
layer11 = Dropout(rate=0.3)

layer12 = Flatten()

layer13 = Dense(
    units=128,
    activation='relu',
    name='dense_hidden_layer'
)

layer14 = Dropout(rate=0.3)

output_layer = Dense(
    units=4,
    activation='softmax',
    name='output_layer'
)


# %%
# Utilisation des couches
x= normalization_layer(inputs)
x = layer1(x)
x = layer2(x)
x = layer3(x)
x = layer4(x)
x = layer5(x)
x = layer6(x)
x = layer7(x)
x = layer8(x)
x = layer9(x)
x = layer10(x)
x = layer11(x)
x = layer12(x)
x = layer13(x)
x = layer14(x)


# Création du modèle
outputs = output_layer(x)
model = Model(inputs=inputs, outputs=outputs)

# %%


model.compile(loss='sparse_categorical_crossentropy', # fonction de perte
              optimizer='adam',                # algorithme d'optimisation
              metrics=['accuracy'])            # métrique d'évaluation

model_history = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=50,
                          callbacks = [reduce_learning_rate,
                                       early_stopping,
                                       time_callback],
                          shuffle=False) 

model.save('cnn16072026_v2.keras')

train_acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']

# %%
model_history.history['accuracy'].__sizeof__()

# %%
# Labels des axes
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Courbe de la précision sur l'échantillon d'entrainement
plt.plot(
         model_history.history['accuracy'],
         label='Training Accuracy',
         color='blue')

# Courbe de la précision sur l'échantillon de test
plt.plot(
         model_history.history['val_accuracy'], 
         label='Validation Accuracy',
         color='red')

# Affichage de la légende
plt.legend()

# Affichage de la figure
plt.show()