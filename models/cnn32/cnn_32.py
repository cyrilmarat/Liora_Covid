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
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Pour encoder les labels
from tensorflow.keras.utils import to_categorical 

# Pour évaluer les performances 
from sklearn import metrics

# Pour visualiser les performances
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import keras



# %%
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%
train_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train_augmented/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    shuffle=True,
    seed=42,
    color_mode='grayscale'
    
)

# %%
train_ds_not_augmented = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    shuffle=True,
    seed = 42,
    color_mode='grayscale'
)



# %%
val_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/validation/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    seed = 42,
    shuffle=True,
    color_mode='grayscale'
)



# %%
test_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/test/",                                 
    image_size=(299, 299),
    batch_size = 32,
    labels="inferred",
    seed = 42,
    shuffle=True,
    color_mode='grayscale'
)

# à calculer sur les labels du train 
y_train_not_augmented = np.concatenate([labels for images, labels in train_ds_not_augmented], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_not_augmented), y=y_train_not_augmented)
class_weight_dict = dict(enumerate(class_weights))


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

@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    """F1Score de Keras adaptée aux labels sparses (entiers) plutôt que one-hot."""
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_ = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=self.num_classes_)
        return super().update_state(y_true, y_pred, sample_weight)

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.01,
    verbose=1,
    mode='max',
    monitor='val_f1_score')

reduce_learning_rate = ReduceLROnPlateau(
                                    monitor="val_f1_score",
                                    patience=3, # si val_f1_score stagne sur 3 epochs consécutives selon la valeur min_delta
                                    min_delta=0.01,
                                    factor=0.1,  # On réduit le learning rate d'un facteur 0.1
                                    cooldown=4,  # On attend 4 epochs avant de réitérer 
                                    mode='max',
                                    verbose=1)

time_callback = TimingCallback()

# %%
# Instanciation des couches
inputs = Input(shape=(299, 299, 1), name="Input")

normalization_layer=Rescaling(1./255)

# Un seul bloc de convolution (32 filtres) - niveau supplémentaire à 64 filtres retiré
layer1 = Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    name='conv_layer1',
    #kernel_regularizer=l2(1e-4)
)

layer2=BatchNormalization(name="BatchNormalization1")

layer3 = MaxPooling2D(
    pool_size=(2, 2),
    name='max_pooling_layer1'
)

layer4 = GlobalAveragePooling2D()
layer5 = Dropout(rate=0.3)

layer6 = Dense(
    units=128,
    activation='relu',
    name='dense_hidden_layer'
)

layer7 = Dropout(rate=0.3)

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



# Création du modèle
outputs = output_layer(x)
model = Model(inputs=inputs, outputs=outputs)

# %%


model.compile(loss='sparse_categorical_crossentropy', # fonction de perte
              optimizer='adam',                # algorithme d'optimisation
              metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')])            # métrique d'évaluation

model_history = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=50,
                          callbacks = [reduce_learning_rate,
                                       early_stopping,
                                       time_callback],
                          #class_weight=class_weight_dict,  # <-- ajouté             
                          shuffle=False) 

model.save('cnn_32.keras')




# %%
# Labels des axes
plt.xlabel('Epochs')
plt.ylabel('F1_Score')

# Courbe de la précision sur l'échantillon d'entrainement
plt.plot(
         model_history.history['f1_score'],
         label='Training f1_score',
         color='blue')

# Courbe de la précision sur l'échantillon de test
plt.plot(
         model_history.history['val_f1_score'], 
         label='Validation f1_score',
         color='red')

# Affichage de la légende
plt.legend()



plt.savefig('cnn_32.png')
np.save('cnn_32.npy',model_history.history)

# Affichage de la figure
plt.show()
