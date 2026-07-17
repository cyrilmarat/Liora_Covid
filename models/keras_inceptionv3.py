# %%
# Pour la manipulation de tableaux
import numpy as np

# Imports pour InceptionV3 en transfer learning avec l'API fonctionnelle de Keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
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


# %%
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%
# InceptionV3 attend des images RGB (3 canaux) -> color_mode='rgb' au lieu de 'grayscale'
# La taille (299, 299) est déjà celle attendue nativement par InceptionV3, donc on la conserve

train_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/train_augmented/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    seed=42,
    color_mode='rgb'
)

# %%
val_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/validation/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    shuffle=False,
    color_mode='rgb'
)

# %%
test_ds = image_dataset_from_directory(
    directory="../../COVID-19_Radiography_Dataset_split/test/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    seed=42,
    shuffle=False,
    color_mode='rgb'
)

# %%
# InceptionV3 a son propre preprocessing (normalisation entre -1 et 1),
# on l'applique en pipeline plutôt qu'avec une couche Rescaling manuelle
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# à calculer sur les labels du train ( les proportions sont les memes malgré l'augmentation )
y_train = np.concatenate([labels for images, labels in train_ds], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# %%
from tensorflow.keras.callbacks import Callback
from timeit import default_timer as timer

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(
                                patience=5,
                                min_delta=0.01,
                                verbose=1,
                                mode='min',
                                monitor='val_loss')

reduce_learning_rate = ReduceLROnPlateau(
                                    monitor="val_loss",
                                    patience=3,
                                    min_delta=0.01,
                                    factor=0.1,
                                    cooldown=4,
                                    verbose=1)

time_callback = TimingCallback()

# %%
# Chargement du modèle InceptionV3 pré-entraîné sur ImageNet, sans le head de classification
# (include_top=False) pour reconstruire notre propre tête adaptée aux 4 classes
inputs = Input(shape=(299, 299, 3), name="Input")

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

# On gèle les poids du réseau de base pour une première phase d'entraînement
# (seul le head de classification est entraîné)
base_model.trainable = False

# %%
# Tête de classification (équivalent des layers 13 à output_layer du script original)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(rate=0.3)(x)
x = Dense(units=128, activation='relu', name='dense_hidden_layer')(x)
x = Dropout(rate=0.3)(x)
output_layer = Dense(units=4, activation='softmax', name='output_layer')(x)

model = Model(inputs=inputs, outputs=output_layer)

# %%
# Phase 1 : entraînement du head uniquement, base_model gelé
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_history = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=50,
                          callbacks=[reduce_learning_rate,
                                     early_stopping,
                                     time_callback],
                          class_weight=class_weight_dict,  # <-- ajouté
                          shuffle=False)

model.save('inceptionv3_head_16072026_v3.keras')

# %%
# Phase 2 (optionnelle) : fine-tuning — on dégèle les dernières couches du base_model
# et on ré-entraîne avec un learning rate faible pour affiner les features à la marge
base_model.trainable = True

# On ne dégèle que les couches les plus profondes (les plus proches de la sortie)
# pour éviter de détruire les features génériques apprises sur ImageNet
fine_tune_at = 249  # InceptionV3 compte ~311 layers ; à ajuster selon le budget de calcul
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR réduit pour le fine-tuning
              metrics=['accuracy'])

fine_tune_history = model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=20,
                              callbacks=[reduce_learning_rate,
                                         early_stopping,
                                         time_callback],
                              class_weight=class_weight_dict,  # <-- ajouté
                              shuffle=False)

model.save('inceptionv3_finetuned_16072026_v3.keras')

# %%
train_acc = fine_tune_history.history['accuracy']
val_acc = fine_tune_history.history['val_accuracy']

# %%
# Labels des axes
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Courbe de la précision sur l'échantillon d'entrainement (phase de fine-tuning)
plt.plot(
         fine_tune_history.history['accuracy'],
         label='Training Accuracy',
         color='blue')

# Courbe de la précision sur l'échantillon de validation (phase de fine-tuning)
plt.plot(
         fine_tune_history.history['val_accuracy'],
         label='Validation Accuracy',
         color='red')

# Affichage de la légende
plt.legend()

# Affichage de la figure
plt.show()
