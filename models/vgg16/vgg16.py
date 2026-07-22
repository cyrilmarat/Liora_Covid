# %%
# Pour la manipulation de tableaux
import numpy as np

# Imports pour VGG16 en transfer learning avec l'API fonctionnelle de Keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
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
import keras

# %%
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%
# VGG16 est nativement entraîné en 224x224 (contrairement à InceptionV3 qui utilise 299x299)
# et attend des images RGB (3 canaux) -> color_mode='rgb'

train_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train_augmented/",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    seed=42,
    shuffle=True,
    color_mode='rgb'
)

train_ds_not_augmented = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train/",                                 
    image_size=(224, 224),
    batch_size = 32,
    labels="inferred",
    shuffle=True,
    seed = 42,
    color_mode='rgb'
)


# %%
val_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/validation/",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    shuffle=True,
    color_mode='rgb'
)

# %%
test_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/test/",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    seed=42,
    shuffle=True,
    color_mode='rgb'
)


    

# %%
# InceptionV3 a son propre preprocessing (normalisation entre -1 et 1),
# on l'applique en pipeline plutôt qu'avec une couche Rescaling manuelle
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# à calculer sur les labels du train non augmenté
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
# Chargement du modèle VGG16 pré-entraîné sur ImageNet, sans le head de classification
# (include_top=False) pour reconstruire notre propre tête adaptée aux 4 classes
inputs = Input(shape=(224, 224, 3), name="Input")

base_model = VGG16(
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

model.compile(loss='sparse_categorical_crossentropy', # fonction de perte
              optimizer='adam',                # algorithme d'optimisation
              metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')])            # métrique d'évaluation

model_history = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=50,
                          callbacks = [reduce_learning_rate,
                                       early_stopping,
                                       time_callback],
                          class_weight=class_weight_dict,  # <-- ajouté             
                          shuffle=False) 

model.save('vgg16_head_16072026_v3.keras')

# %%
# Phase 2 (optionnelle) : fine-tuning — on dégèle le dernier bloc convolutif du base_model
# et on ré-entraîne avec un learning rate faible pour affiner les features à la marge
base_model.trainable = True

# VGG16 est organisé en 5 blocs (block1_conv1 ... block5_conv3, 19 layers au total).
# On ne dégèle que le dernier bloc (block5) pour éviter de détruire les features génériques
# apprises sur ImageNet, tout en gardant un coût de calcul raisonnable
fine_tune_at = 15  # index du premier layer de block5 dans base_model.layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR réduit pour le fine-tuning
               metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')])            # métrique d'évaluation

fine_tune_history = model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=20,
                              callbacks=[reduce_learning_rate,
                                         early_stopping,
                                         time_callback],
                              class_weight=class_weight_dict,  # <-- ajouté
                              shuffle=False)

model.save('vgg16_finetuned.keras')



# %%
# Labels des axes
plt.xlabel('Epochs')
plt.ylabel('f1_score')

# Courbe de la précision sur l'échantillon d'entrainement (phase de fine-tuning)
plt.plot(
         fine_tune_history.history['f1_score'],
         label='Training f1_score',
         color='blue')

# Courbe de la précision sur l'échantillon de validation (phase de fine-tuning)
plt.plot(
         fine_tune_history.history['val_f1_score'],
         label='Validation f1_score',
         color='red')

# Affichage de la légende
plt.legend()

plt.savefig('vgg16.png')
np.save('vgg16.npy',model_history.history)

# Affichage de la figure
plt.show()
