# %%
# Pour la manipulation de tableaux
import numpy as np

# Imports nécessaires pour construire un modèle en transfer learning avec l'API fonctionnelle de Keras
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

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
# Note : color_mode='rgb' (les images grayscale sont dupliquées sur 3 canaux),
# requis pour les poids ImageNet du DenseNet121
train_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train_augmented/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

# %%
train_ds_not_augmented = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

# %%
val_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/validation/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    seed=42,
    shuffle=True,
    color_mode='rgb'
)

# %%
test_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/test/",
    image_size=(299, 299),
    batch_size=32,
    labels="inferred",
    seed=42,
    shuffle=True,
    color_mode='rgb'
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
        self.logs = []
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)

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
    patience=3,
    min_delta=0.01,
    factor=0.1,
    cooldown=4,
    mode='max',
    verbose=1)

time_callback = TimingCallback()

# %%
# Construction du modèle : base DenseNet121 pré-entraînée (ImageNet) + tête custom
inputs = Input(shape=(299, 299, 3), name="Input")
x = preprocess_input(inputs)  # normalisation attendue par DenseNet (pas de Rescaling manuel)

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
base_model.trainable = False  # phase 1 : base gelée

x = base_model(x, training=False)
x = GlobalAveragePooling2D(name="gap")(x)
x = Dropout(rate=0.3)(x)
x = Dense(units=128, activation='relu', name='dense_hidden_layer')(x)
x = Dropout(rate=0.3)(x)
outputs = Dense(units=4, activation='softmax', name='output_layer')(x)

model = Model(inputs=inputs, outputs=outputs)

# %%
# PHASE 1 : entraînement de la tête seule, base gelée
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[reduce_learning_rate, early_stopping, time_callback],
    shuffle=False
)

# %%
# PHASE 2 : fine-tuning — dégel des couches profondes, learning rate réduit
base_model.trainable = True

# On garde les premières couches gelées (features bas-niveau génériques)
# et on ne dégèle que la deuxième moitié du réseau
fine_tune_from = len(base_model.layers) // 2
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR bien plus faible pour le fine-tuning
    metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')]
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[reduce_learning_rate, early_stopping, time_callback],
    shuffle=False
)

model.save('densenet.keras')

# %%
# Fusion des historiques des 2 phases pour le tracé, avec repères de transition
f1_train = history_phase1.history['f1_score'] + history_phase2.history['f1_score']
f1_val = history_phase1.history['val_f1_score'] + history_phase2.history['val_f1_score']

n_head_epochs = len(history_phase1.history['f1_score'])
global_epochs = np.arange(1, len(f1_train) + 1)

# Meilleure époque de fine-tuning au sens de la validation
best_ft_local_epoch = int(np.argmax(history_phase2.history['val_f1_score'])) + 1
best_ft_global_epoch = n_head_epochs + best_ft_local_epoch

plt.figure(figsize=(14, 8))
plt.plot(global_epochs, f1_train, label='Entraînement', color='tab:blue')
plt.plot(global_epochs, f1_val, label='Validation', color='tab:orange')
plt.axvline(x=n_head_epochs + 0.5, color='steelblue', linestyle='--', label='Début du fine-tuning')
plt.axvline(x=best_ft_global_epoch, color='steelblue', linestyle='-.',
            label=f'Meilleur fine-tuning : époque {best_ft_local_epoch} (époque globale {best_ft_global_epoch})')
plt.scatter([best_ft_global_epoch], [f1_val[best_ft_global_epoch - 1]], color='steelblue', zorder=5)

plt.title('Evolution du F1 macro — DenseNet121\nPhase 1 et fine-tuning')
plt.xlabel('Époque globale')
plt.ylabel('F1 macro')
plt.ylim(0, 1.05)
plt.legend()
plt.savefig('densenet121_f1_combined.png', dpi=150, bbox_inches='tight')
np.save('densenet.npy', {'f1_score': f1_train, 'val_f1_score': f1_val})

plt.show()
