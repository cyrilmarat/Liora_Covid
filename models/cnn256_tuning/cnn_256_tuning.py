# %%
# pip install keras-tuner --break-system-packages   (si pas déjà installé)

import numpy as np
import tensorflow as tf
import keras
import keras_tuner as kt

from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv2D, MaxPooling2D, Rescaling,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.utils.class_weight import compute_class_weight
from timeit import default_timer as timer

tf.config.list_physical_devices('GPU')

# %%
# Mêmes datasets que cnn_256.py (batch_size fixe : le tuner ne le fait pas varier ici,
# car les datasets sont déjà batchés en amont)
train_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train_augmented/",
    image_size=(299, 299), batch_size=32, labels="inferred",
    shuffle=True, seed=42, color_mode='grayscale'
)

train_ds_not_augmented = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/train/",
    image_size=(299, 299), batch_size=32, labels="inferred",
    shuffle=True, seed=42, color_mode='grayscale'
)

val_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/validation/",
    image_size=(299, 299), batch_size=32, labels="inferred",
    seed=42, shuffle=True, color_mode='grayscale'
)

test_ds = image_dataset_from_directory(
    directory="../../../COVID-19_Radiography_Dataset_split/test/",
    image_size=(299, 299), batch_size=32, labels="inferred",
    seed=42, shuffle=True, color_mode='grayscale'
)

y_train_not_augmented = np.concatenate([labels for images, labels in train_ds_not_augmented], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_not_augmented), y=y_train_not_augmented)
class_weight_dict = dict(enumerate(class_weights))

# %%
@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    """F1Score de Keras adaptée aux labels sparses (entiers) plutôt que one-hot."""
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_ = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=self.num_classes_)
        return super().update_state(y_true, y_pred, sample_weight)


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


# %%
# Architecture à 4 blocs figée (cnn_256) ; seuls dropout / dense / L2 / learning_rate
# sont exposés à la recherche d'hyperparamètres.
def build_model(hp):
    reg_strength = hp.Choice('l2_reg', values=[0.0, 1e-4, 1e-3])
    regularizer = l2(reg_strength) if reg_strength > 0 else None

    dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    dense_units = hp.Choice('dense_units', values=[64, 128, 256])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    inputs = Input(shape=(299, 299, 1), name="Input")
    x = Rescaling(1. / 255)(inputs)

    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                   activation='relu', kernel_regularizer=regularizer,
                   name=f'conv_layer{i}')(x)
        x = BatchNormalization(name=f'BatchNormalization{i}')(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f'max_pooling_layer{i}')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout_1)(x)
    x = Dense(units=dense_units, activation='relu', name='dense_hidden_layer')(x)
    x = Dropout(rate=dropout_2)(x)
    outputs = Dense(units=4, activation='softmax', name='output_layer')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[SparseF1Score(num_classes=4, average='macro', name='f1_score')]
    )
    return model


# %%
# Hyperband : alloue peu d'epochs à beaucoup de configs, ne pousse que les meilleures
# jusqu'à max_epochs. factor=3 = ratio d'élimination classique.
early_stopping = EarlyStopping(
    patience=3, min_delta=0.01, verbose=1, mode='max', monitor='val_f1_score'
)

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_f1_score', direction='max'),
    max_epochs=30,
    factor=3,
    directory='kt_tuning',
    project_name='cnn256_hp_search',
    seed=42
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    callbacks=[early_stopping, TimingCallback()]
    # class_weight=class_weight_dict,  # décommenter si besoin
)

# %%
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Meilleurs hyperparamètres trouvés :", best_hp.values)

# %%
# Ré-entraînement complet du meilleur modèle avec le budget d'epochs habituel
best_model = tuner.hypermodel.build(best_hp)

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_f1_score", patience=3, min_delta=0.01,
    factor=0.1, cooldown=4, mode='max', verbose=1
)
early_stopping_final = EarlyStopping(
    patience=5, min_delta=0.01, verbose=1, mode='max', monitor='val_f1_score'
)

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[reduce_learning_rate, early_stopping_final, TimingCallback()],
    shuffle=False
)

best_model.save('cnn_256_tuned.keras')
np.save('cnn_256_tuned.npy', history.history)
