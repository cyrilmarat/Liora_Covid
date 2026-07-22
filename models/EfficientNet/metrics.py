import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class MacroMetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_ds, val_ds):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.train_f1 = []
        self.val_f1 = []

        self.train_precision = []
        self.val_precision = []

        self.train_recall = []
        self.val_recall = []

    def evaluate_dataset(self, dataset):

        y_true = []
        y_pred = []

        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)

            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        precision = precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred,
            average="macro"
        )

        return precision, recall, f1

    def on_epoch_end(self, epoch, logs=None):

        train_precision, train_recall, train_f1 = self.evaluate_dataset(self.train_ds)
        val_precision, val_recall, val_f1 = self.evaluate_dataset(self.val_ds)

        self.train_precision.append(train_precision)
        self.val_precision.append(val_precision)

        self.train_recall.append(train_recall)
        self.val_recall.append(val_recall)

        self.train_f1.append(train_f1)
        self.val_f1.append(val_f1)

        if logs is not None:
            logs["train_f1_macro"] = train_f1
            logs["val_f1_macro"] = val_f1

        print(
            f"\n"
            f"Train F1 : {train_f1:.4f} | "
            f"Val F1 : {val_f1:.4f}"
        )