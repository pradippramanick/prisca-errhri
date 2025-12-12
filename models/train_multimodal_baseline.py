import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate, Bidirectional, Layer
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from metrics.err_hri_calc import get_metrics

set_seed(42)


def prepare_seqs(loaded):
    x_sequences = loaded[loaded.files[0]]
    y_sequences = loaded[loaded.files[1]]
    sessions = loaded[loaded.files[2]]
    return x_sequences, y_sequences, sessions


loaded = np.load('data/val_norm_seq5_UserAwkwardness.npz',
                 allow_pickle=True)
val_x_sequences, val_y_sequences, val_sessions = prepare_seqs(loaded)
print(val_x_sequences.shape)

loaded = np.load('data/train_norm_seq5_UserAwkwardness.npz',
                 allow_pickle=True)
train_x_sequences, train_y_sequences, train_sessions = prepare_seqs(loaded)

# Feature indices for each modality
modality_1_indices = list(range(35))  # 'AU01_r' to 'AU45_c'
modality_2_indices = list(range(35, 65))  # 'dist_4_7' to 'vel_dist_7_16'
modality_3_indices = list(range(65, 90))  # 'Loudness_sma3' to 'F3amplitudeLogRelF0_sma3nz'

# Hyperparameters
sequence_length = 5
units = 768
dropout_rate = 0.2
activation = 'sigmoid'
num_classes = 2
loss_function = 'categorical_crossentropy'
optimizer = Adam(learning_rate=0.0001)
batch_size = 512
epochs = 1
l2_rate = 0.001
patience = 10
feature_dim = 90
learning_rate = 0.0001

# Load your data
# Assume data is already split into training and validation sets
# X_train, X_val, y_train, y_val should be numpy arrays
# X_train, X_val shape: (num_samples, sequence_length, feature_dim)
# y_train, y_val shape: (num_samples, num_classes)
X_train = train_x_sequences
y_train = train_y_sequences
X_val = val_x_sequences
y_val = val_y_sequences

# Verify the shapes of the data
print("X_train shape:", X_train.shape)  # Expected: (237761, 5, 90)
print("y_train shape:", y_train.shape)  # Expected: (237761, 2)
print("X_val shape:", X_val.shape)  # Expected: (validation_samples, 5, 90)
print("y_val shape:", y_val.shape)  # Expected: (validation_samples, 2)

# One-hot encode the labels (if not already one-hot encoded)
if y_train.shape[1] != num_classes:
    y_train = to_categorical(y_train, num_classes)
if y_val.shape[1] != num_classes:
    y_val = to_categorical(y_val, num_classes)

# Input layers for each modality
input_modality_1 = Input(shape=(sequence_length, len(modality_1_indices)))
input_modality_2 = Input(shape=(sequence_length, len(modality_2_indices)))
input_modality_3 = Input(shape=(sequence_length, len(modality_3_indices)))

# LSTM layers for each modality
x1 = Bidirectional(
    LSTM(units, kernel_regularizer=l2(l2_rate), recurrent_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate)))(
    input_modality_1)
x2 = Bidirectional(
    LSTM(units, kernel_regularizer=l2(l2_rate), recurrent_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate)))(
    input_modality_2)
x3 = Bidirectional(
    LSTM(units, kernel_regularizer=l2(l2_rate), recurrent_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate)))(
    input_modality_3)

x = Concatenate()([x1, x2, x3])
x = Dropout(dropout_rate)(x)
output = Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(l2_rate))(x)

# Define the model
model = Model(inputs=[input_modality_1, input_modality_2, input_modality_3], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print the model summary
model.summary()


# Define the custom attention layer
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)


class F1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max'):
        super(F1Checkpoint, self).__init__()
        self.validation_data = validation_data
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = -np.Inf if mode == 'max' else np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_pred = self.model.predict(self.validation_data[0])
        val_pred = (val_pred.squeeze() > 0.5).astype(int)
        val_f1 = get_metrics(val_pred, self.validation_data[1])['f1']
        logs[self.monitor] = val_f1

        if self.verbose > 0:
            print(f'Epoch {epoch + 1}: val_f1 = {val_f1:.4f}')

        if self.save_best_only:
            current = val_f1
            if (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best):
                self.best = current
                if self.verbose > 0:
                    print(
                        f'Epoch {epoch + 1}: {self.monitor} improved to {self.best:.4f}, saving model to {self.filepath}')
                self.model.save(self.filepath)
        else:
            if self.verbose > 0:
                print(f'Saving model to {self.filepath}')
            self.model.save(self.filepath)


# Define callbacks
checkpoint_callback = F1Checkpoint(
    validation_data=(
        [X_val[:, :, modality_1_indices], X_val[:, :, modality_2_indices], X_val[:, :, modality_3_indices]], y_val),
    filepath='best_mm_model.h5',
    monitor='val_f1',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    verbose=1,
    restore_best_weights=True
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# Train the model
history = model.fit(
    [X_train[:, :, modality_1_indices], X_train[:, :, modality_2_indices], X_train[:, :, modality_3_indices]],
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(
        [X_val[:, :, modality_1_indices], X_val[:, :, modality_2_indices], X_val[:, :, modality_3_indices]], y_val),
    callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]
)
# Save the final model
#model.save('trained_mm_model_final.h5')
preds = model.predict(
    [X_val[:, :, modality_1_indices], X_val[:, :, modality_2_indices], X_val[:, :, modality_3_indices]])
preds = (preds.squeeze() > 0.5).astype(int)
print(preds.shape, y_val.shape)
print(get_metrics(preds, y_val))
print(classification_report(y_val, preds))

# Plotting training history
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('plot.png')
