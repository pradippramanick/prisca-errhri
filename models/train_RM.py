import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score

set_seed(42)


def prepare_seqs(loaded):
    x_sequences = loaded[loaded.files[0]]
    y_sequences = loaded[loaded.files[1]]
    sessions = loaded[loaded.files[2]]
    return x_sequences, y_sequences, sessions


loaded = np.load('../data/val_org_split_RobotMistake.npz',
                 allow_pickle=True)
val_x_sequences, val_y_sequences, val_sessions = prepare_seqs(loaded)

loaded = np.load('../data/train_org_split_RobotMistake.npz',
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
learning_rate = 0.0001
batch_size = 512
epochs = 5
l2_rate = 0.01
patience = 3
feature_dim = 90
model_name = 'multimodal_LSTM_' + str(units) + '_RM.keras'

X_train = train_x_sequences
y_train = train_y_sequences
X_val = val_x_sequences
y_val = val_y_sequences

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


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
output = Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(l2_rate))(x)

# Define the model
model = Model(inputs=[input_modality_1, input_modality_2, input_modality_3], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print the model summary
model.summary()


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
        val_f1 = f1_score(self.validation_data[1], val_pred, average='macro', zero_division=0)
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
    filepath=model_name,
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