import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

print("="*60)
print("BUILDING REGULARIZED LSTM MODEL")
print("="*60)

with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

EMBEDDING_DIM = 100
LSTM_UNITS = 64
DROPOUT_RATE = 0.5

model = Sequential([
    Embedding(
        input_dim=config['vocab_size'],
        output_dim=EMBEDDING_DIM,
        input_length=config['max_sequence_length']
    ),
    SpatialDropout1D(0.3, name='spatial_dropout'),
    Bidirectional(
        LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3),
        name='bidirectional_lstm'
    ),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(DROPOUT_RATE),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(config['num_emotions'], activation='softmax')
])

model.build(input_shape=(None, config['max_sequence_length']))

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nREGULARIZED MODEL SUMMARY:")
model.summary()

total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,} (regularized model)")

model.save('models/emotion_model_untrained.keras')

config['embedding_dim'] = EMBEDDING_DIM
config['lstm_units'] = LSTM_UNITS
config['dropout_rate'] = DROPOUT_RATE
config['total_params'] = int(total_params)

with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("\n✓ Regularized model and config saved!")
