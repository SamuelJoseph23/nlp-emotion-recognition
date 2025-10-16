import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("="*60)
print("BUILDING LSTM MODEL")
print("="*60)

# Load configuration
print("\nLoading configuration...")
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

print(f"✓ Vocabulary size: {config['vocab_size']}")
print(f"✓ Sequence length: {config['max_sequence_length']}")
print(f"✓ Number of emotions: {config['num_emotions']}")
print(f"✓ Emotions: {config['emotions']}")

# ===== MODEL PARAMETERS =====
EMBEDDING_DIM = 128
LSTM_UNITS = 128
DROPOUT_RATE = 0.5

print("\n" + "="*60)
print("MODEL ARCHITECTURE PARAMETERS")
print("="*60)
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"LSTM units: {LSTM_UNITS}")
print(f"Dropout rate: {DROPOUT_RATE}")

# ===== BUILD MODEL =====
print("\n" + "="*60)
print("BUILDING BIDIRECTIONAL LSTM MODEL")
print("="*60)

model = Sequential([
    # Embedding layer - converts word indices to dense vectors
    Embedding(
        input_dim=config['vocab_size'],
        output_dim=EMBEDDING_DIM,
        input_length=config['max_sequence_length'],
        name='embedding_layer'
    ),
    
    # Spatial dropout - regularization for embedding layer
    SpatialDropout1D(0.2, name='spatial_dropout'),
    
    # Bidirectional LSTM - processes sequence in both directions
    Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=False),
        name='bidirectional_lstm'
    ),
    
    # Dropout for regularization
    Dropout(DROPOUT_RATE, name='dropout_1'),
    
    # Dense hidden layer
    Dense(64, activation='relu', name='dense_hidden'),
    
    # Dropout
    Dropout(0.3, name='dropout_2'),
    
    # Output layer - softmax for multi-class classification
    Dense(config['num_emotions'], activation='softmax', name='output_layer')
])

# ===== COMPILE MODEL =====
print("\nCompiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully")

# ===== MODEL SUMMARY =====
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# ===== SAVE MODEL ARCHITECTURE =====
print("\n" + "="*60)
print("SAVING MODEL ARCHITECTURE")
print("="*60)

# Save model architecture as JSON
model_json = model.to_json()
with open('models/model_architecture.json', 'w') as f:
    f.write(model_json)
print("✓ Saved model architecture to 'models/model_architecture.json'")

# Save initial model
model.save('models/emotion_model_untrained.keras')
print("✓ Saved untrained model to 'models/emotion_model_untrained.keras'")

# Update config with model parameters
config['embedding_dim'] = EMBEDDING_DIM
config['lstm_units'] = LSTM_UNITS
config['dropout_rate'] = DROPOUT_RATE
config['total_params'] = total_params

with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Updated configuration file")

print("\n" + "="*60)
print("MODEL BUILDING COMPLETE!")
print("="*60)
print(f"✓ Model architecture: Bidirectional LSTM")
print(f"✓ Total layers: {len(model.layers)}")
print(f"✓ Total parameters: {total_params:,}")
print("\n✓ Ready for training!")
print("="*60)
