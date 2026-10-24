import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt  # For hyperparameter tuning

# Load data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Extract feature columns (excluding 'TIM' and 'VAR_0')
    X = df.iloc[:, 2:].values  # All columns except 'TIM' and 'VAR_0' as features
    Y = df.iloc[:, 1].values   # 'VAR_0' column as target
    
    return df, X, Y

# Define a deeper and more regularized neural network model
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001)),  # L2 regularization
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),  # Normalize activations
        
        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Increased dropout
        
        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1)  # Output layer for FTSE 100 price prediction
    ])
    
    # Using Adam optimizer with learning rate scheduling
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Load dataset
file_path = 'data.csv'
df, X, Y = load_data(file_path)

# Identify missing target values
missing_indices = df[df['VAR_0'].isna()].index

# Remove rows with missing 'VAR_0' from training data
X_train_complete = df.dropna(subset=['VAR_0']).iloc[:, 2:].values
Y_train_complete = df.dropna(subset=['VAR_0']).iloc[:, 1].values

# Standardize input features
X_scaler = StandardScaler()
X_train_complete = X_scaler.fit_transform(X_train_complete)

# Standardize the target variable
Y_scaler = StandardScaler()
Y_train_complete = Y_scaler.fit_transform(Y_train_complete.reshape(-1, 1)).flatten()

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

# Build and train the model
model = build_model(input_dim=X_train_complete.shape[1])
history = model.fit(
    X_train_complete, Y_train_complete,
    epochs=200,  # Increased training epochs
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# Predict missing 'VAR_0' values
if not missing_indices.empty:
    X_missing = X_scaler.transform(df.iloc[missing_indices, 2:].values)
    Y_predicted_scaled = model.predict(X_missing)
    
    # Convert back to the original scale
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    # Fill missing values with predictions
    df.loc[missing_indices, 'VAR_0'] = Y_predicted.flatten()
    
    # Display the predicted values
    print("Predicted missing 'VAR_0' values:")
    print(df.loc[missing_indices, ['TIM', 'VAR_0']])

# Save updated data
df.to_csv("Generated_Data_filled.csv", index=False)
