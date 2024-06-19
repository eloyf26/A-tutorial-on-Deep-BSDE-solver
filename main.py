import numpy as np
import tensorflow as tf
from time import time
from helper import DefaultRiskPricingModel, FLOAT_TYPE, LEARNING_RATE, BATCH_SIZE, EPOCH_LIMIT

# Configure learning rate and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate= tf.keras.optimizers.schedules.PiecewiseConstantDecay([3000],[LEARNING_RATE,LEARNING_RATE]))
# Reference value for evaluation during training
reference_value = 57.300
training_log = []
start_time = time()

# Initialize the model (Step 1 in Figure 1.5)
model = DefaultRiskPricingModel()
training_log = []

# Training loop
for iteration in range(EPOCH_LIMIT):
    # Generate sample data (Step 2 in Figure 1.5)
    samples, increments = model.generate_samples(BATCH_SIZE)
    increments = tf.cast(increments, FLOAT_TYPE)
    
    # Compute loss and gradients (Steps 3 and 4)
    loss, gradients = model.get_gradients((samples, increments))
    
    # Update model parameters (Step 4)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Current approximation (Grab current solution)
    y_current = model.Y_0.numpy()[0]
    
    # These are not a must, they are computed for logging purposes
    elapsed_time = time() - start_time
    absolute_error = np.abs(y_current - reference_value)
    relative_error = absolute_error / reference_value
    # Append log entry to training log
    training_log.append({
        'Iteration': iteration,
        'Loss': loss.numpy(),
        'Y Current': y_current,
        'Relative Error': relative_error,
        'Absolute Error': absolute_error,
        'Elapsed Time': elapsed_time,
        'Learning Rate': LEARNING_RATE
    })

    # Print progress every 10 iterations
    if iteration % 10 == 0:
        print(f"Iteration: {iteration}")
        print(f"Loss: {loss.numpy():.4f}")
        print(f"Y Current: {y_current:.4f}")
        print(f"Relative Error: {relative_error:.4f}")
        print(f"Absolute Error: {absolute_error:.4f}")
        print(f"Elapsed Time: {elapsed_time:.1f} seconds")
        print(f"Learning Rate: {LEARNING_RATE:.2e}")
        print("-" * 50)

