import numpy as np
import tensorflow as tf
from time import time
from helper import DefaultRiskPricingModel, DTYPE, LEARNING_RATE, BATCH_SIZE, EPOCH_LIMIT

# Configure learning rate and optimizer
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, epsilon=1e-8)

# Reference value for evaluation during training
reference_value = 57.300
training_log = []
print('  Iteration  Loss  RelativeError   AbsoluteError   |   Time  LearningRate')
start_time = time()

# Initialize the model (Step 1 in Figure 1.5)
model = DefaultRiskPricingModel()

# Training loop
for iteration in range(EPOCH_LIMIT):
    # Generate sample data (Step 2 in Figure 1.5)
    samples, increments = model.generate_samples(BATCH_SIZE)
    increments = tf.cast(increments, DTYPE)
    
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
    log_entry = (iteration, loss.numpy(), y_current, relative_error, absolute_error, elapsed_time, LEARNING_RATE)
    training_log.append(log_entry)
    
    # Print progress every 10 iterations
    if iteration % 10 == 0:
        print('{:5d} {:12.4f} {:8.4f} {:8.4f}  {:8.4f}   | {:6.1f}  {:6.2e}'.format(*log_entry))
