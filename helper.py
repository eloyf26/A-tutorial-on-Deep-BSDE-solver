import numpy as np
import tensorflow as tf

# Define constants

# SAMPLING constants
DTYPE = np.float64
DIMENSION = 100  # Problem dimensionality
NUM_TIME_STEPS = 40  # Number of time steps
TIME_GRID = np.linspace(0, 1, NUM_TIME_STEPS)  # Time discretization
TIME_STEP_SIZE = 1.0 / (NUM_TIME_STEPS - 1)  # Time step size
VOLATILITY = 0.2  # Example volatility parameter
INITIAL_X = 100.0  # Initial value for x
MEAN_RETURN = 0.02  # Mean return for x

# EQUATION constants
Y_INIT_MAX = 56
Y_INIT_MIN = 55
INTEREST_RATE = 0.02
DISCOUNT_FACTOR = 2.0/3
HIGH_GAMMA = 0.2
LOW_GAMMA = 0.02
HIGH_THRESHOLD = 50
LOW_THRESHOLD = 70

# NETWORK constants
LEARNING_RATE = 1e-2
BATCH_SIZE = 16
EPOCH_LIMIT = 6000


# Define the DefaultRiskPricingModel class using TensorFlow Keras
class DefaultRiskPricingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dim = DIMENSION
        self.num_time_interval = NUM_TIME_STEPS
        self.delta_t = TIME_STEP_SIZE
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.initial_x = INITIAL_X
        self.volatility = VOLATILITY
        self.interest_rate = INTEREST_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.high_gamma = HIGH_GAMMA
        self.low_gamma = LOW_GAMMA
        self.mean_return = MEAN_RETURN
        self.high_threshold = HIGH_THRESHOLD
        self.low_threshold = LOW_THRESHOLD
        self.gradient_slope = (self.high_gamma - self.low_gamma) / (self.high_threshold - self.low_threshold)

        # Initialize the network for initial value approximation
        self.Y_0 = tf.Variable(np.random.uniform(low=Y_INIT_MIN,
                                                 high=Y_INIT_MAX,
                                                 size=[1]).astype(DTYPE))
        
        # Initialize the network for initial Z approximation
        self.Z_0 = tf.Variable(np.random.uniform(-0.1, 0.1, size=(1, self.dim)).astype(DTYPE))

        # Define helper functions for creating layers
        def dense_layer(units):
            return tf.keras.layers.Dense(units=units, activation=None, use_bias=False)

        def batch_norm_layer():
            return tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-6,
                                                      beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                                                      gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))

        # Initialize gradient estimators at each time step
        # This code is initializing what is shown in Figure 1.4
        self.gradient_estimators = []
        for _ in range(self.num_time_interval - 1):
            gradient_estimator = tf.keras.Sequential()
            gradient_estimator.add(tf.keras.layers.Input(self.dim))
            gradient_estimator.add(batch_norm_layer())
            for _ in range(2):
                gradient_estimator.add(dense_layer(self.dim + 10))
                gradient_estimator.add(batch_norm_layer())
                gradient_estimator.add(tf.keras.layers.ReLU())
            gradient_estimator.add(dense_layer(self.dim))
            gradient_estimator.add(batch_norm_layer())
            self.gradient_estimators.append(gradient_estimator)
            
    # Define the generator function f
    def forward_function(self, time_step, state, value, gradient):
        piecewise = tf.nn.relu(
            tf.nn.relu(value - self.high_threshold) * self.gradient_slope + self.high_gamma - self.low_gamma) + self.low_gamma
        return (-(1 - self.discount_factor) * piecewise - self.interest_rate) * value

    # Define terminal condition g
    def terminal_condition(self, state):
        return tf.reduce_min(state, axis=1, keepdims=True)

    # Function to generate sample data
    def generate_samples(self, num_samples):
        increments = np.random.normal(loc=0.0, scale=np.sqrt(self.delta_t), size=(num_samples, self.dim, self.num_time_interval)).astype(DTYPE)
        samples = np.zeros((num_samples, self.dim, self.num_time_interval + 1), dtype=DTYPE)
        samples[:, :, 0] = np.ones((num_samples, self.dim)) * self.initial_x
        for i in range(self.num_time_interval):
            samples[:, :, i + 1] = (1 + self.mean_return * self.delta_t) * samples[:, :, i] + (self.volatility * samples[:, :, i] * increments[:, :, i])
        return samples, increments

    # Function for forward pass
    def forward_pass(self, inputs):
        samples, increments = inputs
        num_samples = samples.shape[0]
        ones_vector = tf.ones(shape=[num_samples, 1], dtype=DTYPE)
        
        y_values = ones_vector * tf.identity(self.Y_0)
        gradient_values = ones_vector * self.Z_0

        # Iterate through time steps
        for i in range(self.num_time_interval - 1):
            time_step = TIME_GRID[i]
            
            # Compute updates (use Equation 1.22)
            term1 = self.forward_function(time_step, samples[:, :, i], y_values, gradient_values) * self.delta_t
            term2 = tf.reduce_sum(tf.cast(gradient_values, DTYPE) * increments[:, :, i], axis=1, keepdims=True)
            y_values = y_values - term1 + term2
            
            # Update gradient approximations to take dimension into account (just a trick to improve convergence)        
            gradient_values = self.gradient_estimators[i](samples[:, :, i + 1]) / self.dim

        # Final update
        final_time_step = TIME_GRID[self.num_time_interval - 1]
        term1 = self.forward_function(final_time_step, samples[:, :, -1], y_values, gradient_values) * self.delta_t
        term2 = tf.reduce_sum(tf.cast(gradient_values, DTYPE) * increments[:, :, -1], axis=1, keepdims=True)
        y_values = y_values - term1 + term2

        return y_values

    # Function to compute loss
    def calculate_loss(self, inputs):
        samples, _ = inputs
        
        # Step 3 in Figure 1.5
        print(samples.shape)
        predicted_values = self.forward_pass(inputs)
        
        # Step 4 in Figure 1.5
        print(predicted_values)
        terminal_values = self.terminal_condition(samples[:, :, -1])
        loss = tf.reduce_mean(tf.square(terminal_values - predicted_values))
        
        return loss

    # Function to compute gradients
    @tf.function
    def get_gradients(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.calculate_loss(inputs)
            
        # Step 4 (Compute the gradient of the loss w.r.t the weights)
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients
  
