import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # The analyzer looks for these specific neural network layers
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        
        # Flatten and pass through Dense layers
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

model = SimpleCNN()
