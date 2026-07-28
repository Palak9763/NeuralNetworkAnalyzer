import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.Dense(features=10)(x)
        return x

rng = jax.random.PRNGKey(0)
model = SimpleCNN()
print(nn.tabulate(model, rng, console_kwargs={'width': 200, 'no_color': True})(jnp.ones((1,28,28,1))))
