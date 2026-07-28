import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        return nn.Dense(features=10)(x)

model = SimpleCNN()
rng = jax.random.PRNGKey(0)
vars = model.init(rng, jnp.ones((1, 28, 28, 1)))

def apply_fn(params, x):
    return model.apply(params, x)

jaxpr = jax.make_jaxpr(apply_fn)(vars, jnp.ones((1, 28, 28, 1)))
print(jaxpr)

