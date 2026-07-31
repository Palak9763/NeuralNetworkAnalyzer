import jax
import jax.numpy as jnp
from flax import linen as nn

# ---------------------------------------------------
# Transformer Block
# ---------------------------------------------------

class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):

        residual = x

        x = nn.LayerNorm()(x)

        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
        )(x)

        x = nn.Dropout(rate=self.dropout_rate)(
            x,
            deterministic=not train
        )

        x = x + residual

        residual = x

        x = nn.LayerNorm()(x)

        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)

        x = nn.Dense(self.embed_dim)(x)

        x = nn.Dropout(rate=self.dropout_rate)(
            x,
            deterministic=not train
        )

        return x + residual


# ---------------------------------------------------
# CNN Feature Extractor
# ---------------------------------------------------

class CNNEncoder(nn.Module):

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)

        x = nn.BatchNorm(use_running_average=True)(x)

        x = nn.avg_pool(
            x,
            window_shape=(2,2),
            strides=(2,2)
        )

        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)

        x = nn.BatchNorm(use_running_average=True)(x)

        x = nn.avg_pool(
            x,
            window_shape=(2,2),
            strides=(2,2)
        )

        return x


# ---------------------------------------------------
# Hybrid Vision Transformer
# ---------------------------------------------------

class HybridVisionTransformer(nn.Module):

    num_classes: int = 10

    @nn.compact
    def __call__(self, image, tokens, train=True):

        # CNN Branch

        cnn = CNNEncoder()(image)

        cnn = cnn.reshape((cnn.shape[0], -1))

        cnn = nn.Dense(256)(cnn)

        # Token Embedding

        embedding = nn.Embed(
            num_embeddings=5000,
            features=128
        )

        x = embedding(tokens)

        position = self.param(
            "position_embedding",
            nn.initializers.normal(),
            (1, tokens.shape[1], 128)
        )

        x = x + position

        # Transformer Stack

        for _ in range(4):

            x = TransformerBlock(
                embed_dim=128,
                num_heads=8,
                mlp_dim=512
            )(x, train)

        x = x.mean(axis=1)

        # Merge Both Branches

        merged = jnp.concatenate([cnn, x], axis=-1)

        residual = merged

        merged = nn.Dense(512)(merged)
        merged = nn.relu(merged)

        merged = nn.Dense(512)(merged)

        merged = merged + residual

        merged = nn.Dense(256)(merged)
        merged = nn.gelu(merged)

        logits = nn.Dense(self.num_classes)(merged)

        return logits


# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

model = HybridVisionTransformer()

key = jax.random.PRNGKey(0)

dummy_image = jnp.ones((2,64,64,3))

dummy_tokens = jnp.ones((2,32), dtype=jnp.int32)

params = model.init(
    key,
    dummy_image,
    dummy_tokens
)

output = model.apply(
    params,
    dummy_image,
    dummy_tokens,
    train=False
)

print("Output Shape:", output.shape)

print("Backend:", jax.default_backend())

print("Devices:", jax.devices())