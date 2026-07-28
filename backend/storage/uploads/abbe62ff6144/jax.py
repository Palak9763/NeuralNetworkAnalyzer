import jax
import jax.numpy as jnp

def main():
    print("--- 1. JAX NumPy and Device Allocation ---")
    # JAX arrays are immutable and automatically allocate onto GPUs/TPUs if available
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    print(f"Array x: {x}")
    print(f"Device: {x.device()}\n")

    print("--- 2. Automatic Differentiation (grad) ---")
    # Define a simple mathematical function: f(x) = x^3 + 2x
    def f(x):
        return x**3 + 2*x

    # Take the derivative df/dx. Evaluated at x=2.0, df/dx = 3(2)^2 + 2 = 14
    grad_f = jax.grad(f)
    print(f"f(2.0) = {f(2.0)}")
    print(f"df/dx at x=2.0: {grad_f(2.0)}\n")

    print("--- 3. Just-In-Time Compilation (jit) ---")
    # Define a complex operation to see performance optimization
    def heavy_ops(x):
        return jnp.sin(x) + jnp.cos(x) ** 2

    # Compile the function using XLA
    jitted_heavy_ops = jax.jit(heavy_ops)
    
    # Warm-up (triggers compilation)
    _ = jitted_heavy_ops(x)
    
    # Executing the compiled function
    result_jit = jitted_heavy_ops(x)
    print(f"JIT Result: {result_jit}\n")

    print("--- 4. Vectorization (vmap) ---")
    # Suppose we have a function designed for a single 1D vector
    def scale_and_offset(vector, scale, offset):
        return vector * scale + offset

    # A matrix representing a batch of 3 separate vectors
    batch_vectors = jnp.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    # vmap automates batching along the 0th axis without manual loops
    batched_scale_and_offset = jax.vmap(scale_and_offset, in_axes=(0, None, None))
    batch_result = batched_scale_and_offset(batch_vectors, 2.0, 10.0)
    print("Original Batch:\n", batch_vectors)
    print("Batched Result (scaled by 2, offset by 10):\n", batch_result)

if __name__ == "__main__":
    main()