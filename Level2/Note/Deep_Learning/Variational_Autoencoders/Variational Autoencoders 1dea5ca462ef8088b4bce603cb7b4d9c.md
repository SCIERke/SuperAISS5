# Variational Autoencoders

# Autoencoder

An Autoencoder is an unsupervised learning model. Its goal is to create an Encoder and a Decoder such that the output is as close as possible to the original input.

At first, this might sound silly — why would we want the input and output to be the same? Imagine the model is trained to encode and reconstruct images of dogs.

If we then input a cat image, the model's reconstruction will be poor, and the loss will increase significantly. **This behavior makes Autoencoders useful for tasks like anomaly detection and feature learning, because they learn to capture the most important features of the data they were trained on.**

![image.png](Variational%20Autoencoders%201dea5ca462ef8088b4bce603cb7b4d9c/image.png)

# Variational Autoencoder (VAE)

## Limit of Autoencoder

![image.png](Variational%20Autoencoders%201dea5ca462ef8088b4bce603cb7b4d9c/image%201.png)

From the above image infer like this:

- **Each digit (e.g., "5" and "0") is encoded into a latent vector** in a compressed latent space by the encoder part of the autoencoder.
- The dashed line represents **linear interpolation** in the latent space between two encoded points — in this case, moving from "5" to "0".
- When you decode a point **in between** these two (like the middle of the dashed line), the result might not be meaningful (as shown in the weird digit).

### The Limitation This Shows:

Autoencoders **don’t structure the latent space** in a way that ensures smooth transitions between data points. So:

- Interpolating between two valid latent vectors can result in **unrealistic or blurry outputs**.
- There is **no guarantee that a point between two valid encodings corresponds to a valid image** (digit, face, etc).

## VAE could help us!

VAE help us mapping input into a distribution instead of mapping the it into a fixed vector,

![image.png](Variational%20Autoencoders%201dea5ca462ef8088b4bce603cb7b4d9c/image%202.png)

Variational Autoencoders (VAEs) **force the latent space to follow a normal distribution**, making it:

- More continuous
- More meaningful to interpolate within
- Better for generation and smooth transitions

## Reparameterization trick

Backpropagation work with deterministic node, not for stochastic node

![image.png](Variational%20Autoencoders%201dea5ca462ef8088b4bce603cb7b4d9c/image%203.png)

In **VAEs**, we don’t just encode input data into a fixed point in latent space. Instead, we encode it into a **distribution** (usually a Gaussian: mean `μ` and standard deviation `σ`).

So instead of sampling a fixed point like: 

`z = Encoder(x) (normal function)`

we sample it from `z ~ N(μ, σ²)`

From this calculation method of VAEs, we will found the problem that

**Backpropagation (used to train neural networks) only works with deterministic operations.**

But sampling from a probability distribution (i.e. drawing random values) is **stochastic**, and gradients can’t flow through random sampling.

so we use Reparameterization trick to fix this problem to make it work with backpropagation, we rewrite the sampling like this:

$z = μ + σ * ε$

`ε ~ N(0, I)` is just random noise sampled from a standard normal distribution.

and VAE use KL divergence calculation to calculate loss (Fuck you mom i will not explain how to prove it)

![image.png](Variational%20Autoencoders%201dea5ca462ef8088b4bce603cb7b4d9c/image%204.png)