# Optimizer

An **optimizer** is an algorithm or method used in machine learning and deep learning to adjust the weights and parameters of a model in order to minimize or maximize a specific objective function, typically a loss function

---

# Stochastic Gradient Descent (SGD)

SGD (Stochastic Gradient Descent) is **a variant of the Gradient Descent algorithm**, designed to optimize machine learning models by minimizing the loss function as effectively as possible.

Okay, before we dive into SGD, let's first understand what **Gradient** really means.

## What is Gradient ?

![image.png](Optimizer%2017ea5ca462ef808c9955db9cd3ab376b/image.png)

Imagine ,you are in trip hiking but your task is go to the deepest point to find treasue

from the picture imagine that is mountian where x ,y is parameter of model and z is loss.

**Gradient** is a vector that helps us identify the steepest (ชัน) path to the top of a mountain (the direction where y increases most rapidly). If we can use the Gradient to climb to the top, why not use it to find the lowest point of the mountain (the point of lowest loss)?

To do this, we simply move in the opposite direction of the Gradient. This idea can be summarized with the following equation:

![image.png](Optimizer%2017ea5ca462ef808c9955db9cd3ab376b/image%201.png)

new_weight = old_weight - lr(The gradient of the error)

Okay, now we understand what gradient mean. Let’s go back to our SGD!

## What does SGD actually do?

Imagine we have a huge amount of data to handle. To update the gradient and optimize our weights, we’d need to use all the data at once for every step. This can be very time-consuming. So, how can we tackle this problem?

SGD to the rescue!

Instead of using the entire dataset, SGD samples a small batch of data to update our model. This saves time and computational resources. Theoretically, we don’t need the full dataset to calculate the gradient—we can approximate it with just a portion of the data.

This is another advantage of SGD: it doesn’t drastically change the gradient in one step but instead adjusts it **gradually** over time

![image.png](Optimizer%2017ea5ca462ef808c9955db9cd3ab376b/image%202.png)

for more infomation [https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)

## Code Section

To use SGD, we just need to code this:

```python
model.compile(
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.1))
```

---

# Optimizers

![image.png](Optimizer%2017ea5ca462ef808c9955db9cd3ab376b/image%203.png)