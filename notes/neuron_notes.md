Single Neuron
=============

         _______
        |       |
  x₁ ---|w₁     |
        |       |---Output ŷ = σ((x₁⋅w₁ + x₂⋅w₂) + b)
  x₂ ---|w₂     |
        |_______|
            |
          bias

Output = (input 1 x weight 1) + (input 2 x weight 2) + ...  ... + bias
          put through some activation function.

Concepts to implement
---------------------

### Input
    It should take a vector, X, of x-values

### Output
    This will be a scalar (as it is a single neuron, not a network
    it will not output a vector.

### Formula
    ŷ = σ(X·w + b),
      where σ is the activation function.
      and ŷ is the predicted value of y
    This is the *feedforward* part of the process.

### Activation
    For simplicity a sigmoid function will be used, as it seems to
    the most well documented.
            σ(x) = 1 / (1 + exp(-x))

### Loss function
    Typically, this is a sum-of-squares function: Σ(y - ŷ)²
    where y is the actual value, and ŷ is predicted by the algorithm.
    Here, however, as there is only a single binary output, loss will
    be calculated using α(y - ŷ), where α is the learning rate. Thus

    * correct values (where ŷ = y) will lead to no change
        α(0 - 0) = α(1 - 1) = 0

    * incorrect values(where ŷ ≠ y) will lead to a change of ± α
        α(1 - 0) = α    -> positive change in w
        α(0 - 1) = -α   -> negative change in w

