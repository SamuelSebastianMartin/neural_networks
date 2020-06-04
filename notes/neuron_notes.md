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
    It should take a vector, X, of 2 x-values
    These can be supplied serially by the calling program.

### Output
    This will be a scalar (as it is a single neuron, not a network
    The output will represent a categorical decision, 0 or 1

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
    Typically, this is a sum-of-squares function: J = Σ(y - ŷ)²
    where y is the actual value, and ŷ is predicted by the algorithm.

### Weight updates
    Weights are updated to minimise the loss function J.
    Using gradient descent, the updates will be zero at the minimum.
        J(w) = (y - ŷ)² = (y - (X·w + b)),
        Gradient ∂J/∂w = (y - (X·w + b))*X,
    Thus, each weight will be updated separately



