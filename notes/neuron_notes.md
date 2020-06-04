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
    It should take a vector, x, of 2 x-values
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
    For simplicity a sigmoid function will be used, as it seems to be
    the most well documented.
            σ(x) = 1 / (1 + exp(-x))

### Cost function
    Typically, this is a sum-of-squares function: Σ(y - ŷ)²
    where y is the actual value, and ŷ is predicted by the algorithm.

    Here we will define cost *J* as J(W) =  ½(y - ŷ)²
    The '½' is arbitrary, but simplifies the differentiation, later.
    Note that *J* is a funtion of the weights, as they are the variable
    that we will be updating.

### Update Weights
    Weights will be updated in order to minimise the cost function
    using gradient descent.
    New weight = weight + derivative of cost * learning rate

            W := W + ∂J/∂W * α

    The learning rate, α, is some small value to dampen the effect
    of updates. A typical value woud be 0.1

    To find the derivative of the cost ∂J/∂W is complex:
        firstly, it will involve the chain rule, as J is not
        a function of W, directly.
        Cost = J(ŷ) =  ½ Σ(y - ŷ)²     y is a constant
        ŷ = σ(h) = 1 / (1 + exp(-h))
        h = f(w) = w₁ * x₁ + w₂ * x₂ + b

    By the chain rule:  ∂J/∂w₁ = ∂J/∂σ ・ ∂σ/∂h ・∂h/∂W
        ∂J/∂ŷ = ½(2)(y - ŷ) = (y - ŷ)・ŷ
        ∂ŷ/∂h = σ(h)・(1 - σ(h))
        ∂h/∂w₁ = x₁

    Therefore, in terms of variables that are readily available,
    ∂J/∂w₁ = -(y - ŷ)・ŷ(1 - ŷ)・x₁
