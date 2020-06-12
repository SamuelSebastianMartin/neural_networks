Single Neuron
=============

         _______
        |       |
  x₁ ---|w₁     |
        |       |---Output h = x₁⋅w₁ + x₂⋅w₂ + b)
  x₂ ---|w₂     |
        |_______|
            |
          bias


Concepts to implement
---------------------

### Input

    * Matrix, X, of features (independent variables)
        The number of features (columns) = n
        The number of data points (rows) = m
                _                   _
               | x₁⁽¹⁾, x₂⁽¹⁾, x₃⁽¹⁾ |
               | x₁⁽²⁾, x₂⁽²⁾, x₃⁽²⁾ |
               | x₁⁽³⁾, x₂⁽³⁾, x₃⁽³⁾ |
               |_x₁⁽⁴⁾, x₂⁽⁴⁾, x₃⁽⁴⁾_|

    * Vector, y, of labels, size m (for training and testing)

    * Bias (intercept) will be handled as one of the weights
      and generated in the class.
      By adding an aditional column of ones in X matrix,
      the corresponding weight will be the constant term.
                _                      _          _  _
               | 1, x₁⁽¹⁾, x₂⁽¹⁾, x₃⁽¹⁾ |        | b  |
           x = | 1, x₁⁽²⁾, x₂⁽²⁾, x₃⁽²⁾ |  wts = | w₁ |
               | 1, x₁⁽³⁾, x₂⁽³⁾, x₃⁽³⁾ |        | w₂ |
               |_1, x₁⁽⁴⁾, x₂⁽⁴⁾, x₃⁽⁴⁾_|        |_w₃_|


### Output

    * This will be a vector of weights corresponding to the
      coefficients of the features. In this way, as simple
      linear function y = mx + c would be
          X = [1, x₁]  wts = [c, m]

    * Some graphical representation of the decreasing error
      error against epochs (epoch = once through training data)

### Variables

    * epochs = number of times that the data loops in training

    * learning rate = a multiplier to reduce the change made
      by updating weights. Steps might be 0.01, 0.03, 0.1, 0.3 etc.

### Formula
    h = b + x₁·w₁ + x₂·w₂ + x₃·w₃  ...
    Note, it is the weights (coefficients) which are being calculated.
    In matrix terms, this will be X * wts, where wts is a column vector.

### Activation
    There will be no activation function as we are not categorising,
    but looking for exact values.

### Cost function
    Typically, this is a sum-of-squares function: 1/m * Σ(h - y)²
        where the sum is from i = 1 to m
        i.e. the sum of all the data items (rows) in X
        y is the actual value, and h is predicted: X * wts
    we will use: Cost, J(w) = 1/2m * Σ(h - y)²
    Note that J is a funtion of the weights, as they are the variable
    that we will be updating.

    In matrix implementation, the square can be achieved by multiplying
    the matrix with its own transpose:
        J(w) = 1/2m (X∙w - y)'(X∙w - y)
             = 1/2m (h - y)'(h - y)
    The cost is halved to simplify differentiation. This will
    not affect the results, as we now minimise cost/2.


### Update Weights
    Weights will be updated in order to minimise the cost function
    using gradient descent.
    New weight = weight + derivative of cost * learning rate

            w := w - α(∂J/∂w)

    The learning rate, α, is some small value to dampen the effect
    of updates.

    To calculate ∂J/∂w, we are differentiating
            J(w) = 1/2m * Σ(h - y)²
                 = 1/2m * Σ(X*w - y)²

        Note that the only multiplier of w is X, so
        We can write out the update rules for each element of w like this:

            w₀ = w₀ - α * 1/m * (h - y) * x₀  # x₀ = 1: bias term/consant
            w₁ = w₁ - α * 1/m * (h - y) * x₁
            w₂ = w₂ - α * 1/m * (h - y) * x₂
                                    ... * x₃
                                    ... * x₄
                                          The final term is X' (X transpose)
    In matrix terms,
            ∂J/∂w = X'(X*w - y)

    So the update equation becomes:

            w := w - α/m * X'(X*w - y)
