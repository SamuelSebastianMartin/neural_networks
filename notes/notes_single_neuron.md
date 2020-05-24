Single Neuron
=============

        ________
    ____|       |
        |       |___ Output
    ____|       |
        |_______|
            |
          bias

Output = (input 1 x weight 1) + (input 2 x weight 2)
        followed by an activation function.

Example An AI number guesser.
-----------------------------

The computer tries to guess the number 36.4 (say), after training on a varying amount of data.
  * Make training data: 100 tuples are generated, consisting of (a random number, a value to show 'higher' or 'lower' than 36.4'. The actual number should be excluded.
  *  The neuron is trained on the data, to generate the correct weight and bias.
  * The actual data is generated and classified.
  * The computer's guess is half way between the lowest 'higher' value and the highest 'lower' value.

### Functioning

  The output is 1 or 0 (1 = number too high, 0 = number too low)

#### Training
  The input is multiplied by the weight (no bias needed for something so silly?) to calculate *h*
  The weight is between -1 and 1.
  The activation function will be whether *h* is greater or less than 0.
    if 60 triggers a 0 (too low) then the weight will be reduced. That way, next time 60 will get a lower *h* value.
  The weight is adjusted in increments of (say) 0.1

#### Guessing
  The weight value from training will be used, and should give approximately correct results.
  A thousand random numbers are examined
  If they are deemed too high, they are appended to the 'higher' list; too low goes to the 'lower' list.
  The final guess will be halfway between the smallest value in the 'higher' list and the largest value in the 'lower' list.

### Pseudocode

```
training_data = []
    for n in range(100)
        point = random.integer(0,100)
        if point > 36.4:
            training_data.append((point,1))
        if point < 36.4:
            training_data.append((point,0))

set_initial_weights()

class Neuron
    def __init__
        weight = 0
        bias = 0.5
        training_data = get_training_data()
        test_data = get_test_data()
    def train()
        for n in training_data:
            guess = guess()
            if guess > n[1]:
                update(minus)
            if guess < n[1]:
                update(plus)
    def update()
    def guess()
        for n in test_data:
            guess = guess()
            if guess > 0:
                higher.append(guess)
            if guess < 0:
                lower.append(guess)
```
