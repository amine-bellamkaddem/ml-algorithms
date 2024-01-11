'''
This script is linked to the Medium post "Linear Predictors - The Perceptron, From Theory to Algorithms".

Author: Amine Bellamkaddem 
    - Medium: https://medium.com/@abellamk
    - LinkedIn: https://www.linkedin.com/in/amine-bellamkaddem-46a59312/
'''

import numpy as np
from utils import get_random_dataset, show_plot

def main():
    # Generate a random linearly separable dataset (100 data points)
    dataset = get_random_dataset(100)
    examples = dataset[0]
    labels = dataset[1]

    # Display the generated dataset
    show_plot(dataset)

    # Prepend a constant of value 1 to the input vector
    inputs = np.insert(examples, 0, 1, axis=1)

    # Initialize the parameter vector to be all-zero.
    # The first element of w is the bias term
    w = np.zeros(3)

    for i, x in enumerate(inputs):
        # Compute the dot-product of w and x
        y_pred = np.dot(w, x)

        # Get the sign of y_pred
        # You can use `numpy.sign` but you need to handle the case when y_pred = 0
        y_sign = -1 if y_pred < 0 else 1

        # Check if there is a mistake
        if y_sign != labels[i]:
            # Update parameters
            w += labels[i] * x

    print('*' * 50)
    print('Parameters:', w)
    print('*' * 50)
    show_plot(dataset, w, True)

if __name__ == "__main__":
    main()
