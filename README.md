
# micrograd

Tiny automatic differentiation engine and neural network library, built from scratch in Python (with Python standard libraries).

The goal was to implement backpropagation by hand to verify a network built from basic operations can learn. The same principles, applied to tensors, are what libraries like PyTorch automate.

## Description
- Autograd engine (Value): Every number is a node in a computation graph. Every operation is recorded, so backward() can walk the graph in reverse to compute the gradients.
- Neural networks (Neuron, Layer, MLP): Multilayer perceptron built out of Value objects, so training requires no external libraries.
- Helper functions (mse, bce, cce, one_hot, softmax, train_step): Simplified model training.
- Save and load: Model weights can be saved to a text file and loaded back, with a check that the layers match.

## Toy problem: training on XOR
The function demo_xor(...) trains a full MLP to compute the XOR function.

With one output, the MLP easily achieves 100% accuracy. With multiple outputs (computing three pairwise XORs), the MLP converges and often reaches 100% accuracy within 1000 iterations.

This demo confirms that the library does indeed implement neural networks capable of learning non-linear functions.

## Requirements
Python 3.12 or newer (nested f-strings). No dependencies needed.

## Known limitations
- backward() is recursive, so a chain of over 1000 dependent operations will hit Python's recursion limit.
- The logarithm implementation fails with Value(0), so cross-entropy loss may fail if predicted probabilities are saturated.
- The project focuses on the fundamentals of deep learning, so modern techniques like AdamW or GELU are not implemented.

## Credits
The project was inspired by Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) lecture, and extends the idea with additional operations and activations, model saving/loading, and helper functions to create a complete library.