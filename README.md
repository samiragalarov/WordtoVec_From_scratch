# Word2Vec Implementation

This project implements the Word2Vec model, a popular technique in natural language processing for learning distributed representations of words. The implementation includes tokenization, one-hot encoding, softmax function, forward pass, loss calculation, error calculation, backpropagation, and training. The model is trained on a given text corpus to learn vector representations of words in a continuous vector space.

## Tokenization

The `tokenize` function uses regular expressions to tokenize the input text into a list of words.

## Mapping

The `mapping` function creates two dictionaries: `idx_to_word` and `word_to_idx`, which map words to indices and vice versa.

## One-Hot Encoding

The `one_hot_encoding` function generates the training data in the form of a matrix for the Skip-gram model.

## Softmax

The `softmax` function computes the softmax values for a given input array.

## Forward Pass

The `forward_pass` function performs the forward pass of the Word2Vec model, computing the softmax output, hidden layer, and the inner product.

## Loss

The `loss` function calculates the negative log-likelihood loss.

## Error

The `error` function computes the error between the predicted output and the actual word representations.

## Backpropagation

The `backprop` function implements backpropagation to calculate the gradients for the weight matrices.

## Learning

The `learning` function updates the weights using gradient descent.

## Train

The `train` function trains the Word2Vec model for a specified number of epochs, using the provided training data, learning rate, and dimensionality.

## Predict

The `predict` function predicts the most similar words to a given word based on the learned word vectors.

## Visualization

The code includes a scatter plot to visualize the learned word vectors in a 2D space.

# Usage

To use the Word2Vec model:

```python
text = "Your input text here"
tokens = tokenize(text)
idxs, words = mapping(tokens)
mat = one_hot_encoding(tokens, words, window_size=2)

w1, w2, history = train(mat, words, lr=0.01, epochs=1000, dim=2)
predictions = predict('your_word', w1, w2, mat, words, idxs)
