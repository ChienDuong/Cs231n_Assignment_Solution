import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
# giong trong huong dan https://bruceoutdoors.wordpress.com/2016/04/30/cs231n-assignment-1-tutorial-q3-implement-a-softmax-classifier/   (Dung phuong phap tinh loss nhu trong course note, va dung gradient nhu trong huong dan)
  for i in range(num_train):
    scores = X[i, :].dot(W)
    scores -= np.max(scores) # avoid numeric instability
    scoresExp = np.exp(scores)
    scoresNorm = scoresExp / scoresExp.sum()  
    loss += -np.log(scoresNorm[y[i]])
    for j in range(num_classes):
      dW[:, j] += -X[i] * ((j == y[i]) - scoresNorm[j])

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  scores -= np.resize(np.max(scores, axis=1), (num_train, 1))
  scoresExp = np.exp(scores)
  scoresNorm = scoresExp / np.resize(np.sum(scoresExp, axis=1), (num_train, 1))
  loss = -np.log(scoresNorm[np.arange(num_train), y]).sum()

  temp = np.zeros_like(scoresNorm)# tra ve array with same size but every element is 0
  #print("temp1:",temp)
  temp[np.arange(num_train), y] = 1# tat ca cac num_train se la cot, va chon ra nhung diem tai label y gan =1
  #print("temp2:",temp)
  dW += X.T.dot(scoresNorm-temp)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

