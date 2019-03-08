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
  num_train = X.shape[0] #500 examples
  num_classes = W.shape[1] #10 classes
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores -= scores.max()
      correct_class_score = scores[y[i]]
      sum_exp = np.sum(np.exp(scores))
      loss += (-1)* np.log(np.exp(correct_class_score)/sum_exp)
      for j in xrange(num_classes):
          if j == y[i]:
              continue
          dW[:,j] += np.exp(scores[j]) / sum_exp * X[i]
      dW[:,y[i]] += (-1)* ((sum_exp - np.exp(scores[y[i]]))/sum_exp) * X[i]



  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2* reg * W
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
  num_train = X.shape[0] #500 examples
  num_classes = W.shape[1] #10 classes

  scores = X.dot(W) #500X10
  #print(np.max(scores, axis=1).shape)#500, each line vector
  #scores -= np.max(scores, axis=1).T[:,np.newaxis] #for numeric stability substract from each row max
  scores -= scores.max()
  sum_scores_j = np.sum(np.exp(scores), axis=1) #500, each line vector
  loss = (-1)*scores[np.arange(num_train), y] + np.log(sum_scores_j)

  f1 = np.exp(scores)/sum_scores_j.T[:,np.newaxis]
  #print(f1.shape)
  f1[np.arange(num_train),y] -= 1
  dW = X.T.dot(f1)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2* reg * W


  return np.sum(loss), dW
