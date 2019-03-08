import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero 3072=32x32x3 pixels picture  (3072+1)x10

  # compute the loss and the gradient
  num_classes = W.shape[1]  #10
  num_train = X.shape[0] #500 samples
  loss = 0.0
  for i in xrange(num_train): #iterate over samples (500)
    scores = X[i].dot(W) # fx = Xi*w    1x3073 * 3073X10 -> 1X10
    correct_class_score = scores[y[i]] # y[i] - lable 0 - 9
    counter = 0
    for j in xrange(num_classes): #iterate over classes 0 - 9
        if j == y[i]: #skip if j = yi (real label)
            continue
        margin_i = scores[j] - correct_class_score + 1 # note delta = 1
        if margin_i > 0:
            loss += margin_i
            #1) incorrect class gradient
            dW[:,j] += X[i]
            counter += 1
    #2) correct class gradient
    dW[:,y[i]] += (-1)*counter * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # 500X10
  num_train = X.shape[0] #500 samples
  correct_scores = scores[np.arange(num_train), y] # 500,
  margins = np.maximum(0, scores - correct_scores[:,np.newaxis] + 1)
  margins[np.arange(num_train), y]=0 #yi should be 0 (j = yi)

  loss = np.mean(np.sum(margins, axis=1))
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = margins
  binary[margins > 0]=1 #binarize
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T #correct classes
  dW=X.T.dot(binary)

  dW /= num_train

  dW += reg * W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
