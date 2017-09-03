#https://bruceoutdoors.wordpress.com/2016/05/06/cs231n-assignment-1-tutorial-q2-training-a-support-vector-machine/
#trong optimization note
#code: https://github.com/machuiwen/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.                          3073*10
  - X: A numpy array of shape (N, D) containing a minibatch of data.              minibatch*3073
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means     minibatch        
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0] # minibatch
  loss = 0.0
  for i in xrange(num_train):
    #print('wshape:',X[i])
    scores = X[i].dot(W) #3073 .dot 3073*10 = 10x1
    #print('scores.shape:',scores.shape)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:   # y[i] correct class     
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        v = np.zeros((1, num_classes)) #1x10
        v[0, j] = 1 #incorrect class
        v[0, y[i]] = -1 #correct class
        u = np.resize(X[i], (X[i].shape[0], 1)) #X[i]=3073  u = 3073x1
        #print('ushape:',u.shape)
        dW += u.dot(v) #3073x1 .dot 1||-1

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  #loss += reg * np.sum(W * W)
  loss += 0.5 * reg * np.sum(W * W)   # them 1/2 ben ngoai cong thuc cua loss cho de dao ham
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
# doc optimization note

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  y: correct class
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]  #500  
  delta = 1.0
  scores = X.dot(W)  #500x3073 * 3073x10 =500x10.
  #print('chi lay nhung scores dung class, y chua so phan tu cua cot:',scores[np.arange(num_train), y])  500
  ##  C1:
  #scores[np.arange(num_train), y]: lay ra score cua cac  correct class , roi sau do moi resize
  correct_class_scores = np.resize(scores[np.arange(num_train), y], (num_train, 1)) # np.arange(num_train), y] (500) -> 500x1
  ##  C2:#######################################################################
  #correct_class_score = scores[[np.arange(num_train), y]]
  #margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + delta)    
  #print('correct_class_sccores:',correct_class_scores.shape)
  ############################################################################## 
  margins = scores - correct_class_scores + delta
  margins = np.maximum(0, margins)
  # the subtraction of delta is to correct for the contribution of the
  # correct class
  loss = np.sum(margins) / num_train -delta
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
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
#####Cach 1
  ##masktest=np.logical_and(margins, np.ones(margins.shape))#so sanh gia tri margins voi 1, chi can margins>0, thi and 1 =True
  ##print('masktest:',margins)
  ##print('masktest:',np.ones(margins.shape))
  ##print('masktest:',masktest)
  #mask = np.float64(np.logical_and(margins, np.ones(margins.shape)))# margins.shape =500x10, cho nao margins >0 thi mask =1
  ##print('mask:',margins.shape)
  ##mask[np.arange(num_train), y]): cai mask o vi tri cua tung correct class
  ##print('mask:',mask) #500x10
  ##print('mask.sum:',mask.sum(axis=1)) #500 moi gia tri la tong cua cac cot
  ##print('mask[np.arange(num_train), y].shape',mask[np.arange(num_train), y]) #500 gia tri tai correct class
  ## lay nhung diem ma correct class, se tru di tong cua nhung cai con lai, cua sample do  (bao gom ca diem correct luon) =>   [-(tongsigma cua (j=1->n, j #yi) 1(margin i,j<0)] 
  #mask[np.arange(num_train), y] -= mask.sum(axis=1)# axis=1, cong tong cac cot, [1,1,1; 1,1,1]->  [3,3]
  #print('maskafter:',mask)
  #dW = X.transpose().dot(mask) # day la buoc nhan voi xi va tong sigma ben ngoai
####### Cach 1 luoc bo phan giai thich
  #mask = np.float64(np.logical_and(margins, np.ones(margins.shape)))
  #print('mask:',mask)
  #mask[np.arange(num_train), y] -= mask.sum(axis=1)
  #dW = X.transpose().dot(mask)
  
  #dW /= num_train
  #dW += reg * W
###########
########Cach 2 de hieu hon, theo bai giai thich
  # Fully vectorized version. Roughly 10x faster.
  X_mask = np.zeros(margins.shape)
  # column maps to class, row maps to sample; a value v in X_mask[i, j]
  # adds a row sample i to column class j with multiple of v
  X_mask[margins > 0] = 1
  #print ('X_mask:',X_mask)
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] -= incorrect_counts
  dW = X.transpose().dot(X_mask)

  dW /= num_train # average out weights
  dW += reg*W # regularize the weights
########
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
