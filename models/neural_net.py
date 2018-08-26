"""
Module to teach implementation of
a feed forward neural network from scratch 
(no Deep learning libraries for Deep)
"""

import numpy as np


class NeuralNetwork(object):
    """
    Fully connected 2 hidden layer Deep neural network,
    with log loss and SGD (we can implement something more
    fancy later on)
    Attributes:
        input_size (int): Dimensonality of input
        batch_size (int): Number of training samples per batch
        hidden_size_1 (int): Size of first hidden layer
        hidden_size_2 (int): Size of second hidden layer
        output_size (int): Dimensionality of output
        reg (int): Strength of l2 regularization
        loss_func (str): Loss function, can be log loss or square loss
    """

    def __init__(self, input_size, batch_size, hidden_size_1, hidden_size_2, output_size, reg, loss_func):
        # Set up attributes (fields in Java are attributes in Python)
        # Key difference between fields and attributes: privacy doesnt exist in Python
        """
        Initialize neural network
        Args:
            input_size (int): Dimensonality of input
            batch_size (int): Number of training samples per batch
            hidden_size_1 (int): Size of first hidden layer
            hidden_size_2 (int): Size of second hidden layer
            output_size (int): Dimensionality of output
            reg (int): Strength of l2 regularization 
            loss_func (str): Can be either "log_loss" or "square_loss" 
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.reg = reg
        self.loss_func = self.log_loss if loss_func == 'log_loss' else self.square_loss


        # Set up our weight matrices
        # What should the size of our first weight matrix be? Its transforming
        # between input dimension to hidden size 1
        # Input is (batch_size x input_size); Transforming to (batch_size x hidden_size_1) so (input_size x hidden_size_1)


        # What about between the weight matrix between hidden size 1 and 2
        # From (batch_size x hidden_size_1) to (batch_size x hidden_size_2) so (hidden_size_1 x hidden_size_2)

        # What about between hidden size 2 and output
        # From (batch_size x hidden_size_2) to (batch_size x output_size) so (hidden_size_2 x output_size)

    def log_loss(self, y_hat, y, reg):
        """
        Calculate the l2 regularized log loss between y (truth) and y_hat (predictions)
        Args:
            y_hat: Predictions
            y: Truth
            reg: Strength of l2 regularization penalty
        Returns:
            Regularized log loss
            log_loss = 
        """

    def square_loss(selfs, y, y_hat, reg):
        """
        Calculate the l2 regularized square loss between y (truth) and y_hat (predictions)
        Args:
            y: Truth
            y_hat: Predictions
            reg: Strength of l2 regularization penalty

        Returns:
            Regularized square loss
                      
        """
