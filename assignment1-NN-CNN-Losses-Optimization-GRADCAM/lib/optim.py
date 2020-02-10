from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        
        for layer in self.net.layers:
            for i,packed in enumerate(layer.params.items()):
                layer_name = packed[0]
                theta = packed[1]
                
                grad_params = layer.grads[layer_name]
                #vprev = self.velocity[layer_name]
                if layer_name not in self.velocity.keys():
                    self.velocity[layer_name] = np.zeros(grad_params.shape)
                    #vprev = self.velocity[layer_name]
                self.velocity[layer_name] = (self.momentum * self.velocity[layer_name]) - (self.lr*grad_params)
                layer.params[layer_name] += self.velocity[layer_name]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for layer in self.net.layers:
            for i,packed in enumerate(layer.params.items()):
                layer_name = packed[0]
                theta = packed[1]
                
                grad_params = layer.grads[layer_name]
                if layer_name not in self.cache.keys():
                    self.cache[layer_name] = np.zeros(grad_params.shape)
                self.cache[layer_name]=(self.decay*self.cache[layer_name]) + ((1-self.decay)*np.power(grad_params,2))
                num = self.lr * grad_params
                den = np.sqrt(self.cache[layer_name] + self.eps)
                layer.params[layer_name]=layer.params[layer_name] - num/den
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        
        self.t = self.t + 1
        for layer in self.net.layers:
            for i,packed in enumerate(layer.params.items()):
                layer_name = packed[0]
                theta = packed[1]
                
                grad_params = layer.grads[layer_name]
                if layer_name not in self.mt.keys():
                    self.mt[layer_name] = np.zeros(grad_params.shape)
                if layer_name not in self.vt.keys():
                    self.vt[layer_name] = np.zeros(grad_params.shape)
                self.mt[layer_name]  = (self.beta1*self.mt[layer_name]) + (1-self.beta1)*grad_params
                self.vt[layer_name] = (self.beta2*self.vt[layer_name]) + (1-self.beta2)*(grad_params**2)
                mh = self.mt[layer_name]/(1-(self.beta1**self.t))
                vh = self.vt[layer_name]/(1-(self.beta2**self.t))
                layer.params[layer_name] -= self.lr*(mh/(np.sqrt(vh) + self.eps))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
