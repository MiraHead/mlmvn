#!/usr/bin/env python
#encoding=utf8

'''
Module describing layers of neurons for multi-layered neural network.

Neurons have complex weights.

Layers are closely tied with networks that use them. Networks create them
and manage them. Although it is sort of design antipattern - here it's used
more for lighter code of networks... (and layer on itself is not of much use)

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
'''

import numpy as np
import cPickle as pickle


class MVNLayer():
    '''
    Layer of uniform multi-valued/continous neurons with complex weights

    Note:\n
    Counting of outputs is based on uniformity of neurons. For non-uniform
    layer (some neurons discrete, some continuous or with varying number
    and/or size of sectors) is better to implement layer differently or to make
    a wrapper above several layers.\n
    '''

    def __init__(self, n_neurons, n_neurons_prev_layer, learning_rate=1,
                 activation_func=None):
        ''' Initializes layer of neurons.

        @param n_neurons  Integer - number of neurons in layer.
        @param n_neurons_prev_layer Integer - number of neurons in
                                    previous layer.
        @param learning_rate    Floating number from interval (0,1] specifying
                                learning speed. Defaults to 1.
        @param activation_func  Activation function used to compute
                                output on sum of neuron's
                                inputs times weigths.\n
                                Input - numpy vector to which func. apply\n
                                Output - numpy vector of produced outputs\n
                                (both have same dimensions)\n
                                !!! If not specified, identity function will
                                be used resulting in continous activation
                                function.
        '''

        # Number of layer's inputs is number of neurons in previous
        # layer plus 1 for bias input
        n_layer_inputs = n_neurons_prev_layer + 1

        ## Numpy matrix where each column corresponds to neuron.
        # (Row corresponds to weight of one input through all
        # neurons in this layer.)\n
        # Indexing:  na_neurons[weight, neuron]\n
        # Defaultly random weights are set from interval [-0.5, 0.5]
        # for both imaginary and real part of weight.
        self.na_neurons = np.empty((n_layer_inputs, n_neurons))
        self.set_random_weights()

        ## Numpy array of computed errors
        self.na_errors = np.empty(n_neurons, dtype=complex)

        ## Numpy array storing previously computed zets (potentials) of neurons
        # in other words product of weights and input
        self.na_zets = np.empty(n_neurons, dtype=complex)

        ## MVNLayer used while backpropagating error
        self.upper_layer = None

        ## Learning rate
        self.learning_rate = learning_rate

        ## Normalization factor for error sharing principle
        self.norma_sharing = 1.0 / n_layer_inputs

        if activation_func is None:
            ## Activation function
            self.activation_func = lambda x: x
        else:
            self.activation_func = activation_func

    def set_upper_layer(self, layer):
        ''' Sets upper layer - for error correction '''
        self.upper_layer = layer

    def set_learning_rate(self, new_rate):
        ''' Learning rate setter '''
        self.learning_rate = new_rate

    def get_learning_rate(self):
        ''' Learning rate getter '''
        return self.learning_rate

    def count_outputs(self, inputs):
        ''' Counts outputs of last layer - just applies
        activation function on weighted sums for neurons.

        @see MLMVNLayer.count_zets
        '''

        # count activation for each neuron and sample
        return self.activation_func(self.count_zets(inputs))

    def count_zets(self, inputs):
        ''' Counts weighted sums (zets) for given layer

        Zets for all samples are computed at once - inputs are matrix.\n
        DO NOT USE WITH inputs AS VECTOR!!! - outputs matrix would be
        created with bad dimensions (because of speed/memory optimization)

        @param  inputs Numpy matrix of inputs for given neurons and samples.\n
                       Indexing: [sample, feature] (sample on row)
        @return        Numpy matrix of outputs.\n
                       Indexing: [outputs_for_sample, output_of_nth_neuron]
        '''

        outputs = np.empty((inputs.shape[0], self.na_neurons.shape[1]),
                           dtype=complex)

        # counts z for each neuron and sample
        np.dot(inputs, self.na_neurons[1:, ], out=outputs)
        # add bias inputs
        outputs += self.na_neurons[0, ]

        return outputs

    def count_outputs_for_update(self, inputs):
        ''' Counts outputs for given layer when input is vector
        and stores zets for learning alg.

        Use of this function is necessary before learning process. Function
        can be used as well in case that inputs are vector and not a matrix
        (even though zets are internally stored, function does not consume
        any unnecessary memory).

        @param  inputs Numpy vector of inputs for given layer.\n
        @return        Numpy vector of outputs of neurons.\n
        '''

        # counts z for each neuron and sample and store it
        np.dot(inputs, self.na_neurons[1:, ], out=self.na_zets)
        # add bias weights
        self.na_zets += self.na_neurons[0, ]

        # apply activation_func on each neuron's zet
        return self.activation_func(self.na_zets)

    #******BACKPROPAGATION********

    def count_errors(self):
        ''' Compute errors for all neurons in this layers.
        Errors are stored internally in MVNLayer.na_errors.
        '''
        # invert weights of upper layer
        inverted = np.conjugate(self.upper_layer.na_neurons[1:, :])
        factor = np.abs(self.upper_layer.na_neurons[1:, :])
        factor *= factor
        inverted /= factor

        # optimized matrix product of errors and inverted upper_layer weights
        # .T  means transpose matrix...
        np.dot(self.upper_layer.na_errors,
               inverted.T,
               out=self.na_errors)

        self.na_errors *= self.norma_sharing

    #******FORWARD UPDATE*********

    def update_weights(self, inputs):
        ''' Forward propagation of corrections.

        Updates MVNLayer.na_neurons appropriately.

        @param inputs       Numpy vector of inputs for layer - one sample!
        '''

        # divide errors by absolute values of previous zets
        self.na_errors /= np.abs(self.na_zets)
        self.na_errors *= self.learning_rate

        self.na_neurons[0, ] += self.na_errors  # ...update bias weights
        self.na_neurons[1:, ] += np.outer(inputs.conjugate(), self.na_errors)
        # see http://docs.scipy.org/doc/numpy/reference/
        # /generated/numpy.outer.html

    def get_number_of_outputs(self):
        ''' Returns number of outputs of layer. '''
        return self.na_neurons.shape[1]

    def get_number_of_inputs(self):
        ''' Returns number of inputs for layer (each neuron). '''
        return self.na_neurons.shape[0] - 1

    def set_random_weights(self):
        ''' Sets weights from interval [-0.5, 0.5] for both imaginary and real
        part of weight.
        '''
        old_shape = self.na_neurons.shape
        size_of_weights = old_shape[0] * old_shape[1]
        # <optimization> usage of += ops saves memory - no intermediate array
        self.na_neurons = np.random.random(size_of_weights) * 1j
        self.na_neurons -= 0.5j
        self.na_neurons += np.random.random(size_of_weights)
        self.na_neurons -= 0.5
        self.na_neurons = self.na_neurons.reshape(old_shape)

    def set_weights_by_func(self, init_weights_func):
        ''' Function that can be used to set different than random weights.\n

        @param init_weights_func Function that applied to numpy matrix of
                                 weights (MVNLayer.na_neurons) transforms
                                 them into desired ones.\n
        '''

        self.na_neurons = init_weights_func(self.na_neurons)

    # *******************  SAVING/LOADING *************

    def save_layer(self, out_file):
        ''' Saves layer's weights and learning rate to binary file with
        cPickle.

        @param out_file Binary file opened for writing.
        '''
        pickle.dump(self.learning_rate, out_file)
        pickle.dump(self.na_neurons, out_file)

    def load_layer(self, in_file):
        ''' Loads layer's weights and learning rate from binary file with
        cPickle.

        Layer should be initialized by network which uses it with proper
        dimensions for various numpy arrays before weights are loaded

        @param in_file Binary file opened for reading with pickled
                       representation of layer.
        '''
        self.learning_rate = pickle.load(in_file)
        self.na_neurons = pickle.load(in_file)


class MVNLastLayer(MVNLayer):
    '''
    Last layer is a special case of MLMVNLayer providing
    different error computation and slightly different
    updating of weights.
    '''

    def count_errors_last_layer(self, desired_outputs):
        ''' Compute errors for all neurons in last layers.

        Errors are stored internally in MVNLayer.na_errors.

        @param desired_outputs Numpy vector of desired outputs for all neurons
                               in last layer.\n
                               If bisector optimization is used, it is
                               passed as desired_outputs.
        '''

        # normalize zets to unit circle (suitable both for continuous
        # and discrete case of activation function)
        self.na_zets /= np.abs(self.na_zets)

        # count errors... note that in continuous case self.na_zets
        # contained actual outputs -> so everything is OK in both cases
        self.na_errors = (desired_outputs - self.na_zets)
        self.na_errors *= self.norma_sharing

    def update_weights(self, inputs):
        ''' Forward propagation of corrections.

        Updates MVNLayer.na_neurons appropriately.

        @param inputs       Numpy vector of inputs - one sample!
        @param last_outputs Previous outputs of neurons in layer.
        '''
        self.na_errors *= self.learning_rate

        self.na_neurons[0, ] += self.na_errors  # ...update bias weights
        self.na_neurons[1:, ] += np.outer(inputs.conjugate(), self.na_errors)
        # see http://docs.scipy.org/doc/numpy/reference/
        # /generated/numpy.outer.html
