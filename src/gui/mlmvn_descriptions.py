#!/usr/bin/env python
#encoding=utf8

"""
This module stores description labels
for mlmvn networks and improves thus readability
and usability of gui_networks module
"""

MLMVN_DESC = {
    "ContinuousMLMVN":

    r"""Represents multi-layered network with all layers
    with simple neurons with continuous activation function.\n
    Together with RMSEandAccLearningDAP can be used for learning
    with soft margins (even for nominal attributes)\n
    <b>Parameters:</b>\n
    <i><b>Layers specification</b></i>\n
    -layers are specified by numbers of neurons in them in ascending order
    and separated by signs '->' to emphasize connection between layers\n
    -eg. <i>first hidden layer -> second hidden layer -> output layer</i>\n
    -might look like <i>50->8->2</i>\n
    ... 50 continuous neurons in first hidden layer\n
    ... 8 continuous neurons in second hidden layer\n
    ... 2 continuous output neurons\n
    <i><b>Initial learning speed</b></i>\n
    -complex/real number which specifies initial speed of learning. Use 1.0 for normal
    speed. If you want to use complex number use format "_real_ + _imag_ j" eg. "-2+3j".
    """,

    "DiscreteMLMVN":

    r"""Represents multi-layered network with all layers
    with neurons with the same discrete (step) activation function. This
    activation function divides complex plane into as many sectors as
    specified by <i>number of sectors</i> parameter. Output of neuron
    than gets assigned discrete number depending on sector to which it
    belongs. Even discrete output neurons can be used with numeric
    attributes to classify them into some bins/intervals.\n
    <b>Parameters:</b>\n
    <i><b>Number of sectors</b></i>\n
    -into how many sectors divide complex plane. Generally it should be at
    least max(number of values for output attributes), but it can be even
    more making reserve for 'unclassified' samples.\n
    <i><b>Layers specification</b></i>\n
    -layers are specified by numbers of neurons in them in ascending order
    and separated by signs '->' to emphasize connection between layers\n
    -eg. <i>first hidden layer -> second hidden layer -> output layer</i>\n
    -might look like <i>50->8->2</i>\n
    ... 50 neurons with discrete activation function in first hidden layer
    ... 8 neurons with discrete activation function in second hidden layer
    ... 2 output neurons with discrete activation function\n
    <i><b>Initial learning speed</b></i>\n
    -complex/real number which specifies initial speed of learning. Use 1.0 for normal
    speed. If you want to use complex number use format "_real_ + _imag_ j" eg. "-2+3j".
    """,

    "DiscreteLastLayerMLMVN":

    r"""Represents multi-layered network with all hidden layers conatining
    only simple continuous neurons and last layer
    with neurons with discrete (step) activation function. This
    activation function divides complex plane into as many sectors as
    specified by <i>number of sectors</i> parameter. Output of neuron
    than gets assigned discrete number depending on sector to which it
    belongs. Even discrete output neurons can be used with numeric
    attributes to classify them into bins/intervals.\n
    <b>Parameters:</b>\n
    <i><b>Number of sectors</b></i>\n
    -into how many sectors divide complex plane. Generally it should be at
    least max(number of values for output attributes), but it can be even
    more making reserve for 'unclassified' samples.\n
    <i><b>Layers specification</b></i>\n
    -layers are specified by numbers of neurons in them in ascending order
    and separated by signs '->' to emphasize connection between layers\n
    -eg. <i>first hidden layer -> second hidden layer -> output layer</i>\n
    -might look like <i>50->8->2</i>\n
    ... 50 continuous neurons in first hidden layer\n
    ... 8 continuous neurons in second hidden layer\n
    ... 2 output neurons with discrete activation function\n
    <i><b>Learning speed</b></i>\n
    -complex/real number which specifies initial speed of learning. Use 1.0 for normal
    speed. If you want to use complex number use format "_real_ + _imag_ j" eg. "-2+3j".
    """,
}
