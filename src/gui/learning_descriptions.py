#!/usr/bin/env python
#encoding=utf8

RMSE_ACC = r"""<b>Stopping criteria</b>\n
    <b><i>Accuracy on training set</i></b> Accuracy on learning set
    must be bigger than specified real number from interval [0,1] (if equals
    to 0, this parameter is not used at all) and\n
    <b><i>Accuracy on validation set</i></b> (same as previous but on
    validation set held out of learning) and\n
    <b><i>RMSE on training set</i></b> Root mean squared error represents mean
    angle error in radians. Can be specified experimentally for each
    network and dataset. If set to number bigger than 2 pi (6.28), it is not
    taken into account at all. Learning is stopped if RMSE on learning set
    is lower than specified treshold and\n
    <b><i>RMSE on validation set</i></b> The same as RMSE on training set but
    for validation set.\n\n
    <b>Samples for learning:</b>\n
    <b><i>% of samples to learn from</i></b> Specifies portion of incorrectly
    classified samples to learn from when each iteration is finished. Samples
    are chosen randomly. If equal to
    0.0 or less, only one random sample out of incorrect ones is used for
    learning.
    """

LEARNING_DESC = {
    "RMSEandAccLearning":

    r"""Learning for MLMVNs with discrete output, sample is incorrectly
    classified if the output of MLMVN is not in sector to whose bisector it
    was encoded. In each iteration RMSE and Accuracy metrics are counted on
    training set (and on validation set if it has at least one sample).\n
    In each iteration specified <i>% of incorrect samples to learn from</i>
    is used to update network's weights.\n
    In default settings this learning tries only to achieve 1.0 accuracy on
    training set (classify 100 % of samples from learning set correctly).\n
    Learning is stopped if all specified stopping criteria hold.\n\n
    """ + RMSE_ACC,

    "RMSEandAccLearningSM":

    r"""Learning for both discrete and continuous output MLMVNs.Sample is
    incorrectly classified if angle between the output of MLMVN and desired
    output (encoded output attribute(s)) is bigger than treshold specified
    by parameter <i><b>Angle precision</b></i>.\n
    This can be used for soft margins learning with discrete outputs, by
    setting <i>Angle precision</i> to (0,1] * <i>half_of_sector</i>\n
    where <i>half_of_sector = pi / number_of_sectors</i>\n
    If <i>Angle precision</i> is close to zero, learning might be very hard to
    converge (all samples classified as incorrect all the time).\n
    If <i>Angle precision</i> is bigger than half of sector, learning may
    finish sooner, but learned MLMVN may have much lower performance.\n\n
    """ + RMSE_ACC

}
