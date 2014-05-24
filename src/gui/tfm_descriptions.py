#!/usr/bin/env python
#encoding=utf8

"""
This module stores description labels
for transformations and improves readability
and usability in gui_transformations module

"""

TFM_DESC = {
    "DiscreteBisectorTfm":

    r"""transforms categorical or
    numeric attributes with integer value to bisectors of sectors.
    \n <b>Parameters</b> \n
    <i>Number of values</i> specifies number of sectors used to
    encode data and has to be equal or greater than number of
    values of categorical
    attribute (equality is preferred).""",

    "MinMaxNormalizeTfm":

    r"""transforms numeric attribute
    values to complex numbers
    (unit circle vectors). At first value <i>x</i> of numeric
    attribute <i>A</i> is normalized by transformation: \n
    <i>x_transformed = (x - min(A)) / (max(A) - min(A))</i>\n
    consequently each value <i>x_transformed</i> gets assigned
    unit circle vector with angle from interval <i>[0, 2*pi - "mapping_gap"]</i>.
    \n So that:
    \n <i>min(A)</i> gets mapped to <i>0 + 0i</i> and
    \n <i>max(A)</i> gets mapped to <i>cos(angle) +
    sin(angle)*i</i> where <i>angle = 2*pi - "mapping gap"</i>
    \n <b>Parameters:</b> \n
    <i>Mapping gap</i> parameter
    specifies size of mapping gap in radians. It should be floating number
    from interval <i>[0,2pi)</i>. But typically is smaller than
    <i>pi</i>.
    \n <b>Notes:</b>\n
    Too small mapping gap limits transformation's performance for future
    usage of MLMVN (on unseen dataset which was not used to compute
    <i>min(A)</i> and <i>max(A)</i> on - very big or very small values
    get mixed
    with the original ones in this case). Big gap on the other side
    results in mapping all the values of attribute close to each other,
    making it thus more difficult to distinguish necessary differences;
    which may negatively affect learning process.
    """
}
