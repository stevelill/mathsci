"""
Text-formatting routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import pandas as pd
import numpy as np


def _preferred_format(precision, sci=False):
    """
    Default number format for segreg.

    Parameters
    ----------
    precision: int

    Returns
    -------
    number_format: str
    """
    #format_str = '{0:.' + str(precision) + 'g}'

    if sci:
        pref_format = '{0:.' + str(precision) + 'e}'
    else:
        pref_format = '{0:.' + str(precision) + 'f}'

    # TODO: use this together with format(x, '.3f') below
    # return '.' + str(precision) + 'f}'

    return pref_format


def format_num(val, num_decimal=4, sci=False):
    """
    Formats input number to the default number format for segreg.

    Parameters
    ----------
    val: numeric type
    num_decimal: int

    Returns
    -------
    formatted_input: str
    """
    format_str = _preferred_format(num_decimal, sci=sci)
    return format_str.format(val)


def get_decimal_formatter(precision=3, sci=False):
    """
    Default number format for segreg.

    Parameters
    ----------
    num_decimal: int

    Returns
    -------
    number_format: str
    """
    def formatter(x):
        if sci:
            format_str = '.' + str(precision) + 'e'
        else:
            format_str = '.' + str(precision) + 'f'

        return format(x, format_str)
        # return format(x, '.3g')
        # return "{0:.3}".format(x)
    return formatter


def get_text_line(num=50):
    """
    Creates a str line useful for adding to printed messages.

    Parameters
    ----------
    num: int
        the length of the line (number of dashes)

    Returns
    -------
    line: str
    """
    elements = ["-" for dummy in range(num)]
    return "".join(elements)


def underline(text):
    """
    Adds an underline to given text.

    Parameters
    ----------
    text: str

    Returns
    -------
    underlined_text: str
    """
    line = "".join(["-" for dummy in text])
    return text + "\n" + line


def format_output(precision=4, sci=False):
    """
    Sets global options for scipy and pandas to control number and other
    formatting.

    Parameters
    ----------
    precision: int
        precision for decimal formatting
    """
    # TODO: was 120
    linewidth = 220
    np.set_printoptions(precision=precision,
                        linewidth=linewidth,
                        suppress=True)
    pd.set_option('precision', precision)

    pd.options.display.float_format = _preferred_format(precision=precision,
                                                        sci=sci).format
    pd.options.display.width = linewidth
    pd.options.display.max_columns = None

    # TODO: put a number here so works with all versions
    #pandas.options.display.max_colwidth = -1

    # new versions
    pd.options.display.max_colwidth = None
