import numpy as np

def unbox_numpy(n):
    """
    Return native type boxed in numpy type
    :param n: numpy type
    :return: unboxed value
    """
    return np.asscalar(n)

def is_numpy_type(n):
    return isinstance(n, np.generic)

def convert_if_numpy(n):
    if is_numpy_type(n):
        return unbox_numpy(n)
    else:
        return n
