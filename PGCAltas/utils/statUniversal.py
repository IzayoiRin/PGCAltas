import numpy as np


class UniError(Exception):
    pass


exp = np.exp
log = np.log
sum_ = np.sum


def sequencing_switch(switcher: str, *sargs, **skwargs):

    def fpkm2tpm(fpkm):
        return exp(log(fpkm) - log(sum_(fpkm)) + log(1e6))

    def counts2tpm(counts, efflen):
        rate = log(counts) - log(efflen)
        denom = log(sum_(exp(rate)))
        return exp(rate - denom + log(1e6))

    def count2fpkm(counts, efflen):
        n = sum_(counts)
        return exp(log(counts) + log(1e9) - log(efflen) - log(n))

    def count2effcount(counts, length, efflen):
        return counts * (length / efflen)

    switch = {
        "f2t": fpkm2tpm,
        "c2t": counts2tpm,
        "c2f": count2fpkm,
        "c2ef": count2effcount
    }

    return switch[switcher](*sargs, **skwargs)


def eq_decorator(func):

    def inner(*arrs, dim=None, **kwargs):
        try:
            ret = func(*arrs, dim)
        except Exception as e:
            if kwargs.get('raise_exc', True):
                raise UniError(e)
            else:
                return None

        if kwargs.get('int0', None):
            return ret.astype(np.int0)

        return ret

    return inner


@eq_decorator
def eq(arr1, arr2, dim=None):
    if dim is None:
        return np.equal(arr1, arr2)
    try:
        flag = arr1.shape[dim] == arr2.shape[dim]
    except IndexError:
        raise IndexError('Dimension out of range')
    if not flag:
        return np.equal(arr1, arr2)
    n = arr1.shape[dim]
    if dim is 0:
        return np.array([np.equal(arr1[i, :], arr2[i, :]).all() for i in range(n)])
    return np.array([np.equal(arr1[:, i], arr2[:, i]).all() for i in range(n)])


def split_arr(arr, seq):
    ret = list()
    n = len(arr)
    if n <= seq:
        ret.append(arr)
        return ret
    for i in range(n // seq):
        s, e = i*seq, (i+1)*seq
        ret.append(arr[s:e])
    if e < n:
        ret.append(arr[e:])
    return ret
