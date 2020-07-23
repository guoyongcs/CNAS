def _sign(number):
    if number > 0.0:
        return 1
    elif number < 0.0:
        return -1
    else:
        return 0


def compute_kendall_tau(a, b):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    assert len(a) == len(b), "Sequence a and b should have the same length while computing kendall tau."
    length = len(a)
    count = 0
    total = 0
    for i in range(length-1):
        for j in range(i+1, length):
            count += _sign(a[i] - a[j]) * _sign(b[i] - b[j])
            total += 1
    tau = count / total
    return tau
