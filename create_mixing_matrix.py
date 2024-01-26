import numpy as np

def create_mix_mat(pattern, n, random_ratio=0.9):
    if pattern == 'complete':
        res = np.ones((n, n)) / n
    elif pattern == 'ring':
        res = np.zeros((n, n))
        for i in range(n):
            res[i, i] = 1 / 3
            res[i, (i - 1) % n], res[i, (i + 1) % n] = 1 / 3, 1/ 3
    elif pattern == 'random':
        print("Note: random graph may not be connected")
        res = np.ones((n, n)) / n
        for i in range(n):
            temp = np.random.binomial(1, random_ratio, n)
            for idx, v in enumerate(temp):
                if idx != i and v == 0:
                    res[i, idx] = 0.0
                    res[i, i] += 1 / n
    elif pattern == 'ladder':
        # TODO: implement ladder
        res = np.ones((n, n)) / n
    else:
        raise ValueError("This pattern is not defined")

    return res