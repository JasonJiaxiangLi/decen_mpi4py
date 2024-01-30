import numpy as np

def create_mix_mat(pattern, n, random_ratio=0.8):
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
        if n % 2 != 0:
            raise ValueError("For ladder graph, the number of nodes must be even")
        res = np.eye((n, n))
        for i in range(n // 2):
            if i != 0:
                res[i, i - 1] = 1 / n
                res[i, i] -= 1 / n
            if i != n // 2 - 1:
                res[i, i + 1] = 1 / n
                res[i, i] -= 1 / n
            res[i, i + n // 2] = 1 / n
            res[i, i] -= 1 / n

            j = i + n // 2
            if j != n // 2:
                res[j, j - 1] = 1 / n
                res[j, j] -= 1 / n
            if j != n - 1:
                res[j, j + 1] = 1 / n
                res[j, j] -= 1 / n
            res[j, j - n // 2] = 1 / n
            res[j, j] -= 1 / n
    else:
        raise ValueError("This pattern is not defined")

    return res