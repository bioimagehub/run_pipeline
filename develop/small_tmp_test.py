import numpy as np

eigenvectors = np.array([[ 0.          , 0.          , 1.        ],
                          [-0.95079749, -0.30981308, 0.        ],
                          [ 0.30981308, -0.95079749, 0.        ]])


result = ''.join(['(' + ','.join(map(str, ev)) + ')' for ev in eigenvectors])

print(result)