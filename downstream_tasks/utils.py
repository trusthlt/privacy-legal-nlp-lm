import logging

import numpy as np
from sklearn import metrics


def tune_threshs(probas, truth):
    res = np.zeros(probas.shape[1])

    assert np.alltrue(probas > 0.0), logging.info(f"min: {np.min(probas)}, index: {np.argmin(probas)}")
    assert np.alltrue(probas < 1.0), logging.info(f"max: {np.max(probas)}, index: {np.argmax(probas)}")

    for i in range(probas.shape[1]):
        if np.sum(truth[:, i]) > 4 :
            thresh = max(
                np.linspace(
                    0.0,
                    1.0,
                    num=100,
                ),
                key=lambda t: metrics.f1_score(y_true=truth[:, i], y_pred=(probas[:, i] > t), pos_label=1, average='binary')
            )
            res[i] = thresh
        else:
            # res[i] = np.max(probas[:, i])
            res[i] = 0.5

    return res


def apply_threshs(probas, threshs):
    res = np.zeros(probas.shape)

    for i in range(probas.shape[1]):
        res[:, i] = probas[:, i] > threshs[i]

    return res



