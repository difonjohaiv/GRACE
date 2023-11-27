import numpy as np
import pandas as pd


def get_labels():
    fname = "thucnews_labels.csv"
    target = np.array(pd.read_csv(fname)['label'])
    target2id = {label: indx for indx, label in enumerate(set(target))}
    result = [target2id[label] for label in target]
    return result
