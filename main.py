import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])
