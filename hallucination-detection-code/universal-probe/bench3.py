import numpy as np,os
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import bench2 as B  # reuse loaders? fallback inline
