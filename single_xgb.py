import xgboost as xgb
import pandas as pd
import numpy as np


def my_xgb_cv(x_train,y_train, NFOLDS, xgb_params):
	dtrain = xgb.DMatrix(data=x_train, label=y_train)
	bst = xgb.cv(xgb_params, dtrain, 10000, NFOLDS,early_stopping_rounds=60,verbose_eval= 50)
	return bst
