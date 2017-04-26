import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

import base_data as bd
import basic_parameters as bp
import single_xgb as sx
import stack_model as sm

train, test, y_train, ntrain, ntest, listing_id, listing_id_train = bd.load_base_data()



xgb_params = bp.get_basic(3)

print(x_test.shape)

bst = sx.my_xgb_cv(x_train,y_train, NFOLDS = 6,xgb_params = xgb_params)
best_rounds = np.argmin(bst['test-mlogloss-mean'])
print("----------------------------------------------------")
print(bst.iloc[best_rounds])
files_name = bst.iloc[best_rounds]["test-mlogloss-mean"]
print("----------------------------------------------------")

dtrain = xgb.DMatrix(data=x_train, label=y_train)
bst = xgb.train(xgb_params, dtrain, best_rounds)

dtest = xgb.DMatrix(data=x_test)
preds = bst.predict(dtest)
preds = pd.DataFrame(preds, columns = ['high', 'medium', 'low'])                                                               
preds['listing_id'] = listing_id

preds.to_csv('upload/only_google_noFound_cv_' + str(best_rounds) + '_' + str(files_name) + '_' + 'my_preds.csv', index=None)
