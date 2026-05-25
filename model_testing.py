import logging
import sys
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
from log_code import setup_logging
logger=setup_logging('model_testing')

def testing(reg,X_test,y_test):

    try:
        logger.info(f'Testing the model')
        test_prediction=reg.predict(X_test)
        logger.info(f'Testing completed')
        logger.info(f'the mean squared error is {mean_squared_error(y_test,test_prediction)}')
        logger.info(f'the root mean squread error is{np.sqrt(root_mean_squared_error(y_test,test_prediction))}')
        logger.info(f'the test accuracy of the model is{r2_score(y_test,test_prediction)}')
    except Exception as e:
        er_ty,er_msg,er_line=sys.exc_info()
        logger.info(f'error in line no : {er_line.tb_lineno} due to : {er_msg}')

