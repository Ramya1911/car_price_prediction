import logging

import numpy as np
import sklearn
from sklearn .linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from log_code import setup_logging
logger=setup_logging("model_training")
import sys
def training(X_train,y_train):
    try:
        global reg
        logger.info(f'training model')
        reg=LinearRegression()
        reg.fit(X_train,y_train)
        predictions=reg.predict(X_train)
        logger.info(f'training complete')
        logger.info(f'The mean square error is {mean_squared_error(y_train,predictions)}')
        logger.info(f'the root mean squared error is{np.sqrt(mean_squared_error(y_train,predictions))}')
        logger.info(f'The r2 score is {r2_score(y_train,predictions)}')
        return reg
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f'error in line no:{er_line.tb_lineno} due to : {er_msg}')
