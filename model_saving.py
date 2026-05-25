import logging
import sys
from log_code import setup_logging
logger=setup_logging('model_saving')
import pickle

def saving( reg,X_train):
    try:
        with open('mini_project_2.pkl', 'wb') as f:
            pickle.dump(reg, f)
        logger.info(f'model saved successfully')
        with open('mini_project_2.pkl','rb') as f:
            m= pickle.load(f)

        logger.info(f'the sample prediction is {m.predict(X_train.head(1))}')




    except Exception as e:
        er_ty,er_msg,er_line=sys.exc_info()
        logger.info(f'error at line no is {er_line.tb_lineno} due to {er_msg}')


