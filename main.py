import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import logging
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
from log_code import setup_logging
logger = setup_logging("main")
from sklearn.model_selection import train_test_split
from model_training import training
from model_testing import testing
from model_saving import saving
class CAR_PRICE_PREDICTION:
    def __init__(self,data):
        try:
            self.data= data
            self.df = pd.read_csv(self.data)
            logger.info(f'the data set shape is {self.df.shape}')
            self.df=self.df.drop(['Car ID'],axis=1)
            logger.info(f'the data set shape after removing the id column is {self.df.shape}')
            logger.info(f'the columns: {self.df.columns}')
            logger.info(f'the data types of columns before{self.df.dtypes}')
            logger.info(f'the nuemerical data types of columns before:\n {self.df.select_dtypes(exclude='str')}')
            logger.info(f'the object data type columns :\n {self.df.select_dtypes(include='str')}')
            self.obj_cols=self.df.select_dtypes(include='str').columns
            for i in self.obj_cols:
                a={}
                s=0
                for j in self.df[i]:
                    if j not in a:
                        a[j]=s
                        s+=1
                    else:
                        pass
                logger.info(f'{a}')
                self.df[i] = self.df[i].map(a).astype(int)
            logger.info(f'the columns data types after converting to nuemeric \n {self.df.dtypes} ')
            self.X=self.df.drop(['Price'],axis=1)
            self.y=self.df['Price']
            logger.info(f'the independent data is\n {self.X}')
            logger.info(f'the dependent data is \n {self.y}')
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'the training data set size is \n {self.X_train.shape},{self.y_train.shape}')
            logger.info(f'the testing data set shape is \n {self.X_test.shape},{self.y_test.shape}')
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

    def training_the_model(self):
        try:
            reg=training(self.X_train,self.y_train)
            self.reg=reg
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            logger.info(f'error in line no:{er_line.tb_lineno} due to : {er_msg}')


    def testing_the_model(self):
        try:
            testing(self.reg,self.X_test,self.y_test)
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            logger.info(f'error in line no : {er_line.tb_lineno} due to : {er_msg}')

    def saving_the_model(self):
        try:
            saving(self.reg,self.X_train)
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            logger.info(f'error in line no{er_line.tb_lineno} due to :{er_msg}')












if __name__ == "__main__":
    obj=CAR_PRICE_PREDICTION("car_price_prediction_.csv")
    obj.training_the_model()
    obj.testing_the_model()
    obj.saving_the_model()
