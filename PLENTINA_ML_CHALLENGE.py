from flask import Flask
from flask_restful import Api,Resource,reqparse,abort,fields,marshal_with
from flask_sqlalchemy import SQLAlchemy



app=Flask(__name__)
api=Api(app)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///PLENTINADB.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://eoaanmphxbnuvp:f74035ddab268012f0e500dac515b0f749947c4261384285179ea3ea1abcc2ea@ec2-3-212-143-188.compute-1.amazonaws.com:5432/d4apbo7o7ts7rg'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
#db.create_all()

#class transactions(db.Model):
#    step = db.Column(db.Integer,primary_key=True,nullable=False)
#    type = db.Column(db.String, nullable=False)
#    amount = db.Column(db.Float, nullable=False)
#    nameOrig = db.Column(db.String, nullable=False)
#    oldbalanceOrig = db.Column(db.Float, nullable=False)
#    newbalanceOrig = db.Column(db.Float, nullable=False)
#    nameDest = db.Column(db.String, nullable=False)
#    oldbalanceDest = db.Column(db.Float, nullable=False)
#    newbalanceDest = db.Column(db.Float, nullable=False)
#    isFraud = db.Column(db.Integer, nullable=False)
#    def __repr__(self):
#        return f"transactions(step = {step}, type = {type}, amount = {amount},nameOrig = {nameOrig},oldbalanceOrig = {oldbalanceOrig},newbalanceOrig = {newbalanceOrig},nameDest = {nameDest},oldbalanceDest = {oldbalanceDest},newbalanceDest = {newbalanceDest},isFraud = {isFraud})"

db.create_all()




import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from pickle import load

df=pd.read_csv("reduced_transactions_train.csv")

df.to_sql('transactions',con=db.engine, if_exists='append',index=False)

#Fraud Detection Model: load fraud detection Model 
model=load(open("gbdtmodel.pkl",'rb'))



#transaction_table=db.Table('transactions',db.metadata,autoload=True,autoload_with=db.engine)


transaction_data=reqparse.RequestParser()
transaction_data.add_argument("step",type=int,help='please put step',required=True)
transaction_data.add_argument("type",type=str,help='please put transaction type',required=True)
transaction_data.add_argument("amount",type=float,help='please put transaction amount',required=True)
transaction_data.add_argument("nameOrig",type=str,help='please put transaction origin',required=True)
transaction_data.add_argument("oldbalanceOrig",type=float,help='please put old original balance',required=True)
transaction_data.add_argument("newbalanceOrig",type=float,help='please put new original balance',required=True)

transaction_data.add_argument("nameDest",type=str,help='please put transaction destination',required=True)
transaction_data.add_argument("oldbalanceDest",type=float,help='please put old destination balance',required=True)
transaction_data.add_argument("newbalanceDest",type=float,help='please put new destination balance',required=True)




#Feature Engineering Function
def transform_data(df):
    new_df=pd.DataFrame()
    
    #if the origin account has no initial balance
    y=np.where(df['oldbalanceOrig']>0,df['oldbalanceOrig'],1)
    x=df['newbalanceOrig']
    #percentage inflow/outlflow of account
    new_df['percentage_diff_balanceOrig']=np.round(((x/y)-1)*100,3)
    #amount in the account before transaction: Is there an initial balance in the account
    new_df['NoAmountBalanceOrig']=np.where(df['oldbalanceOrig']>0,1,0)

    #if the destination account has no initial balance
    y=np.where(df['oldbalanceDest']>0,df['oldbalanceDest'],1)
    x=df['newbalanceDest']
    #percentage inflow/outlflow of the account
    new_df['percentage_diff_balanceDest']=np.round(((x/y)-1)*100,3)
    #amount in the account before transaction: Is there an initial balance in the account
    new_df['NoAmountBalanceDest']=np.where(df['oldbalanceDest']>0,1,0)

    #Transaction Inflow or Outlfow to each account 
    #1:Inflow
    #0: No Change
    #-1: Outflow
    new_df['signbalanceOrig']=np.sign(df['newbalanceOrig']-df['oldbalanceOrig'])
    new_df['signbalanceDest']=np.sign(df['newbalanceDest']-df['oldbalanceDest'])

    x=np.where(df['amount']>0,df['amount'],1)
    new_df['AmountBalanceDiffOrig']=np.absolute((df['newbalanceOrig']-df['oldbalanceOrig']))/x
    new_df['AmountBalanceDiffDest']=np.absolute((df['newbalanceDest']-df['oldbalanceDest']))/x

    new_df['AmountBalanceDiffOrig']=new_df['AmountBalanceDiffOrig'].round(3)
    new_df['AmountBalanceDiffDest']=new_df['AmountBalanceDiffDest'].round(3)

    new_df['isAmount']=np.where(df['amount']>0,1,0)
    
    
    xi=df[['step','type']]
    xii=new_df[['percentage_diff_balanceOrig', 'NoAmountBalanceOrig','percentage_diff_balanceDest', 'NoAmountBalanceDest', 'signbalanceOrig','signbalanceDest', 'AmountBalanceDiffOrig', 'AmountBalanceDiffDest','isAmount']]
    
    return pd.concat([xi,xii],axis=1)


@app.route('/is-fraud', methods = ['POST'])

def fraud_detection():
    #parse args to python dictionary (unordered)
    params=transaction_data.parse_args()
    #Append to existing database
    
    params_df=pd.DataFrame({'step':params['step'],
                            'type':params['type'],
                            'amount':params['amount'],
                            'nameOrig':params['nameOrig'],
                            'oldbalanceOrig':params['oldbalanceOrig'],
                            'newbalanceOrig':params['newbalanceOrig'],
                            'nameDest':params['nameDest'],
                            'oldbalanceDest':params['oldbalanceDest'],
                            'newbalanceDest':params['newbalanceDest'],
                            'isFraud':-1},index=[0])
    params_df.to_sql('transactions',con=db.engine, if_exists='append',index=False)
    
   

    #Query database with appended transaction
    
    query=f"""SELECT*
          FROM transactions 
          WHERE "transactions.step" = {params['step']} AND
          "transactions.amount" = {params['amount']} AND
          "transactions.type" = '{params['type']}' AND 
          "transactions.nameOrig" = '{params['nameOrig']}' AND 
          "transactions.nameDest" = '{params['nameDest']}' 
          """
    result=pd.read_sql_query(query,con=db.engine)
    
    y_pred=transform_data(result)
    select_choice=str(np.where(model.predict(y_pred.values)==1,"true","false")[0])
    return {'isFraud':select_choice}

@app.route('/is-corn', methods = ['GET'])

def corn_detection():
    return {'corny':'butter corn'}



if __name__=='__main__':
    app.run(debug=True,use_reloader=False)
    #app.run(debug=True)
