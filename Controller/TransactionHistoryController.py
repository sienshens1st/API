from flask import *
import pandas as pd
import numpy as np



def TransactionHistory(namafile, user):

    nama_file = 'Files\\' + namafile + '.csv'
    df = pd.read_csv(nama_file,sep=',')
    df = df.dropna()
    df = df.loc[df['SALES_QTY'] > 0]

    # remove unnused column 
    df = df.drop(columns=['TRX_NUMBER', 'BRANCH_CODE','RAYON_EXP_CODE','RAYON_EXP_DESC','SALES_CHANNEL_CODE','SALES_CHANNEL_DESC','PRODUCT_FAMILY_CODE','PRODUCT_FAMILY_DESC'] ,axis=1)
    df = df.loc[df['SHIP_TO_ID'] == user].reset_index(drop=True)
    df['TRX_DATE'] = df['TRX_DATE'].str.split(' ').str[0]
    df['dates'] = pd.to_datetime(df['TRX_DATE'])
    df = df.sort_values('dates',ascending=False)
    # remove unnused column 
    df = df.drop(columns=['dates','SHIP_TO_ID'] ,axis=1)

    result =df.to_json(orient="records")
    parsed = json.loads(result) 

    return jsonify(parsed),200
