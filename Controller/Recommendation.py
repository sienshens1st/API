from flask import *
import pandas as pd
import numpy as np


def CalculateRecommendation(namafile, user):

    nama_file = 'Files\\' + namafile + '.csv'
    print(nama_file)
    df = pd.read_csv(nama_file,sep=',')
    df = df.dropna()
    df = df.loc[df['SALES_QTY'] > 0]

    df_orders = df.assign(
    totalfarmaitem = np.where(df['PRODUCT_FAMILY_CODE']== 1,df['ITEM_CODE'],0),
    totalgoodsitem = np.where(df['PRODUCT_FAMILY_CODE']==2 ,df['ITEM_CODE'],0)
    ).groupby('SHIP_TO_ID').agg(
        totalfarmaitem = ('totalfarmaitem','nunique'),
        totalgoodsitem = ('totalgoodsitem','nunique'),
        uniqueitembought = ('ITEM_CODE','nunique'),
    )

    df_orders = df_orders.loc[(df_orders['uniqueitembought'] >= 1)]
    
    features = ["totalfarmaitem", "totalgoodsitem"]
    customer = df_orders.dropna(subset=features)
    data = customer[features].copy()

    centroid_default = [(25,30),(50,200),(250,30),(400,350)]
    centroids = pd.DataFrame(centroid_default,columns=['totalfarmaitem','totalgoodsitem']).T

    labels = get_labels(data, centroids)
    labels.value_counts()

    max_iterations = 100
    centroid_count = 4

    # centroids = random_centroids(data, centroid_count) gapake ini karena uda pake centroid default
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, centroid_count)
        # plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    picked_customer = user

    if picked_customer in customer[labels == 0].index:
        df_cluster = customer[labels == 0]
        print("customer is in cluster 0")
    elif picked_customer in customer[labels == 1].index:
        df_cluster = customer[labels == 1]
        iscluster = 1
        print("customer is in cluster 1")
    elif picked_customer in customer[labels == 2].index:
        df_cluster = customer[labels == 2]
        iscluster = 2 
        print("customer is in cluster 2")
    elif picked_customer in customer[labels == 3].index:
        df_cluster = customer[labels == 3]
        iscluster = 3
        print("customer is in cluster 3")
    else:
        print("customer is not found!!!!")
        return jsonify("Customer transaction is not found."),400

    df_onlycluster = pd.merge(df_cluster, df, on=['SHIP_TO_ID'], how='left', indicator='Exist')
    df_onlycluster.drop('totalfarmaitem', inplace=True, axis=1)
    df_onlycluster.drop('totalgoodsitem', inplace=True, axis=1)
    df_onlycluster.drop('uniqueitembought', inplace=True, axis=1)
    df_onlycluster['Exist'] = np.where(df_onlycluster.Exist == 'both', True, False)
    df = df_onlycluster

    df= df.dropna()
    df = df.loc[df['SALES_QTY'] > 0]
    df = df.drop(columns=['TRX_DATE', 'TRX_NUMBER', 'BRANCH_CODE','GROSS_SALES_AMOUNT','RAYON_EXP_CODE','RAYON_EXP_DESC','SALES_CHANNEL_CODE','SALES_CHANNEL_DESC','PRODUCT_FAMILY_CODE','PRODUCT_FAMILY_DESC','Exist'] ,axis=1)

    # aggregate by item code
    agg_ratings = df.groupby('ITEM_CODE').agg(mean_rating = ('SALES_QTY', 'mean'),
                                                    number_of_ratings = ('SALES_QTY', 'count')).reset_index()

    # Keep the item with over 20 transaction 
    agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>1]
    agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=True).head()
    df_GT100 = pd.merge(df, agg_ratings_GT100[['ITEM_CODE']], on='ITEM_CODE', how='inner')

    # sum quantity if same itemcode + ship
    df_GT100['SALES_QTY_SUM'] = df_GT100.groupby(['SHIP_TO_ID', 'ITEM_CODE'])['SALES_QTY'].transform('sum')
    new_df = df_GT100.drop_duplicates(subset=['SHIP_TO_ID', 'ITEM_CODE'])
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop(columns=['SALES_QTY'],axis=1);
    df_GT100 = new_df

    matrix = df_GT100.pivot_table(index='ITEM_CODE', columns='SHIP_TO_ID', values='ITEM_CODE')

    matrix_norm = matrix.apply(standarize,axis=1)
    matrix_norm

    # Pearson corelation
    matrix_norm_filledna = matrix_norm.fillna(0)
    np.seterr(divide='ignore')
    with np.errstate(divide='ignore',invalid='ignore'):
        correlation = np.corrcoef(np.array(matrix_norm_filledna))
    correlation = pd.DataFrame(correlation, columns=matrix_norm_filledna.T.columns)
    correlation['ITEM_CODE'] = matrix_norm_filledna.T.columns
    correlation = correlation.set_index('ITEM_CODE')
    correlation

    testable = matrix_norm

    picked_user = picked_customer
    predictscore= pd.DataFrame(columns=testable.columns,index=testable.index)[picked_user]

    index = 0
    for i in testable[picked_user]:
        total = 0
        testable_indexName = testable[picked_user].index[index]
        userValue = testable[picked_user][index] 
        sumpembilangnp = ((testable[picked_user]*correlation[testable_indexName])).sum()  - np.nansum(userValue)
        sumpenyebutnp = np.nansum(abs(correlation[testable_indexName])) -1
        total = (sumpembilangnp/sumpenyebutnp)
        predictscore[index] = total
        index +=1

    dfmerge = pd.DataFrame()
    dfmerge[picked_user] = testable[picked_user]

    # take item which user has not bought
    dfmergetest = dfmerge.merge(predictscore,left_on='ITEM_CODE',right_on='ITEM_CODE')
    dfmergetest = dfmergetest[dfmergetest[dfmergetest.columns[0]].isnull()]
    # drop original column 
    dfmergetest = dfmergetest.drop(columns=[dfmergetest.columns[0]])
    # sort by highest value 
    dfmergetest = dfmergetest.sort_values(by=[dfmergetest.columns[0]], ascending=False)
    #remove negative weight and take top 10 item 
    dfmergetest = dfmergetest.where(dfmergetest > 0 ).dropna().head(20)
    result = dfmergetest.index.tolist()    

    return jsonify(result),200

def random_centroids(data, k):
    centroids = []
    for i in range(k):       
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

def new_centroids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids


def standarize(row):
    new_row = (row - row.min())/(row.max() - row.min())
    return new_row

def weighted_sum_numerator(user, corr,userValue):
    numerator = sum([np.nansum(user[i])*np.nansum(corr[i]) for i in range(len(user))])
    return numerator - userValue