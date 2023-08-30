from asyncore import dispatcher_with_send
import imp
import sys

import pandas as pa
import numpy as np
from svm import svm_HD
from GBDT import GBDT_HD
from CART import CART_HD
from NB import NB_HD
from threeNN import threeNN_HD
from fiveNN import fiveNN_HD
from RF import RF_HD
from MLP import MLP_HD
from AdaB import AdaB_HD
from sklearn.model_selection import StratifiedKFold
def read_file(filename):
    df = pa.read_csv(filename)
    return df
def mashi_dis(data,mu):
    ans = []
    index_train = data.shape[0]
    for i in range(data.shape[0]):
        x = data[i,:-1].T
        ans_t = np.sqrt(np.dot(np.dot((x - mu).T, np.cov(data[0,:-1])), (x - mu)))
        ans.append(ans_t)
    return np.array(ans)
def ou_dis(data,mu):
    S = np.var(np.vstack([data[:, :-1], mu]), axis=0, ddof=1)
    X = np.square(data[:, :-1] - mu)
    dis = np.sqrt(np.sum(X/S,axis = 0))
    return dis


def get_mean_duoshu(df,dis_id):
    data = df[df.iloc[:,-1]==0].values
    mean_data = np.mean(data[:,:-1],axis = 0)
    if dis_id == 0:
        dis_arr = mashi_dis(data,mean_data)
    else:
        dis_arr = ou_dis(data,mean_data)
    df_sort = pa.Series(dis_arr).sort_values()
    return df_sort
def get_mean_shaoshu(data):
    return np.mean(data[data.iloc[:,-1]==1].values[:,:-1])
def HDJC(data,id=0,stride = 1):
    get_data = []
    num = 0
    df_lot = get_mean_duoshu(data,id)
    df_lot_shape = len(df_lot)
    df_little = get_mean_shaoshu(data)
    df_little_shape = len(data[data.iloc[:,-1].values==1])
    df_lot_little_dis = np.sum(np.square(data.iloc[df_lot.index,:-1].values-df_little),axis = 1)
    df_lot_index = np.argmin(df_lot_little_dis)
    df_lot_index_beifen = df_lot_index
    if df_little_shape%2==0:
        get_data_shape = df_little_shape/2

        if (df_lot_index + 1 > df_lot_shape / 2):
            if (df_lot_shape - df_lot_index - 1 < get_data_shape):
                while (df_lot_index_beifen - get_data_shape + 1 >= 0):
                    get_data.append([])
                    if (df_lot_shape - df_lot_index_beifen - 1 <= get_data_shape):
                        a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                        get_data[num].append(data.iloc[df_lot.index[a_index:], :].values)
                        get_data[num].append(data[data.iloc[:, -1] == 1].values)
                        num += 1
                        df_lot_index_beifen -= 1
                    else: 
                        if (df_lot_index_beifen - get_data_shape + 1 >= 0):
                            a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                            b_index = (df_lot_index_beifen + get_data_shape).astype(int)
                            get_data[num].append(data.iloc[df_lot.index[a_index:b_index], :].values)
                            get_data[num].append(data[data.iloc[:, -1] == 1].values)
                            num += 1
                            df_lot_index_beifen -= 1
            else:
                cal_dis = df_lot_shape - df_lot_index_beifen - get_data_shape
                while (cal_dis >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape).astype(int)
                    get_data[num].append(data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen += 1
                    cal_dis -= 1
                df_lot_index_beifen = df_lot_index
                while (df_lot_index_beifen - get_data_shape + 1 >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(
                        data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen -= 1
        else: 
            if (df_lot_index + 1 < get_data_shape):
                while (df_lot_index_beifen + 1 + get_data_shape <= df_lot_shape):
                    get_data.append([])
                    if (df_lot_index_beifen + 1 <= get_data_shape):
                        a_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                        get_data[num].append(data.iloc[df_lot.index[:a_index], :].values)
                        get_data[num].append(data[data.iloc[:, -1] == 1].values)
                        num += 1
                        df_lot_index_beifen += 1
                    else: 
                        if (df_lot_index_beifen + 1 <= df_lot_shape):
                            a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                            b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                            get_data[num].append(data.iloc[df_lot.index[a_index:b_index], :].values)
                            get_data[num].append(data[data.iloc[:, -1].values == 1].values)
                            num += 1
                            df_lot_index_beifen += 1
            else:
                cal_dis = df_lot_index_beifen + 1 - get_data_shape
                while (cal_dis >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].iloc[:,:].values)
                    num += 1
                    df_lot_index_beifen -= 1
                    cal_dis -= 1
                df_lot_index_beifen = df_lot_index
                while (df_lot_index_beifen + get_data_shape + 1 <= df_lot_shape):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(
                        data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen += 1
    else:
        get_data_shape = (df_little_shape-1)/2
        if (df_lot_index + 1 > df_lot_shape / 2):
            if (df_lot_shape - df_lot_index - 1 < get_data_shape):
                while (df_lot_index_beifen - get_data_shape + 1 >= 0):
                    get_data.append([])
                    if (df_lot_shape - df_lot_index_beifen - 1 <= get_data_shape):
                        a_index = df_lot.index[(df_lot_index_beifen - get_data_shape).astype(int):]
                        get_data[num].append(data.iloc[a_index, :].values)
                        get_data[num].append(data[data.iloc[:, -1] == 1].iloc[:,:].values)
                        num += 1
                        df_lot_index_beifen -= 1
                    else: 
                        if (df_lot_index_beifen - get_data_shape + 1 >= 0):
                            a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                            b_index  = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                            get_data[num].append(data.iloc[df_lot.index[a_index:b_index], :].values)
                            get_data[num].append(data[data.iloc[:, -1] == 1].values)
                            num += 1
                            df_lot_index_beifen -= 1
            else:
                cal_dis = df_lot_shape - df_lot_index_beifen - 1 - get_data_shape
                while (cal_dis >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen += 1
                    cal_dis -= 1
                df_lot_index_beifen = df_lot_index
                while (df_lot_index_beifen - get_data_shape + 1 >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(
                        data.iloc[
                        df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen -= 1
        else: 
            if (df_lot_index + 1 < get_data_shape):
                while (df_lot_index_beifen + 1 + get_data_shape <= df_lot_shape):
                    get_data.append([])
                    if (df_lot_index_beifen + 1 <= get_data_shape):
                        a_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                        get_data[num].append(data.iloc[df_lot.index[:a_index], :].values)
                        get_data[num].append(data[data.iloc[:, -1] == 1].values)
                        num += 1
                        df_lot_index_beifen += 1
                    else: 
                        if (df_lot_index_beifen + 1 <= df_lot_shape):
                            a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                            b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                            get_data[num].append(data.iloc[df_lot.index[a_index:b_index], :].values)
                            get_data[num].append(data[data.iloc[:, -1] == 1].values)
                            num += 1
                            df_lot_index_beifen += 1
            else:
                cal_dis = df_lot_index_beifen + 1 - get_data_shape
                while (cal_dis >= 0):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen -= 1
                    cal_dis -= 1
                df_lot_index_beifen = df_lot_index
                while (df_lot_index_beifen + get_data_shape + 1 <= df_lot_shape):
                    get_data.append([])
                    a_index = (df_lot_index_beifen - get_data_shape).astype(int)
                    b_index = (df_lot_index_beifen + get_data_shape + 1).astype(int)
                    get_data[num].append(
                        data.iloc[df_lot.index[a_index:b_index],:].values)
                    get_data[num].append(data[data.iloc[:, -1] == 1].values)
                    num += 1
                    df_lot_index_beifen += 1
    return get_data
def cal_train_test(train,test,num = 5):
    mean_train = []
    for i in range(len(train)):
        d = np.vstack([train[i][0],train[i][1]])
        train[i] = d
        mean_train.append(np.mean(d[:,:-1],axis = 0))
    mean_test = np.mean(test.iloc[:,:-1].values,axis = 0)
    mean_dis = []
    for i in mean_train:
        mean_dis.append(np.sum(np.square(mean_test - i)))
    mean_dis = pa.Series(mean_dis).sort_values()
    end_index = (mean_dis.index[:num]).astype(int).values
    return np.array(train)[end_index]
if __name__ == '__main__':
    file_name = 'data/ecoli1.csv'
    df = read_file(file_name)
    num = 30
    #dis_id = 1
    end_data = []
    train = []
    test = []
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    X,Y = df.iloc[:,:-1].values,df.iloc[:,-1].values
    iter = 0
    for train_index, test_index in skf.split(X, Y):
        train.append(df.iloc[train_index,:])
        test.append(df.iloc[test_index, : ])
        get_data = HDJC(train[iter],0)
        end_data.append(cal_train_test(get_data,test[iter],num))
        iter += 1
    sys.stdout = open(file_name[5:-4]+".txt", mode='w', encoding='utf-8')
    print('*************************************************************SVM*************************************************************')
    svm_HD(end_data=end_data,test = test,num=num,train = train)
    print('*************************************************************CART*************************************************************')
    CART_HD(end_data=end_data,test = test,num=num,train = train)
    print('*************************************************************RF*************************************************************')
    RF_HD(end_data=end_data,test = test,num=num,train = train)


