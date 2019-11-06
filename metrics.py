import numpy as np
from math import sqrt


def accuracy_score(y_true,y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0]==y_predict.shape[0],\
    'the size of y_true must be equal to y_predict'

    return sum(y_true==y_predict)/y_true.shape[0]

def mean_squared_error(y_true,y_predict):
    '''计算y_true和y_predict之间的MSE'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'

    return np.sum((y_predict-y_true)**2/len(y_predict))

def root_mean_squared_error(y_true,y_predict):
    '''计算y_true和y_predict之间的RMSE'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'

    return sqrt(np.sum((y_predict-y_true)**2/len(y_predict)))

def mean_absolute_error(y_true,y_predict):
    '''计算y_true和y_predict之间的MAE'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'

    return np.sum(np.absolute(y_true-y_predict))/len(y_predict)

def r2_score(y_true,y_predict):
    '''计算y_true和y_predict之间的R2'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'

    return 1-mean_squared_error(y_true,y_predict)/np.var(y_true)

def r2_score_new(y_true,y_predict):
    '''计算y_true和y_predict之间的R2，使用论文中的公式，即简单相关系数的平方'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'

    return (len(y_true)*y_true.dot(y_predict)-np.sum(y_true)*np.sum(y_predict))**2/((len(y_true)*y_predict.dot(y_predict)-np.sum(y_predict)**2)*(len(y_true)*y_true.dot(y_true)-np.sum(y_true)**2))

def F_inspection(y_true,y_predict):
    '''检验Y与x之间是否存在线性关系，请使用训练样本集检验'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'
    SSR=(y_predict-np.mean(y_true)).dot(y_predict-np.mean(y_true))
    print('SSR={0}'.format(SSR))
    SSE=(y_true-y_predict).dot(y_true-y_predict)
    print('SSE={0}'.format(SSE))
    F=SSR/(SSE/(len(y_true)-2))
    print('若检验统计量F={0}大于F(α)(1,{1})，则拒绝原假设（Y与x之间无线性关系），即认为Y与x之间线性关系显著!'.format(F,len(y_true)-2))
    return F

def TN(y_true,y_predict):
    assert len(y_true)==len(y_predict)
    return np.sum((y_true==0) & (y_predict==0))

def FP(y_true,y_predict):
    assert len(y_true)==len(y_predict)
    return np.sum((y_true==0) & (y_predict==1))

def FN(y_true,y_predict):
    assert len(y_true)==len(y_predict)
    return np.sum((y_true==1) & (y_predict==0))

def TP(y_true,y_predict):
    assert len(y_true)==len(y_predict)
    return np.sum((y_true==1) & (y_predict==1))

def confusion_matrix(y_true,y_predict):
    return np.array([
        [TN(y_true,y_predict),FP(y_true,y_predict)],
        [FN(y_true,y_predict),TP(y_true,y_predict)]
    ])

def precision_score(y_true,y_predict):
    FP=np.sum((y_true==0) & (y_predict==1))
    TP=np.sum((y_true==1) & (y_predict==1))
    try:
        return TP/(FP+TP)
    except:
        return 0.0

def recall_score(y_true,y_predict):
    FN=np.sum((y_true==1) & (y_predict==0))
    TP=np.sum((y_true==1) & (y_predict==1))
    try:
        return TP/(FN+TP)
    except:
        return 0.0

def TPR(y_true,y_predict):
    FN=np.sum((y_true==1) & (y_predict==0))
    TP=np.sum((y_true==1) & (y_predict==1))
    try:
        return TP/(FN+TP)
    except:
        return 0.0

def FPR(y_true, y_predict):
    FP = np.sum((y_true == 0) & (y_predict == 1))
    TN = np.sum((y_true == 0) & (y_predict == 0))
    try:
        return FP / (FP + TN)
    except:
        return 0.0
def r2_mod(y_true,y_predict,p):
    '''修正后的R2，需要传入一个p，即变量个数'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'
    return 1-(1-r2_score(y_true,y_predict))*(len(y_predict)-1)/(len(y_true)-p-1)

def re(y_true,y_predict):
    '''计算偏差'''
    assert len(y_true)==len(y_predict),\
        'the size of y_true must equals to the size of y_predict'
    return np.abs(y_true-y_predict)/y_true

def RSD(data):
    '''返回一组一维数据的RSD'''
    assert data.ndim==1,\
    'Pls input data with one dimension'
    return np.sqrt(np.sum((data-np.mean(data))**2)/len(data-1))/np.mean(data)