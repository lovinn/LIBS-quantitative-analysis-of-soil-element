import numpy as np
from math import sqrt

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
    
def Dixon(data):
    '''输入一个一维数组，进行Dxion判定，返回异常值的索引'''
    critical_data={'3':0.941,'4':0.765,'5':0.642,'6':0.560,'7':0.507,'8':0.554,'9':0.512,'10':0.477,'11':0.576,'12':0.546,'13':0.521,'14':0.546,'15':0.525,'16':0.507,'17':0.490,'18':0.475,'19':0.462,'20':0.45,'21':0.44,'22':0.43,'23':0.421,'24':0.413,'25':0.406,'26':0.399,'27':0.393}
    index=np.argsort(data)
    out_index=[]
    while 1:
        r1=(data[index[-1]]-data[index[-3]])/(data[index[-1]]-data[index[2]])
        r2=(data[index[2]]-data[index[0]])/(data[index[-3]]-data[index[0]])
        if r1 >= r2:
            if r1 > critical_data['{0}'.format(len(data))]:
                out_index.append(index[-1])
                data = np.delete(data, index[-1])
                index = np.argsort(data)
            else:
                break
        elif r2 > critical_data['{0}'.format(len(data))]:
            out_index.append(index[0])
            data = np.delete(data, index[0])
            index = np.argsort(data)
        else:
            break
    return np.array(out_index)

def RSD(data):
    '''返回一组一维数据的RSD'''
    assert data.ndim==1,\
    'Pls input data with one dimension'
    return np.sqrt(np.sum((data-np.mean(data))**2)/len(data-1))/np.mean(data)
1+1=2
