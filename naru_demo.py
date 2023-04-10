import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size,mid_size, output_size):
        super(Autoencoder, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, mid_size),
            nn.ReLU())  # 编码器结构

        self.decoder = nn.Sequential(
            nn.Linear(mid_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid())  # 解码器结构

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x

import numpy as np
import pandas as pd
from datetime import datetime
import csv

csvpath='stu_table_tiny.csv'
pretreatmentpath=csvpath.split('.')[0]+'_pretreatment.csv'
blocksize=1024*1024#B
unitsize=50#B

# def copyfile(src_file,dst_file):
#     with open(src_file,'rb') as fsrc,open(dst_file,'wb') as fdst:
#         while True:
#             buf=fsrc.read(blocksize*10)
#             if not buf:
#                 break
#             fdst.write(buf)

print("Loading Data \"{}\" ......".format(csvpath))
with open(csvpath,'r') as csv_file:
    reader=csv.reader(csv_file,delimiter=',')
    next(reader)
    num_rows=0
    for row in reader:
        num_rows+=1
headrows=5
headdata=pd.read_csv(csvpath,nrows=headrows)
num_cols=headdata.shape[1]
chunksize=(int)(blocksize/(unitsize*num_cols))
# copyfile(csvpath,pretreatmentpath)
print(headdata)
# data = pd.read_csv(csvpath)
# 预处理数据，将数据转换为全整数+浮点数类型

# 进行数据的标准化和归一化处理
def normalize(x):
    # ((x-x_mean)/x_std-x_min)/(x_max-x_min)
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_normalized = (x - x_mean) / x_std
    percentiles = [1, 99] # 截断异常值
    pmin, pmax = np.percentile(x_normalized, percentiles)
    x_truncated = np.clip(x_normalized, pmin, pmax)

    # 然后将处理过的数组移动到 0-1 范围内：
    x_min = np.min(x_truncated)
    x_max = np.max(x_truncated)
    x_normalized_truncated = (x_truncated - x_min) / (x_max - x_min)
    return x_normalized_truncated,x_mean,x_std,x_max,x_min

def newnormalize(x,nor):
    x_mean=nor[0]
    x_std=nor[1]
    x_max=nor[2]
    x_min=nor[3]
    x=(x-x_mean)/x_std
    return (x-x_min)/(x_max-x_min)

# 日期类型转换
dateclass = ['date_of_birth', 'enrollment_date']
datenor=[]
dateclass_index=[headdata.columns.get_loc(c) for c in dateclass]
def convert2date(data):
    return datetime.fromisoformat(data).timestamp()
def convertdate(data,dateclass):
    datedata = data[dateclass].values
    intdatedata = np.zeros((datedata.shape[0], len(dateclass)), dtype=int)
    for i in range(len(dateclass)):
        intdatelist=[convert2date(date) for date in datedata[:, i]]
        intdatedata[:,i] = np.array(intdatelist, dtype=int)
        nor_res=normalize(intdatedata[:,i])
        data[dateclass[i]]=nor_res[0]
        datenor.append(list(nor_res[1:]))
        # data[dateclass[i]]=intdatedata[:,i]
# data=pd.read_csv(pretreatmentpath,usecols=dateclass_index)
# convertdate(data,dateclass)
# data.to_csv(pretreatmentpath,index=False)

# bool类型转换
boolclass=['is_married']
boolclass_index=[headdata.columns.get_loc(c) for c in boolclass]
def convert2bool(data):
    if(data):
        return 1
    else:
        return 0

def convertbool(data,boolclass):
    booldata=data[boolclass].values
    intbooldata=np.zeros((booldata.shape[0], len(boolclass)), dtype=int)
    for i in range(len(boolclass)):
        intboollist=[convert2bool(d) for d in booldata[:,i]]
        intbooldata[:,i]=np.array(intboollist, dtype=int)
        data[boolclass[i]]=intbooldata[:,i]
        # data[boolclass[i]]=normalize(intbooldata[:,i])
# data=pd.read_csv(pretreatmentpath,usecols=boolclass_index)
# convertbool(data,boolclass)
# data.to_csv(pretreatmentpath,index=False)

# word2vec
strclass=[]
for col in headdata.columns:
    dt=type(headdata[col].values[0])
    if(dt is np.int64 or dt is np.float64):
        continue
    elif(col in dateclass):
        continue
    elif(col in boolclass):
        continue
    else:
        strclass.append(col)
strclass_index=[headdata.columns.get_loc(c) for c in strclass]

otherclass=headdata.columns.values
otherclass=list(otherclass[[item not in strclass+boolclass+dateclass for item in otherclass]])
otherclass_index=[headdata.columns.get_loc(c) for c in otherclass]

othernor=[]
import os
if(os.path.isfile(pretreatmentpath)):
    os.remove(pretreatmentpath)
for chunk_data in pd.read_csv(csvpath,chunksize=chunksize):
    convertdate(chunk_data,dateclass)
    convertbool(chunk_data,boolclass)
    for i in range(len(otherclass)):
        nor_res=normalize(chunk_data[otherclass[i]])
        chunk_data[otherclass[i]]=nor_res[0]
        othernor.append(list(nor_res[1:]))
    chunk_data.to_csv(pretreatmentpath,mode='a',header=True,index=False)

from gensim.models import word2vec
import os
model_path='./model/'
if not os.path.exists(model_path):
    os.makedirs(model_path) # 如果文件夹不存在，创建文件夹
    print("model文件夹创建成功！")
else:
    print("model文件夹已存在！")

class MySentences(object):
    def __iter__(self):
        for chunk in pd.read_csv(csvpath,chunksize=1):
            yield chunk[strclass].values[0].tolist()
strveclen=50
if(not os.path.isfile(model_path+'word2vec.model')):
    print('Training word2vec......')
    sentences=MySentences()
    str_model=word2vec.Word2Vec(sentences,sg=1,vector_size=strveclen,window=5,min_count=0,negative=3,sample=0.001,hs=1,workers=4)
    str_model.save(model_path+'word2vec.model')
else:
    print('Loading exist word2vec model......')
str_model=word2vec.Word2Vec.load(model_path+'word2vec.model')
print('Finish!')
# str_model.wv['steven']
def convertstr(data,str_model):
    try:
        return str_model.wv[data]
    except KeyError:
        return np.zeros(strveclen,dtype=float)
    
def tovec(oneline,input_size):
    input_v=np.zeros(input_size)
    v_index=0
    for a_i in oneline:
        if(type(a_i) is str):
            vec=convertstr(a_i,str_model)
            # vec=normalize(vec)
            for k in range(strveclen):
                input_v[v_index]=vec[k]
                v_index=v_index+1
        else:
            input_v[v_index]=a_i
            v_index=v_index+1
    # 此时input_v为输入数据
    return input_v

def get_input_size(headdata,i,strveclen):
    input_size=0
    for j in range(i):
        if(headdata.columns[j] in strclass):
            input_size=input_size+strveclen
        else:
            input_size=input_size+1
    return input_size

# 需要为每一个属性构建神经网络来计算P(V_i|v_1,v_2,...,v_{i-1})

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("Using device:",device)
# 需要将模型、数据都移动到GPU上

sizes=[]
# import torch.optim.lr_scheduler as lr_scheduler
def trainingmodel(headdata,strclass,str_model):
    for i in range(num_cols):
        if(i==0):continue
        # 对于每一个属性，神经网络的输入为input_v的拼接
        # input_size=0
        # for j in range(i):
        #     if(headdata.columns[j] in strclass):
        #         input_size=input_size+strveclen
        #     else:
        #         input_size=input_size+1
        # input_size=len(data.columns)+len(strclass)*strveclen-len(strclass)
        input_size=get_input_size(headdata,i,strveclen)
        # 读取第i列
        unique=np.unique(pd.read_csv(pretreatmentpath,usecols=[i]).values)
        output_size=len(unique)
        mid_size=(int)((input_size+output_size)/2)
        hidden_size=256
        sizes.append([input_size,hidden_size,mid_size,output_size])
        if(os.path.isfile(model_path+'net{}.model'.format(i))):
           print("Aleady have "+model_path+'net{}.model'.format(i))
           continue
        # 定义神经网络
        net=Autoencoder(input_size,hidden_size,mid_size,output_size).to(device)
        # 定义优化器
        optimizer = torch.optim.Adam(net.parameters())
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        # 定义损失函数
        criterion = torch.nn.BCELoss().to(device) # 二元交叉熵损失函数
        # # 定义学习率调度器
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        # # 定义自适应学习率
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        total_epoch=10
        for epoch in range(total_epoch):
            for line in range(num_rows): # 一行作为训练单元
                # input_v=np.zeros(input_size)
                # v_index=0
                # for j in range(i):
                #     a_i=data.values[line,j]
                #     if(type(a_i) is str):
                #         for k in range(strveclen):
                #             input_v[v_index]=str_model.wv[a_i][k]
                #             v_index=v_index+1
                #     else:
                #         input_v[v_index]=a_i
                #         v_index=v_index+1
                tmpdata=pd.read_csv(pretreatmentpath,skiprows=line,nrows=1,usecols=list(range(i+1))).values
                input_v=tovec(tmpdata[0,:i],input_size)
                # 此时input_v为输入数据
                # 测试tovec函数
                target_p=np.zeros(output_size)
                # 真实的概率值的计算应该通过统计的方法进行计算，需要计算P(V_i|v_1,v_2,...,v_{i-1})
                indices=np.where(unique==tmpdata[0,i])
                target_p[indices[0]]=1

                # 将input_v和target_p转换为Tensor
                input_v=torch.tensor(input_v).float().to(device)
                target_p=torch.tensor(target_p).float().to(device)

                optimizer.zero_grad() # 将梯度清零
                outputs=net(input_v) # 输入数据，得到输出结果
                loss=criterion(outputs,target_p) # 计算损失函数
                # 在模型的输出里面有参数信息
                loss.backward() # 反向传播
                # 模型参数的.grad属性代表了梯度信息
                optimizer.step() # 更新权重参数
                # scheduler.step()# 学习率调度
                # # 计算验证集损失并调整学习率
                # val_loss = nn.functional.mse_loss(outputs, target_p)
                # scheduler.step(val_loss)
            if((epoch+1)%(total_epoch/10)==0):
                print('Net {} Epoch [{}/{}], Loss:{:.4f}'.format(i,epoch+1,total_epoch,loss.item()))
            # 输出损失函数值，函数值下降说明模型在学习如何拟合数据
        # 保存训练好的网络模型
        torch.save(net.state_dict(),model_path+'net{}.model'.format(i))
trainingmodel(headdata,strclass,str_model)

# 现在模型训练完毕了，应该可以进行预测工作了
# 加载模型
nets=[]
for i in range(num_cols):
    if(i==0):
        continue
    net=Autoencoder(sizes[i-1][0],sizes[i-1][1],sizes[i-1][2],sizes[i-1][3])
    net.load_state_dict(torch.load(model_path+'net{}.model'.format(i)))
    net.to(device)
    nets.append(net)

def auto_regress_model(v,i):
    if(i==1):
        unique=np.unique(pd.read_csv(pretreatmentpath,usecols=[i]).values)
        return [1/len(unique)]*len(unique)
    # 构造输入的向量，得到输出的结果
    # for j in range(i):
    #     if(j in boolclass_index):
    #         v[j]=convert2bool(v[j])
    #     elif(j in dateclass_index):
    #         v[j]=newnormalize(convert2date(v[j]),datenor[dateclass_index.index(j)])
    #     elif(headdata.columns[j] in otherclass):
    #         nor=othernor[otherclass.index(headdata.columns[j])]
    #         v[j]=newnormalize(v[j],nor)
    # 采样是从归一化之后的表格采样，所以只剩下文本没有处理
    input_v=tovec(v[:i],sizes[i-1][0]) # tovec中有对其它字符串的处理
    input_v=torch.tensor(input_v).float().to(device)
    with torch.no_grad():
        return nets[i-1](input_v)

def renormalize(pv,R,i):
    if(len(R[i])==0):
        return 1
    coldata=pd.read_csv(pretreatmentpath,usecols=[i])
    unique=np.unique(coldata.values)
    p=0
    for j,u in enumerate(unique):
        if(i in strclass_index): # 字符串类型
            if(u in R[i]):
                p=p+pv[j]
        elif(R[i][0]==R[i][1]):
            if(u==R[i][0]):
                p=pv[j]
                break
        # elif(i in boolclass_index): # bool类型
        #     if(u==R[i][0] or u==R[i][1]):
        #         p=p+pv[j]
        elif(R[i][0]<=u and u<=R[i][1]):
            p=p+pv[j]
    return p/sum(pv)

def sample(R,attr_idx,pv):
    if(attr_idx!=1):
        pv=pv.cpu()
        pv=np.array(pv,dtype='double')
    pvsum=sum(pv)
    if(pvsum!=1):
        pv=[p/pvsum for p in pv]
    while True:
        unique=np.unique(pd.read_csv(pretreatmentpath,usecols=[attr_idx]).values)
        assert(len(unique)==len(pv))
        if(len(R[attr_idx])==0):
            sample_index=np.random.choice(list(range(len(pv))),size=1,p=pv)
            sample_value=unique[sample_index]
            return sample_value
        elif(attr_idx in strclass_index): # 字符串类型
            satisfy=[u in R[attr_idx] for u in unique]
            unique=unique[satisfy]
            if(len(unique)==0):
                return [' ']
            sat_pv=[p for i,p in enumerate(pv) if(satisfy[i])]
            sumpv=sum(sat_pv)
            sat_pv=[p/sumpv for p in sat_pv]
            sample_index=np.random.choice(list(range(len(unique))),size=1,p=sat_pv)
            sample_value=unique[sample_index]
            return sample_value
        else:
            satisfy=[R[attr_idx][0]<=u and u<=R[attr_idx][1] for u in unique]
            unique=unique[satisfy]
            if(len(unique)==0):
                return [0]
            sat_pv=[p for i,p in enumerate(pv) if(satisfy[i])]
            sumpv=sum(sat_pv)
            sat_pv=[p/sumpv for p in sat_pv]
            sample_index=np.random.choice(list(range(len(unique))),size=1,p=sat_pv)
            sample_value=unique[sample_index]
            return sample_value

def convertR(R):
    for i in range(len(R)):
        if(len(R[i])==0):
            continue
        if(i in strclass_index): # 字符串类型
            continue
        elif(i in dateclass_index): # 日期类型
            tmp=R[i][0]
            R[i][0]=newnormalize(convert2date(tmp),datenor[dateclass_index.index(i)])
            tmp=R[i][1]
            R[i][1]=newnormalize(convert2date(tmp),datenor[dateclass_index.index(i)])    
        elif(i in boolclass_index): # bool类型
            tmp=R[i][0]
            R[i][0]=convert2bool(tmp)
            tmp=R[i][1]
            R[i][1]=convert2bool(tmp)
            if(R[i][0]>R[i][1]):
                tmp=R[i][1]
                R[i][1]=R[i][0]
                R[i][0]=tmp
        else: # 整数和浮点数类型
            tmp=R[i][0]
            R[i][0]=newnormalize(tmp,othernor[otherclass_index.index(i)])
            tmp=R[i][1]
            R[i][1]=newnormalize(tmp,othernor[otherclass_index.index(i)])

    return R
            
        

# 算法1.2 基于Naru模型的范围查询基数估计
def cardinality_estiamte(S,R):
    # 输入：采样点个数S，R={R1,R2,R3,...,Rm}查询条件的取值范围
    # 输出：满足查询条件的联合概率P
    P=0
    R=convertR(R) # 最开始输入的R和后面采样的R不一样，最开始的是字符串类型的日期
    for j in range(S): # 采样S个采样点，并计算对应的联合概率
        p=1 # 计算查询语句的实际归一化基数估计值
        v=[0]*len(R)
        for i in range(1,len(R)):
            # 将v输入自回归模型得到P(Vi|v1,v2,...,vi-1)
            pv=auto_regress_model(v,i)
            # 在取值范围内将pv重新归一化得到P(Vi|v1,...,vi-1,Vi∈Ri)
            p=p*renormalize(pv,R,i)
            if(p==0):
                break
            v[i]=sample(R,i,pv)[0] # 对属性Ai的取值进行采样
        # print("Sample {} p:{:.4f}".format(j+1,p))
        P=P+p
    P=P/S
    return P

def q_error(estimate_num,actual_num):
    if(min(estimate_num,actual_num)==0):
        return max(estimate_num,actual_num)/1e-5
    return max(estimate_num,actual_num)/min(estimate_num,actual_num)

# S=30
# R=[[],[],[],
#    [10,30],# 年龄
#    [],[],[],
#    [],[],[],]
# P=cardinality_estiamte(S,R)
# if(type(P) is torch.Tensor):
#     P=P.cpu()
# predict_num=P*num_rows

# df=pd.read_csv(csvpath)
# res=df.query('age>=10 & age<=30')
# real_num=len(res)
# print("预测的个数为：",int(predict_num))
# print("实际的个数为：",real_num);
# print("Q-error：{:.4f}".format(q_error(predict_num,real_num)))

def randomR():
    R=[]
    for i in range(num_cols):
        coldata=pd.read_csv(csvpath,usecols=[i])
        a,b=np.sort(np.random.choice(coldata.values[:,0],size=2,replace=False))
        # replace=False代表两个值不会重复，sort保证a<b
        R.append([a,b])
    return R

def randomRi(i,seed):
    R=[]
    np.random.seed(seed)
    coldata=pd.read_csv(csvpath,usecols=[i])
    a,b=np.sort(np.random.choice(coldata.values[:,0],size=2,replace=False))
    # replace=False代表两个值不会重复，sort保证a<b
    for j in range(num_cols):
        if(j==i):
            R.append([a,b])
        else:
            R.append([])
    return R

def bool2str(b):
    if(b):
        return '1'
    else:
        return '0'

def real_num(R):
    df=pd.read_csv(csvpath)
    sql=""
    for i,r in enumerate(R):
        if(len(r)==0):
            continue
        attr=headdata.columns[i]
        if(i in strclass_index):
            sql=sql+attr+"=='"+r[0]+"' | "+attr+"=='"+r[1]+"'"
        elif(i in boolclass_index):
            sql=sql+attr+"=="+bool2str(r[0])+" | "+attr+"=="+bool2str(r[1])
        elif(i in dateclass_index):
            sql=sql+attr+">='"+str(r[0])+"' & "+attr+"<='"+str(r[1])+"'"
        else:
            sql=sql+attr+">="+str(r[0])+" & "+attr+"<="+str(r[1])
    try:
        print("查询条件为："+sql)
        res=df.query(sql)
        return len(res)
    except KeyError:
        return 0

def pre_num(R):
    S=50
    P=cardinality_estiamte(S,R)
    if(type(P) is torch.Tensor):
        P=P.cpu()
    pre_num=P*num_rows
    return pre_num

def test_func(R):
    # print("查询条件")
    # for i,r in enumerate(R):
    #     if(len(r)==0):
    #         continue
    #     attr=headdata.columns[i]
    #     if(i in strclass_index):
    #         print(attr+"=='"+r[0]+"' | "+attr+"=='"+r[1]+"'")
    #     elif(i in boolclass_index):
    #         print(attr+"=='"+bool2str(r[0])+"' | "+attr+"=='"+bool2str(r[1])+"'")
    #     elif(i in dateclass_index):
    #         print(attr+">='"+str(r[0])+"' & "+attr+"<='"+str(r[1])+"'")
    #     else:
    #         print(attr+">="+str(r[0])+" & "+attr+"<="+str(r[1])+"")
    rn=real_num(R)
    pn=pre_num(R)
    print("实际个数：",rn)
    print("预测个数：",int(pn))
    qError=q_error(pn,rn)
    print("Q-error：{:.4f}".format(q_error(pn,rn)))
    return qError

sumqError=0
for i in range(num_cols):
    R=randomRi(i,10)
    sumqError=sumqError+test_func(R)
print("Mean q-error: {:.4f}".format(sumqError/num_cols))

# R=[
#     [],[],[],[],
#     [],[],[],[],
#     [],[True,False]
# ]
# test_func(R)

print('Finish!')