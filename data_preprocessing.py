import numpy as np
import pandas as pd
import codecs
from numpy import *

# 读取txt->矩阵
# A = zeros((178935, 3), dtype=int)  # 先创建一个全零方阵A，并且数据的类型设置为int浮点型
A = zeros((136233, 5), dtype=int)  # 先创建一个全零方阵A，并且数据的类型设置为int浮点型
# dataset_TSMC2014_NYC_new
f = open('./data/FNYC.txt')  # 打开数据文件文件
lines = f.readlines()  # 把全部数据文件读到一个列表lines中
# print(lines)
A_row = 0  # 表示矩阵的行，从0行开始
print(A)
for line in lines:  # 把lines中的数据逐行读取出来
    list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以\t来分割行数据，然后把处理后的行数据返回到list列表中
    A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
    A_row += 1  # 然后方阵A的下一行接着读
print(A)

# 将矩阵以.npy形式存储
np.save('./data/FNYC.npy', A)
test = np.load('./data/FNYC.npy', encoding="latin1")  #加载文件
# doc = open('2NYC2.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)  #将打印内容写入文件中

# 读取txt->矩阵
# A = zeros((9985, 3), dtype=float)  # 先创建一个全零方阵A，并且数据的类型设置为int浮点型
A = zeros((1953, 3), dtype=float)  # 先创建一个全零方阵A，并且数据的类型设置为int浮点型
# dataset_TSMC2014_NYC_new
f = open('./data/FNYC_POI.txt')  # 打开数据文件文件
lines = f.readlines()  # 把全部数据文件读到一个列表lines中
# print(lines)
A_row = 0  # 表示矩阵的行，从0行开始
for line in lines:  # 把lines中的数据逐行读取出来
    list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以\t来分割行数据，然后把处理后的行数据返回到list列表中
    A[A_row:] = list[0:3]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
    A_row += 1  # 然后方阵A的下一行接着读
print(A)

# 将矩阵以.npy形式存储
np.save('./data/FNYC_POI.npy', A)
test = np.load('./data/FNYC_POI.npy', encoding="latin1")  #加载文件
# doc = open('2NYC2_POI.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)  #将打印内容写入文件中


