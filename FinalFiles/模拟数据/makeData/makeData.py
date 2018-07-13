# -*- coding:utf-8 -*-
""" 
   @author: xiaolinzi
   @file: data.py 
   @time: 2018/05/07
"""
import os
import random

def get_random(max_num,min_num):
    max_num = float(max_num)
    min_num = float(min_num)
    rap = max_num - min_num
    return random.random() * rap + min_num

def get_data(file_name):
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name, file_name)
    with open(file_path, 'r') as f:
        data = f.read()
    tmp_data = data.split()
    arr1 = tmp_data[:20] # 最大值
    arr2 = tmp_data[20:] # 最小值
    res = []
    for max_num, min_num in zip(arr1, arr2):
        result = get_random(max_num,min_num)
        res.append(round(result,5))

    return res

def save_data(file_name,data):
    dir_name = os.path.dirname(__file__)
    file_path = os.path.join(dir_name,file_name)
    str_res = ''
    for day in data:
        str_res += ','.join([str(tmp) for tmp in day]) + ",Happy"+'\n'
    with open(file_path,'w') as f:
        f.write(str_res)


if __name__ == '__main__':
    file_name = 'happy2.txt'
    datas = []
    for _ in range(400):
        res = get_data(file_name)
        datas.append(res)

    file_name = 'data_happy2.txt'
    save_data(file_name,datas)

