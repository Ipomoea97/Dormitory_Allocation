import pandas as pd

# 读取Data.xlsx文件
df = pd.read_excel('Data.xlsx')
print('数据列名:')
print(df.columns.tolist())
print('\n前5行数据:')
print(df.head())
print('\n数据形状:', df.shape)