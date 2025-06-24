import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

# ------------------------------------------excle和csv-------------------------------------------------------------------
# 使用read_excel读取Excel文件(excel文件不用制定encoding，而csv文件需要指定),sheet_name=0默认就是第一个工作表所以在此处不再指定
# 由于原工作表有索引列，所以把第一列作为索引列，默认是None会自动创建一个从0开始的索引
df = pd.read_excel('Sample - Superstore.xls', index_col=0)
# 输出行数和列数
print(f"该工作表有 {df.shape[0]} 行，{df.shape[1]} 列。")
# 查看前5行
print(df.head(5))

# %%
# 缺失值---重复值---异常值
# 缺失值
print(df.isnull().sum())
# print(df.isnull().head(5))
# 保留至少有2个非空值的行
df = df.dropna(thresh=2)
# 删除缺失值所在的列
df = df.dropna(subset=['Postal Code'])
print(df.isnull().sum())
# 重复值
# 删除重复记录drop_duplicates用于删除重复值
df.drop_duplicates(keep='first',inplace=True)
# 异常值
Percentile = np.percentile(df['Sales'], [0, 25, 50,75, 100])
IQR = Percentile[3] - Percentile[1]
UpLimit = Percentile[3] + IQR*1.5  # 计算临界值上界
LowLimit = Percentile[1] - IQR * 1.5  # 计算临界值下界
# 1.IQR判断异常值
abnormal = [i for i in df['Sales'] if i > UpLimit or i < LowLimit]
# 等价于abnormal=[]
# for i in de['sales']:
#       if i > UpLimit or i < LowLimit:
#             abnormal.append(i)
print('箱型图的四分位距（IQR）检测出的array中异常值为：\n', abnormal)
print('箱型图的四分位距（IQR）检测出的异常值比例为：\n', len(abnormal) / len(df['Sales']))
# 2.利用3sigma原则对异常值进行检测
Sales_mean = np.array(df['Sales']).mean()  # 计算平均值
Sales_std = np.array(df['Sales']).std()  # 计算标准差
Sales_cha = df['Sales'] - Sales_mean  # 计算元素与平均值之差
print(f'Sales列平均值为：{Sales_mean}')
print(f'Sales列标准差为：{Sales_std}')
print(Sales_cha.head(5))
# 返回异常值所在位置
# 重置索引（在数据清洗后添加）
df = df.reset_index(drop=True)  # drop=True表示不保留原索引
# 使用重置后的索引
ind = [i for i in range(len(Sales_cha)) if np.abs(Sales_cha.iloc[i]) > 3 * Sales_std]
abnormal = df['Sales'].iloc[ind].tolist()  # 返回异常值
print('3sigma原则检测出的array中异常值为：\n', abnormal)
print('3sigma原则检测出的异常值比例为：\n', len(abnormal) / len(df['Sales']))
# 删除异常值
df_clean_iqr = df[(df['Sales'] <= UpLimit) & (df['Sales'] >= LowLimit)]
print(f"\n清洗后数据量：{df_clean_iqr.shape[0]}行（删除{len(df)-len(df_clean_iqr)}行异常值）")

# %%

# 遍历读取每一行每一列的数据
# for row in range(df.shape[0]):
#     for col in range(df.shape[1]):
#         cell_value = df.iloc[row, col]
#         print(cell_value)
# 输出列名
print(df.columns)
# 输出每个行的值
print(df.values)

# 筛选（取Sales列大于Sales列均值的列为df_1）
df_1 = df[df['Sales'] > df['Sales'].mean()]
print(df_1)

# DataFrame的分组和聚合
# 按Region分组并保存分组对象
region_group = df.groupby('Region')
print(region_group.groups)
# 计算每个Region的销售总额
region_sales = region_group['Sales'].sum()
print(region_sales)

# 按Region和State分组并保存分组对象
region_state_group = df.groupby(['Region', 'State'])
print(region_state_group.groups)  # 查看按Region和State分组的结果
# 计算每个Region和State组合的平均利润
region_state_profit = region_state_group['Profit'].mean()
print(region_state_profit)


'''
完整的read_csv参数
pd.read_csv(filepath,sep=',',header='infer',names=None,index_col=None,dtype=None,engine=None,nrows=None,encoding=None)
header:int或sequence，表示将某行数据作为列名，默认为infer，表示自动识别
names:array，表示列名
index_col:int,sequence,Flase,代表索引的位置，取sequence代表为多重索引
dtype:dict,代表写入的数据类型，默认None
engine:c，python，表述数据解析引擎
nrows:int，表示读取前n行
'''


# -------------------------------------------------csv-----------------------------------------------------------------
#  header=None表示不将第一行作为列名而是把所有的都视为数据
# 如果没有这个参数low_memory=False，他会提示你这个文件中有的数据列中有浮点型又有字符串型啥的报错，如果有了这个参数之后就不会报错了
data1 = pd.read_csv(r"C:\Users\86195\Desktop\飞桨项目\file\class\1.fetch\销售流水记录2.csv", header=None, encoding='gb18030', low_memory=False)
print('使用read_csv读取的销售流水记录表2的长度为：', len(data1))
# 输出从第0行到第4行，第0列到第3列
print(data1.iloc[0:5, 0:4])

# 导入os包用来解析路径
import os
# 定义目录路径（这里是反斜杠）
directory = '/Users/86195/Desktop/飞桨项目/file/class/1.fetch/'
# 创建目录
os.makedirs(directory, exist_ok=True)
print('销售流水记录表1写入文本文件前目录内文件列表为：\n', os.listdir('/Users/86195/Desktop/飞桨项目/file/class/1.fetch/'))
data1.to_csv('/Users/86195/Desktop/飞桨项目/file/class/1.fetch/salesave.csv', sep=';', index=False)  # 将data1以CSV格式存储
print('销售流水记录表表写入文本文件后目录内文件列表为：\n', os.listdir('/Users/86195/Desktop/飞桨项目/file/class/1.fetch/'))


# ------------------------------------------------sql-------------------------------------------------------------------
# 连接和操作mysql，原生sql需求
import pymysql
# 创建数据库连接
import sqlalchemy

# 创建一个mysql连接器，用户名为root，密码为Chaohua020830
# 地址为mysql.sqlpub.com:3306，数据库名称为sale2、数据库文件可见/home/aistudio/data/class/1.fetch/sale2.sql
sqlalchemy_db = sqlalchemy.create_engine(
    'mysql+pymysql://root:Chaohua020830@localhost:3306/test')
print(sqlalchemy_db)

# read_sql_query查看test数据库中的数据表数目,con=sqlalchemy_db,表示传入之前的sql连接
formlist = pd.read_sql_query('show tables', con=sqlalchemy_db)
print('test数据库数据表清单为：', '\n', formlist)
# 使用read_sql_table函数读取表sale2
detail1 = pd.read_sql_table('sale2', con=sqlalchemy_db)
print('使用read_sql_table读取sale2表的长度为：', len(detail1))
# 使用read_sql函数读取表sale2
detail2 = pd.read_sql('select * from sale2', con=sqlalchemy_db)
print('使用read_sql函数 + sql语句读取销售流水记录表的长度为：', len(detail2))
detail3 = pd.read_sql('sale2', con=sqlalchemy_db)
print('使用read_sql函数+表格名称读取的销售流水记录表的长度为：', len(detail3))

# to_sql存储orderData
detail1.to_sql('sale_copy', con=sqlalchemy_db, index=False, if_exists='replace')
# read_sql读取test表格
formlist1 = pd.read_sql_query('show tables', con=sqlalchemy_db)
print('新增一个表格后test数据库数据表清单为：', '\n', formlist1)
# read_sql_query	执行任意 SQL 查询，返回 DataFrame
# read_sql_table	直接读取单个表，自动解析表结构
# read_sql	通用函数，支持 SQL 语句或表名，整合前两者功能


'''
基础概念：
NumPy库
数组计算包括但不限于多维数组，线性代数，傅里叶变换，随机数生成
两种对象：n维数组对象，ndarray和ufunc函数
Pandas库
使用基础是NumPy，可以提供高性能的矩阵运算
用于数据挖掘和数据分析，也可以数据清洗
可以读取各种格式的数据源，且返回的对象相同
Matplotib库
常与NumPy库结合，可以创建静态，动态和交互式的图表
Seaborn库
建立在Matplotib库基础上，集成了pandas的数据结构
通过更简洁的API绘制信息更丰富，更具吸引力的图像
常与pandas结合
Sklearn库
机器学习的工具
数据挖掘和数据分析

'''

'''
series:可以保存不同类型的数据一维数组，与numpy中的一维array相似
DataFrame:二维，相当于series的容器，DataFrame的单列数据为第一个series
Panel:三维

df的一些属性
---在不指定行索引值的时候---
df.values返回表对象
df.index获取行索引
df.columns获取列索引（一般为列名）
df.shape获取数据行列数
df.ndim查看数据是几维的，通常是二维的
df.size行*列
df.loc['行索引名','列索引名']返回指定行列的值
df.iloc['1','1']返回第2行2列的值
df.axes获取列索引
df.T行列对调
df.info()相比于values更详细的查看对象类型
df.head(n)查看前n行数据
df.tail(n)查看后n行数据
df.describe()对对象-计数，平均数，标准差，最小值，最大值等
df['列名'].value_counts()查看数据表中指定列的不同值各有多少
df.count()获取各列的非空值

----指定行索引之后（行索引就不算数据列了）---
head，tail，loc，iloc都要相应的发生变化
'''



# ------------------------------------------缺失值---重复值----异常值-------------------------------------------------------
'''
缺失值
重复值
异常值
'''
#缺失值的检测---isnull(),notnull(),isna(),notna()
#1、定义方法
def missing_values_table(df):
    # 计算所有的缺失值
    mis_val = df.isnull().sum()

    # 计算缺失值比例
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # 将结果拼接成dataframe
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # 将列重命名
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: '缺失值', 1: '占比（%）'})

    # 按照缺失值降序排列，并把缺失值为0的数据排除
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '占比（%）', ascending=False).round(1)

    # 打印信息
    print("传入的数据集中共 " + str(df.shape[1]) + " 列.\n"
                                           "其中 " + str(mis_val_table_ren_columns.shape[0]) +
          "列包含缺失值")

    # 返回缺失值信息的dataframe
    return mis_val_table_ren_columns
# 调用刚刚写的方法，来检测名为train的这个DataFrame中，每一列缺失比例
train_missing= missing_values_table(df)
train_missing

#2、导这个包可视化
import missingno as msno
# 使用条形图来展示缺失情况
msno.bar(df_1)
# 使用矩阵图来展示缺失情况
msno.matrix(df_1)

'''
缺失值的处理
1、缺失值只有几个数据可以直接用dropna()按行删除即可
2、一列80%以上数据都缺drop()把列删了
3、缺失数据的这一列，数据不是时间序列类型，使用均值or前后填充即可
4、缺失数据为时间序列类型，使用线性查补法来差补数据

删除缺失值、填充缺失值、差补缺失值
1、df.dropna(axis=0, how='any',thresh=None,subset=None,inplace=False),用于删除缺失值所在地一行或一列
axis=0按行删，=1按列删
how='any'当任何值为NaN时就删除整行整列，='all'所有行为NaN才删
thresh=3，只要这行或列有3个及以上的非空值就不删
subset，删除指定列
inplace=True直接修改原数据，False修改原数据的副本
2、df.fillna(value=None,method=None,axis=None,inplace=False,limit=None,downcast=None)
value填充的数据，可以为变量，字典，series，df对象
method：pad或ffill，使用缺失值前面的值填充缺失值。backfill或bfill使用后面的填充缺失值
axis=0填充缺失值的行，=1填充列
limit连续填充的最大数量，int值
3、df.interpolate(method='linear',axis=0,limit=None,inplace=False,limit_direction=None,limit_area=None,downcast=None,**kwargs)
method:填充方法linear，time（时间），index或values，nearest（临近插值），barycentric（重心坐标差值）
limit_direction:按指定方向进行填充，forward，backforward，both
'''

#重复值的检测
print(df.duplicated(subset=None,keep='first'))
#subset表示识别重复的列索引，keep表示以什么方式保留重复项，first保留第一项，last保留最后一项，False将所有相同的数据标记为重复项

#筛选出df中重复值标记为True的数据记录
print(df[df.duplicated()])
'''
重复值的处理
df.drop_duplicated(subset=None,keep='first',inplace=False,ignore_index=False)
inplace=True更新原数据，=False不更新原数据
ignore_index=True对删除重复值的对象的行索引重新排序，=False不重新排序
'''
#删除重复值
print(df.drop_duplicates())
df = df.drop_duplicates()


#异常值的检测
#1、3西格玛
#定义方法，输入df中的列数据，返回异常值及索引
def three_sigma(ser):
    # 计算平均值
    mean_data=ser.mean()
    # 计算标准差
    std_data=ser.std()
    # 小于μ-3σ或大于μ+3σ的数据均为异常值
    rule=(mean_data-3*std_data > ser) | (mean_data+3*std_data < ser)
    # 然后np.arange方法生成一个从0开始，到ser长度-1结束的连续索引，再根据rule列表中的True值，直接保留所有为True的索引，也就是异常值的行索引
    index=np.arange(ser.shape[0])[rule]
    # 获取异常值
    outliers=ser.iloc[index]
    return outliers

# 对Sales列进行异常值检测，只要传入一个数据列
print(three_sigma(df['Sales']))
# 2、使用箱线图进行检测异常值（直观）
df.boxplot(column='Sales')
plt.show()
#定义方法实现自动抓取异常值的索引及值
def box_outliers(ser):
    #对待检测的数据集进行排序
    new_ser=ser.sort_values()
    # 判断数据的总数量是奇数还是偶数
    if new_ser.count()%2==0 :
        #计算Q3，Q1,IQR
        Q3=new_ser[int(len(new_ser)/2):].median()
        Q1=new_ser[:int(len(new_ser)/2)].median()
    elif new_ser.count()%2 !=0 :
        Q3=new_ser[int(len(new_ser)/2-1):].median()
        Q1=new_ser[:int(len(new_ser)/2-1)].median()
    IQR=round(Q3-Q1,1)
    rule=(round(Q3+1.5*IQR,1)<ser) | (round(Q1-1.5*IQR,1)>ser)
    index=np.arange(ser.shape[0])[rule]
    #获取异常值及其索引
    outliers=ser.iloc[index]
    return outliers
print(box_outliers(df['Sales']))


'''
异常值的处理
1、删除异常值
df.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors=‘raise’)
labels表示要删除的航索引和列索引，df.drop([0,1])删除第一行和第二行
2、替换异常值
df.replace(to_replace=None, Value=None, inplace=False, limit=None, regex=False, method=‘pad’)
to_place 表示被替换的值
'''
