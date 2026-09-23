import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")      #读取文件：read_文件格式（）
print(df.head())        # 默认为前五行数据
print(df.shape)         # 查看数据一共多少行多少列
print(df.info())        # 查看每一列的列名，数据类型
print(df.isnull().sum())    #查看每列缺失值数量

print("-------------------------------一、认识数据-------------------------------")
# 一、认识数据
#1. 一共有多少乘客：
print(f"乘客数为：{df.shape[0]}")

#2. Titanic 一共有多少个字段？
print(f"字段数为：{df.shape[1]}")

#3. 查看所有字段名称
#应该使用df.columns
print(f"字段名称分别为：{df.columns}")

#4. 统计男性和女性各有多少人
#先df["列名称"]选中这一列，然后value_counts()获得值与对应数量
print(df["Sex"].value_counts())

#5. 统计有多少人活了下来
#补充：value_counts()括号内的参数
#(1) normalize=True: 显示比例
#(2) 默认sort=True: 按数量从多到少排列
#(3) dropna=True: 不统计NAN缺失值
print(df["Survived"].value_counts())

#6. 统计 Titanic 中三个舱位（Pclass）分别有多少人
print(df["Pclass"].value_counts())

#7. 只求男性有多少人
#方法一： 用value_counts()["male"]
print("男性数量为：",df["Sex"].value_counts()["male"])

#方法二： .sum()
print("女性数量为：",(df["Sex"]=='female').sum())

print()
print("-------------------------------二、进一步探索数据-------------------------------")
print()
# 二、进一步探索数据
#1. 男女的生还人数分别是多少？
print("男性的生还人数为：", ((df["Survived"]== 1) & (df["Sex"]=='male')).sum())
print("女性的生还人数为：", ((df["Survived"]== 1) & (df["Sex"]=='female')).sum())

#2. 计算男女各自的生还率
print("男性的生还率为：",round((((df["Survived"]== 1) & (df["Sex"]=='male')).sum() / (df['Sex'] == "male").sum()),3))
print("女性的生还率为：",round((((df["Survived"]== 1) & (df["Sex"]=='female')).sum() / (df['Sex'] == "female").sum()),3))

#3. 一二三等车厢的生还人数分别是多少？
print("一等车厢的生还人数是: ",((df["Pclass"]== 1) & (df["Survived"]==1)).sum())
print("二等车厢的生还人数是: ",((df["Pclass"]== 2) & (df["Survived"]==1)).sum())
print("三等车厢的生还人数是: ",((df["Pclass"]== 3) & (df["Survived"]==1)).sum())

#4. 三个车厢的生还率分别是多少？
print("一等车厢的生还率为： ",round((((df['Survived']==1)&(df['Pclass']==1)).sum() / (df['Pclass']==1).sum()),3))
print("二等车厢的生还率为： ",round((((df['Survived']==1)&(df['Pclass']==2)).sum() / (df['Pclass']==2).sum()),3))
print("三等车厢的生还率为： ",round((((df['Survived']==1)&(df['Pclass']==3)).sum() / (df['Pclass']==3).sum()),3))

#5. 用groupby简化上述代码          df.groupby("分组列")["要计算的列"].统计方法()
#总人数：
print("总人数：", df.groupby("Pclass")["Survived"].size())          #size()用于统计一组中有多少行数据
#生还人数：
print("生还人数： ",df.groupby("Pclass")["Survived"].sum())
#生还率：
print("生还率： ",df.groupby("Pclass")["Survived"].mean())

print()
print("-------------------------------三、数据清洗部分(age)-------------------------------")
print()
#三、数据清洗部分(age)
#1. 找出缺失值及数量
print(df.isnull().sum())

#2. 查看哪些人age值缺失         df[***]表示只保留***中返回true的行
print(df[df["Age"].isnull()]["Name"])

#分男女
print("男性Age缺失：",df[(df["Age"].isnull()) & (df["Sex"] == 'male')])
print("女性Age缺失：",df[(df["Age"].isnull()) & (df["Sex"] == 'female')])

#3. 直接删除age缺失值( dropna(subset=[]) )
#print(f"处理前的行数： ",df.shape[0])
#df = df.dropna(subset=["Age"])
#print(f"处理后的行数： ",df.shape[0])

#4. 用某值填充age缺失值( fillna() )
#df["Age"] = df["Age"].fillna(30)
#print("处理后Age的缺失值： ",df["Age"].isnull().sum())

#5. 分别用男女平均年龄填充
#求平均年龄
print(round(df.groupby("Sex")["Age"].mean(),3))

#用平均年龄分别填充男女的age缺失值
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"),"Age"] = 30.727
print("总缺失值：",df["Age"].isnull().sum())
print("男性缺失值：",((df["Age"].isnull()) & (df["Sex"]=="male")).sum())
print("女性缺失值：",((df["Age"].isnull()) & (df["Sex"]=="female")).sum())
print("-------------")
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"),"Age"] = 27.916
print("总缺失值：",df["Age"].isnull().sum())
print("男性缺失值：",((df["Age"].isnull()) & (df["Sex"]=="male")).sum())
print("女性缺失值：",((df["Age"].isnull()) & (df["Sex"]=="female")).sum())

6.#查看age最终总体分布情况
print("age最终总体分布情况",df["Age"].describe())
print()
#男性
print("男性",df[df["Sex"]=='male']["Age"].describe())
print()
#女性
print("女性",df[df["Sex"]=='female']["Age"].describe())

print()
print("-------------------------------四、数据清洗部分(cabin)-------------------------------")
print()
#四、数据清洗部分(cabin)    即船舱号
#1. 查看缺失值数量
print("缺失值数量为：",df["Cabin"].isnull().sum())

#2. 计算缺失率
print("缺失率为：", round(df["Cabin"].isnull().sum() / df.shape[0],3))
print(df["Cabin"].value_counts())

#3. 删除Cabin列（缺失值过高，参考价值不大）
df = df.drop(columns = ["Cabin"])
print(df.shape)

print()
print("-------------------------------五、数据清洗部分(embarked)-------------------------------")
print()
#五、数据清洗部分(embarked)    即登船港口
#1. 查看缺失值数量
print("缺失值数量为：",df["Embarked"].isnull().sum())

#2. 找到缺失值的信息
print(df[df["Embarked"].isnull()])

#3. 根据相似特征推断目标值
#其他为一等舱的人，他们的embarked是什么
print("一等舱",df[df['Pclass']==1]["Embarked"])
print()
#但基本没啥用，还是看embarked的众数
print(df["Embarked"].mode())

#4. 填充为众数S
df["Embarked"] = df["Embarked"].fillna("S")
print("Embarked剩余空值： ",df["Embarked"].isnull().sum())

print()
print("-------------------------------六、数据清洗收尾-------------------------------")
print()

""" 总结：
读取 Titanic
   ↓
检查数据结构
   ↓
发现缺失值
   ↓
Age：按性别平均年龄填充
   ↓
Cabin：缺失率 77.1%，删除整列
   ↓
Embarked：缺失 2 个，用众数 S 填充
   ↓
最后检查
   ↓
891 × 11，全部无缺失 ✅
"""
print(df.info())
print(df.isnull().sum())
print(df.shape)

#保存清洗后的数据
df.to_csv("titanic_cleaned.csv", index=False)
print("ok!")

