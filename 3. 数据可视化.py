import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

df = pd.read_csv("titanic_cleaned.csv")

#一: 男女生还率柱状图(bar)
print("一: 男女生还率柱状图")
plt.figure()
#1. 准备数据:
survival_sex = round(df.groupby("Sex")["Survived"].mean(),4)

#2. 建表  plt.bar(横轴 index, 纵轴 values)
chart1 = plt.bar(survival_sex.index,survival_sex.values)

#3. 添加标题与坐标轴名称
plt.title("Titanic Survival Rate by Sex")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

#4. 纵轴生还率变为百分比  ylim:设置纵轴显示范围
plt.ylim(0,1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

#5. 在柱子顶部显示具体生还率 (text: 在图的指定位置添加文字) 利用循环
for i in range(len(survival_sex.values)):
    value = survival_sex.values[i]
    plt.text(i,value,f"{value:.1%}",ha="center")        #ha表示自动调到最中间

#plt.show()

print()
print("------------------------------------")
print()

#二: 不同舱位的生还率柱状图(bar)
print("图表二: 不同舱位的生还率图:")
plt.figure()
#1. 准备数据
survival_class = df.groupby("Pclass")["Survived"].mean()

#2. 建表
chart2 = plt.bar(survival_class.index,survival_class.values)

#3. 添加标题与坐标轴名称
plt.title("Titanic Survival Rate by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")

#4. 修改xy轴图例
plt.xticks([1,2,3],["first class","second class","third class"])

plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# 柱子上添加百分比
for i in range(len(survival_class.values)):
    value = survival_class.values[i]
    plt.text(i + 1, value + 0.02, f"{value:.1%}", ha="center")

#plt.show()

print()
print("------------------------------------")
print()

#三: 所有乘客年龄分布直方图(hist)
print("三: 年龄分布直方图")
plt.figure()
#1. 准备数据
age = df["Age"]

#2. 建图
chart3 = plt.hist(age,bins=10,edgecolor='black')
                #不用自行设置xy轴, bins设置区间数量 , edgecolor设置分界线颜色
#3. 命名
plt.title("Titanic Passenger Age Distribution")
plt.xlabel("age")
plt.ylabel("number")

plt.tight_layout()
plt.show()

print()
print("------------------------------------")
print()

#四: 年龄与票价的散点图(scatter)
print("四: 年龄与票价的散点图")
plt.figure()
#1. 准备数据
age = df["Age"]
fare = df["Fare"]

#2. 创建图表
chart4 = plt.scatter(age,fare)

#3. 命名
plt.title("The relation of age and fare")
plt.xlabel("age")
plt.ylabel("fare")

plt.ylim(0,250)

plt.show()

print()
print("------------------------------------")
print()

#五. 不同年龄生还率的柱状图(bar)
print("五. 不同年龄生还率的柱状图")
plt.figure()
#1. 数据准备
survival_age = df.groupby("Age")["Survived"].mean()

#2. 建表
chart5 = plt.bar(survival_age.index,survival_age.values)
AgeOrder = ["婴儿","儿童","青年","年轻人","中年","中老年","老年"]
survival_age = survival_age.reindex(AgeOrder)

#3. 命名
plt.title("Titanic Survival Rate by Age")
plt.xlabel("age")
plt.ylabel("survival rate")


plt.show()







