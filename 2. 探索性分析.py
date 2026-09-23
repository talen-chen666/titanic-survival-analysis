import pandas as pd
import numpy as np

df = pd.read_csv("titanic_cleaned.csv")

#一、 整体概况：
print("一、 整体概况：")
#1. 有多少人生还，多少人死亡：
print(df["Survived"].value_counts())
print("生还人数为：",df["Survived"].value_counts()[1])
print("死亡人数为：",df["Survived"].value_counts()[0])
print()

#2. 求整体生还率
print("整体生还率为：",round(df["Survived"].mean(),3))
print()
print("----------------------------------------")
print()

#二、性别与生还的关系
print("二、性别与生还的关系")
#1. 男女的生还率分别是多少
print("男女生还率分别为：",round(df.groupby("Sex")["Survived"].mean(),3))
print()

#2. 男女生还的各有多少人
print("男女生还人数：",df.groupby("Sex")["Survived"].sum())
print()
print("----------------------------------------")
print()

sex_analysis = df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"]).round(3)
sex_analysis.columns = ["总人数", "生还人数", "生还率"]

#三、舱位与生还的关系
print("三、舱位与生还的关系")
#1. 三个等级的舱位生还的各有多少人
print("三个等级的舱位生还的各有:",df.groupby("Pclass")["Survived"].sum())
print()

#2. 三个舱位的生还率：
print("生还率分别为： ",round(df.groupby("Pclass")["Survived"].mean(),3))
print()
print("----------------------------------------")
print()

class_analysis = df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"]).round(3)
class_analysis.columns = ["总人数", "生还人数", "生还率"]


#四、年龄与生还率的关系
print("四、年龄与生还率的关系")
#1. 查看年龄的基本分布
print("年龄的基本分布为：",df["Age"].describe())
print()

#2. 年龄分箱处理: pd.cut(要分箱的数据, bins=区间边界, labels=区间名称)
""" 0–12：儿童
    13–18：青少年
    19–30：年轻人
    31–45：中年
    46–60：中老年
    61+：老年  """
df["AgeGroup"]=pd.cut(df["Age"],bins=[0,13,19,31,46,61,81],labels=["儿童","青少年","年轻人","中年","中老年","老年"])
print(df["AgeGroup"].value_counts())
print()

#3. 计算不同年龄段的生还率
print("不同年龄段的生还率分别为：",round(df.groupby("AgeGroup")["Survived"].mean(),3))
print()

#4. 计算不同年龄段的生还人数
print("不同年龄段的生还人数分别为：",df.groupby("AgeGroup")["Survived"].sum())
print()
print("----------------------------------------")
print()

age_analysis = df.groupby("AgeGroup", observed=True)["Survived"].agg(
    ["count", "sum", "mean"]
).round(3)

age_analysis.columns = ["总人数", "生还人数", "生还率"]

#五. 性别 × 舱位 × 生还率
print("五. 性别 × 舱位 × 生还率")
#1. 计算生还率:
print("性别舱位与生还率的对应关系为: ",round(df.groupby(["Sex","Pclass"])["Survived"].mean(),3))
print()

#2. 查看每组性别 × 舱位对应的生还人数
print("性别与舱位对应的人数为: ",df.groupby(["Sex","Pclass"])["Survived"].sum())
print()

#3. agg聚合用法
# count: 非空值数量(即总人数)
# sum: 生还人数             mean: 生还率
print(df.groupby(["Sex","Pclass"])["Survived"].agg(["count","sum","mean"]).round(3))

sex_class_analysis = df.groupby(["Sex", "Pclass"])["Survived"].agg(
    ["count", "sum", "mean"]
).round(3)

sex_class_analysis.columns = ["总人数", "生还人数", "生还率"]

#六. 保存eda分析结果
sex_analysis.to_csv("sex_survival_analysis.csv", encoding="utf-8-sig")
class_analysis.to_csv("pclass_survival_analysis.csv", encoding="utf-8-sig")
age_analysis.to_csv("agegroup_survival_analysis.csv", encoding="utf-8-sig")
sex_class_analysis.to_csv("sex_pclass_survival_analysis.csv", encoding="utf-8-sig")