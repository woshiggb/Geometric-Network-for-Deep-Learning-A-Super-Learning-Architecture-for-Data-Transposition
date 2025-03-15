from code__ import *
import matplotlib.pyplot as plt
import matplotlib.colors as ppopo
import numpy as np
import sympy as Stod
import joblib

# 定义函数 f(x) = x^2
Mm = modelkl(N=[1, 10, 10, 1], rand_=10)
from sympy import symbols, sympify, lambdify

x = symbols('x')
user_input = input("Please enter a function (like, x**2 + 1 - x): 'x**2 + 2*x + 1'(You can say Help): ")
expr = sympify(user_input)  # 转换为 sympy 表达式
f = lambdify(x, expr)  # 转换为函数
num_lts = 1 / 100
cinS = [[0]] + [[num_lts * i] for i in range(1, 101)]
outS = [[f(j) for j in cinS[i]] for i in range(len(cinS))]

# TRAIN_MODELS = Train2([100 , [[1 , 10 , 10 , 10 , 10 , 1] , 20]] , 10)
TRAIN_MODELS = joblib.load("model.joblib")
MODELS = TRAIN_MODELS.model[0]
Ns_f1 = 1
timre_num = 100
for s in range(len(MODELS.mdn)):
    for i in range(timre_num):
        for j in range(Ns_f1):
            MODELS.mdn[s].forward(cinS[i])
            MODELS.mdn[s].backward(outS[i])


joblib.dump(TRAIN_MODELS, './model.joblib')
TRAIN_MODELS = joblib.load("model.joblib")
print("Save is Over.")

while True:
    IN = input("Cin x(Like:0.1,0.2,0.35) :")
    print(TRAIN_MODELS.model[0].mdn[0].forward([eval(IN)]))

mts = list(ppopo.CSS4_COLORS.keys())

get_num = 1
def g(x):
    return [[MODELS.mdn[isn].forward([y], cnt=0)[0] for y in x] for isn in range(min(MODELS.N1, len(mts)))]

# 生成 x 值
nus = 1 / 10000
x = [i * nus for i in range(0, 10001)]

# 计算 y 值
y_f = [f(x_i) for x_i in x]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y_f, label='f(x) = F1(x)', color='blue')

goo = g(x)
for i in range(len(goo)):
    print(sum([abs(goo[i][j] - y_f[j]) for j in range(len(goo[i]))]) / len(goo[i]))

for i in range(1):
    plt.plot(x, goo[i], label="""\/ """, color="red")

# 添加标题和标签
plt.title('Plot of f = F1 and g = f2')   
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

# 保存图像到文件
plt.savefig('plot.png')  # 保存为 'plot.png' 文件
plt.close()  # 关闭图形，避免显示

print("图像已保存为 'plot.png'")
