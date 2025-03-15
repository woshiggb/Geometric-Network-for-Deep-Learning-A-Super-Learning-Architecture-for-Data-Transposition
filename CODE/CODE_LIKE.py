

import numpy as np
import matplotlib.pyplot as plt

# 激活函数（sigmoid）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 激活函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

def f(x): return (x+1)*np.log10(x+1)**2 - np.tanh(x)


# 生成数据集
def generate_data(num_samples):
    x = np.random.rand(num_samples, 1)  # 生成 0 到 1 之间的随机数
    y = f(x)    # 目标函数
    return x, y

# 定义神经网络结构
class NeuralNetwork:
    def __init__(self):
        # 初始化权重
        self.weights_input_hidden1 = np.random.rand(1, 2)
        self.weights_hidden1_hidden2 = np.random.rand(2, 3)
        self.weights_hidden2_output = np.random.rand(3, 1)

    def forward(self, x):
        # 前向传播
        self.hidden1 = sigmoid(np.dot(x, self.weights_input_hidden1))
        self.hidden2 = sigmoid(np.dot(self.hidden1, self.weights_hidden1_hidden2))
        self.output = sigmoid(np.dot(self.hidden2, self.weights_hidden2_output))
        return self.output

    def backward(self, x, y, learning_rate):
        # 反向传播
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden2_error = output_delta.dot(self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden2)

        hidden1_error = hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden1)

        # 更新权重
        self.weights_hidden2_output += self.hidden2.T.dot(output_delta) * learning_rate
        self.weights_hidden1_hidden2 += self.hidden1.T.dot(hidden2_delta) * learning_rate
        self.weights_input_hidden1 += x.T.dot(hidden1_delta) * learning_rate

# 主程序
num_iterations = 100
num_samples = 100
num_test_points = 10000
learning_rate = 0.1

# 记录误差
errors = []
errors2 = []

import code__


# 训练模型
nn = NeuralNetwork()
nn2 = code__.Train2([20, [[1, 2,2,2, 1], 2], 100])
xs , ys = [], []
x_train, y_train = generate_data(num_samples)
xs , ys = x_train , y_train
some_errors = []

for _ in range(10):
    x_train, y_train = generate_data(num_samples)
    #xs += list(x_train) 
    #ys += list(y_train)
    #print(xs , ys)
    nn2.train_one(x_train,y_train, 1, 1)
    nn.forward(x_train) 
    nn.backward(x_train, y_train, learning_rate)
    for _ in range(len(xs)):  # 每个模型训练 100 次
        nn2.model[0].mdn[0].forward(xs[_])
        nn2.model[0].mdn[0].backward(ys[_])
    nn2.train_one  
    # 测试模型
    x_test = [[i*0.01] for i in range(100)]
    y_test = [[f(x_test[j][0])] for j in range(100)]
    y_pred = np.array([nn.forward(x_test[i]) for i in range(len(x_test))])
    num , b = [0 for i in range(len(x_test[0]))] , 0
    #nn2.model[0].m  dn[0].forward([0])
    num = sum([abs(nn2.model[0].mdn[0].forward(i)[0] - f(i[0]))**2 for i in x_test]) / len(x_test)
    #for i in x_test:
    	#print(i , 1)
    	#print(nn2.model[0].mdn[0].sekf_outs)
    #	num =[b*num[j]/(b+1)+abs(nn2.model[0].mdn[0].forward(i)[0] - f(i[0]))/(b+1) for j in range(len(x_test[0]))]
    #	b+=1
    some_errors = [sum([abs(i.forward(x_test[j])[0] - f(x_test[j][0])) for j in range(len(x_test))]) for i in nn2.model[0].mdn]
    mse2 = np.mean(np.array(num)**2)

    # 计算均方误差
    mse = np.mean((y_test - y_pred) ** 2)
    errors.append(code__.math.log10(mse))
    #breaks
    errors2.append(code__.math.log10(mse2))

# 绘制函数图
x_plot = np.linspace(0, 1, num_test_points)
y_plot = [f(x_plot[i]) for i in range(len(x_plot))]  
print(len(errors))
plt.figure(figsize=(10, 6))
plt.plot([i*100 for i in range(10)], errors, label='MLP_train', color='blue')
plt.plot([i*100 for i in range(10)], errors2, label='Single-model training', color='red', alpha=0.7)
plt.title('Neural Network Approximation of log x ** 2 - tanh(x))')
plt.xlabel('x')
plt.legend()
plt.grid()
plt.show()

# 输出误差信息
print(f'平均均方误差: {np.mean(errors)}')
