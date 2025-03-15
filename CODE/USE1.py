import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import code__
import time  # 导入time模块以便使用sleep

# 初始化数据和参数
player_points = []
drawing = False  # 标记是否正在绘制
player_line = None

# 创建画布
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True)

plt.title("Interactive Function Drawing")

# 提交按钮
submit_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
submit_button = Button(submit_ax, 'Submit')

# 模拟的函数f，通过传入的点训练和生成输出
class FunctionModel:
    def __init__(self):
        self.points = []
    
    def train(self, points):
        self.points = points
    
    def output(self, pg):
        if pg == 1:
            return self.points
        elif pg == 0:
            return None

# 开始绘制
def on_press(event):
    global drawing
    if event.inaxes == ax:
        drawing = True

# 更新绘制
def on_motion(event):
    global player_line
    if event.inaxes == ax and drawing:
        x, y = event.xdata, event.ydata
        player_points.append((x, y))
        x_data, y_data = zip(*player_points)  # 解压为x和y坐标列表
        if player_line is None:
            player_line, = ax.plot(x_data, y_data, 'ro')  # 创建新的Line2D对象
        else:
            player_line.set_xdata(x_data)
            player_line.set_ydata(y_data)
        ax.figure.canvas.draw()

# 停止绘制
def on_release(event):
    global drawing
    drawing = False

# 创建模型
model = code__.Train2([10, [[1, 2,2,2,2,5, 1], 1], 10])

# 提交按钮的事件处理
def on_submit(event):
    global player_points, player_line
    if player_points:
        cin, out = [[i[0]] for i in player_points], [[j[1]] for j in player_points]
        #model.train_one(cin, out, 10, 1)
        #model.train_one(cin , out , 1 , 1)

        # 逐点进行前向传播和反向传播
        for i in range(len(cin)):
            for j in model.model[0].mdn:
                j.forward(cin[i])
                j.backward(out[i])
                
                # 更新图像
                output_points = [[0.01 * k, model.model[0].mdn[0].forward([0.01 * k])[0]] for k in range(100)]
                if output_points:
                    ax.clear()  # 清除当前图像
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.grid(True)
                    ax.set_title("Interactive Function Drawing")
                    x_data, y_data = zip(*output_points)
                    ax.plot(x_data, y_data, 'b-')  # 绘制新的函数图像
                    ax.plot(*zip(*player_points), 'ro')  # 绘制用户的点
                    plt.pause(0.01)  # 暂停1秒
                    
        ax.figure.canvas.draw()

# 绑定事件
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
submit_button.on_clicked(on_submit)

plt.show()
