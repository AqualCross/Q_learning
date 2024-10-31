import numpy as np
import random
import os

class MyEnv:
    def __init__(self):
        # 动作空间: 上下左右
        self.action_space_num = 4

        # 状态空间：6个房间
        self.state_space_num = 6

        # 初始化状态
        self.state = 2  # 开始时在2号房间

        # 目标状态（出口）
        self.goal_state = 5

        # 在当前状态执行一个动作后转移到下一个状态
        # 行标为当前状态，列标为动作，值为下一个状态
        self.transitions = [
            [0, 0, 0, 4],  # 0
            [1, 1, 5, 3],  # 1
            [3, 2, 2, 2],  # 2
            [4, 2, 1, 3],  # 3
            [4, 3, 0, 5],  # 4
        ]

        # 记录路径
        self.path = [self.state]

    def step(self, action):
        """在当前状态采取行动，得到奖励和下一状态"""
        next_state = self.transitions[self.state][action]

        reward = -1  # 每步都给予负奖励
        if next_state == self.goal_state:
            reward = 100  # 到达目标状态获得正奖励
            done = True
        else:
            done = False

        self.state = next_state
        self.path.append(next_state)
        return next_state, reward, done

    def reset(self):
        self.state = 2
        self.path = [self.state]
        return self.state

    # def render(self, path=None, log_dir=None):
    #     # 房间坐标,用于绘图
    #     room_positions = {
    #         0: (0, 1),
    #         1: (1, 1),
    #         2: (2, 0),
    #         3: (1, 0),
    #         4: (0, 0),
    #         5: (1, 2),
    #     }
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #
    #     # 绘制房间
    #     for room, pos in room_positions.items():
    #         color = 'grey' if room == self.goal_state else 'white'
    #         ax.add_patch(plt.Rectangle(pos, 1, 1, edgecolor='black', facecolor=color))
    #         ax.text(pos[0] + 0.5, pos[1] + 0.5, str(room), ha='center', va='center')
    #
    #     # 绘制路径
    #     if path is None:
    #         path = self.path
    #     for i in range(len(path) - 1):
    #         from_pos = room_positions[path[i]]
    #         to_pos = room_positions[path[i + 1]]
    #         ax.plot([from_pos[0] + 0.5, to_pos[0] + 0.5], [from_pos[1] + 0.5, to_pos[1] + 0.5], 'r-')
    #
    #     ax.set_xlim(0, 3)
    #     ax.set_ylim(0, 3)
    #     ax.set_aspect('equal')
    #     if log_dir is not None:
    #         # 保存图像
    #         file_path = os.path.join(log_dir, 'path.png')
    #         plt.savefig(file_path)
    #     else:
    #         # 显示图像
    #         plt.show()

    def render(self, path=None, log_dir=None):
        from PIL import Image, ImageDraw

        # 读取图片
        img_path = "asset/img.png"
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)

        # 定义房间的位置坐标（图片坐标拾取https://uutool.cn/img-coord/）
        room_positions = {
            0: (94,75),
            1: (232,100),
            2: (348,130),
            3: (209,162),
            4: (100,173),
            5: (293,42)  # Goal state
        }

        if path is None:
            path = self.path
        # 在图片上绘制路径
        for i in range(len(path) - 1):
            from_pos = room_positions[path[i]]
            to_pos = room_positions[path[i + 1]]
            draw.line([from_pos, to_pos], fill="red", width=3)

        if log_dir is not None:
            # 保存图像
            file_path = os.path.join(log_dir, 'result.png')
            image.save(fp=file_path, format='PNG')
        else:
            image.show()


class Q_learing:
    def __init__(self, seed):
        self.seed = seed    # 随机种子
        random.seed(self.seed)

        self.env = MyEnv()

        # Q-learning 参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.1  # 探索率

        self.q_table = np.zeros((6, 4))  # Q表
        self.best_path = None
        self.best_reward = -np.inf

    def choose_action(self, state, explore=True):
        """
        根据当前状态选择动作
        以epsilon的概率选择探索
        以1-epsilon的概率选择利用
        """
        if explore and random.uniform(0, 1) < self.epsilon:
            # 以epsilon的概率随机选择动作（探索）
            action = random.randint(0, self.env.action_space_num - 1)
        else:
            # 选择当前策略下的最佳动作（利用）
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        alpha = self.alpha
        gamma = self.gamma
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_table[state, action] = new_value

    def play(self, explore=True):
        """玩一局"""
        state = self.env.reset()
        total_reward = 0    # 就是有限马尔可夫决策过程的return但是return是python的关键字
        while True:
            action = self.choose_action(state, explore)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            self.update_q_table(state, action, reward, next_state)
            state = next_state
            if done:
                return total_reward, self.env.path

    def train(self, episodes: int):
        for episode in range(episodes):
            reward, path = self.play()
            if reward > self.best_reward:
                self.best_path = path
                self.best_reward = reward

            # print('reward:', reward)
            # print('path:', path)
            # if episode % 20 == 0:
            #     self.env.render()

    def show(self):
        """保存结果"""
        log_dir = f'log/{self.seed}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建或打开一个日志文件
        log_file_path = os.path.join(log_dir, 'output.log')
        with open(log_file_path, 'w') as log_file:
            # 重定向标准输出
            import sys
            original_stdout = sys.stdout
            sys.stdout = log_file

            print('q_table:')
            print(self.q_table)
            reward, path = self.play(explore=False)
            print('reward:', reward)

            # 恢复标准输出
            sys.stdout = original_stdout
        self.env.render(path, log_dir)


if __name__ == '__main__':
    q_learing = Q_learing(0)
    q_learing.train(10)
    q_learing.show()
    q_learing = Q_learing(9)
    q_learing.train(10)
    q_learing.show()
