import gym
import numpy as np
import time

class CartPoleRenderer:
    def __init__(self, params_path='best_params.npy'):
        self.env = gym.make('CartPole-v1')
        
        try:
            self.params = np.load(params_path)
            if len(self.params) != 10:
                raise ValueError("Invalid parameter dimensions")
        except Exception as e:
            print(f"加载参数失败: {str(e)}")
            exit(1)
        
        print("参数加载成功，开始演示...")

    def get_action(self, state):
        """ 使用加载的参数决策 """
        W = self.params[:8].reshape(4, 2)
        b = self.params[8:]
        q_values = np.dot(state, W) + b
        return np.argmax(q_values)

    def run_demo(self):
        """ 运行并渲染演示 """
        state = self.env.reset()
        total_reward = 0
        
        for _ in range(500):
            self.env.render()
            
            action = self.get_action(state)
            
            next_state, reward, done, _ = self.env.step(action)
            
            total_reward += reward
            state = next_state
            
            time.sleep(0.02)
            
            if done:
                break
                
        print(f"总奖励: {total_reward}")
        print("演示结束，窗口将在5秒后关闭...")
        time.sleep(5)
        self.env.close()

if __name__ == "__main__":
    renderer = CartPoleRenderer()
    renderer.run_demo()