import gym
import numpy as np

class PSOCartPoleSolver:
    def __init__(self, num_particles=50, max_steps=500):
        self.env = gym.make('CartPole-v1')
        self.num_particles = num_particles
        self.max_steps = max_steps
        
        self.w = 0.729
        self.c1 = 1.49445 
        self.c2 = 1.49445
        
        self.dim = 10
        self.global_best = None
        self.global_best_score = -np.inf
        
        self.particles = np.random.uniform(-1, 1, 
                                         (num_particles, self.dim))
        self.velocities = np.zeros((num_particles, self.dim))
        self.personal_bests = self.particles.copy()
        self.personal_scores = np.full(num_particles, -np.inf)

        self.global_best_score_list = []
        self.mean_scores = []
        
    def get_action(self, params, state):
        """ 使用线性策略选择动作 """
        W = params[:8].reshape(4, 2)
        b = params[8:]
        q_values = np.dot(state, W) + b
        return np.argmax(q_values)
    
    def evaluate(self, params):
        """ 评估参数性能 """
        total_reward = 0
        state = self.env.reset()
        for _ in range(self.max_steps):
            action = self.get_action(params, state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        return total_reward
    
    def optimize(self, generations=100):
        for gen in range(generations):
            scores = np.zeros(self.num_particles)
            
            for i in range(self.num_particles):
                scores[i] = self.evaluate(self.particles[i])

                if scores[i] > self.personal_scores[i]:
                    self.personal_scores[i] = scores[i]
                    self.personal_bests[i] = self.particles[i].copy()
                    
                if scores[i] > self.global_best_score and self.global_best_score != 500:
                    self.global_best_score = scores[i]
                    self.global_best = self.particles[i].copy()
            
            print(f"Gen {gen+1}, Best: {self.global_best_score}, Avg: {np.mean(scores):.1f}")
            self.global_best_score_list.append(self.global_best_score)
            self.mean_scores.append(np.mean(scores))
            
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            self.velocities = (self.w * self.velocities +
                             self.c1 * r1 * (self.personal_bests - self.particles) +
                             self.c2 * r2 * (self.global_best - self.particles))
            
            self.particles += self.velocities
            self.particles = np.clip(self.particles, -5, 5)
            
        self.env.close()
        np.save('best_params.npy', self.global_best)
        return self.global_best
    
    def plot_results(self):
        import matplotlib.pyplot as plt
        best_scores = np.array(self.global_best_score_list)
        mean_scores = np.array(self.mean_scores)
        len_scores = len(best_scores)
        plt.plot(range(len_scores), best_scores, label='Best Score')
        plt.plot(range(len_scores), mean_scores, label='Mean Score')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('PSO CartPole Performance')
        plt.legend()
        plt.grid()
        plt.savefig('pso_cartpole_performance.png')


if __name__ == "__main__":
    num_particles = 30
    generations = 50
    solver = PSOCartPoleSolver(num_particles=num_particles)
    best_params = solver.optimize(generations=generations)
    solver.plot_results()