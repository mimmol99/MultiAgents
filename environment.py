import gym

class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state = self.env.reset()
        self.done = False

    def step(self, action):
        self.state, reward, self.done, _ = self.env.step(action)
        return self.state, reward, self.done

