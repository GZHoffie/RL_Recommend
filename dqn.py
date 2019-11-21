# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from keras.layers import Dense, LSTM, Input, Embedding
from keras.models import Model
from recogym import Configuration, env_1_args
from recogym.agents import Agent

EPISODES = 1000


class DQNAgent(Agent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)

        self.input_dim = self.config.num_products
        self.action_size = self.config.num_products
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.session = [1, 0]  # first stands for Organic, second stands for Session

        self.action0 = np.zeros((self.action_size + 2))

    def _build_model(self):
        # Build embedding layers
        input1 = Input(batch_shape=(None, None, self.input_dim + 2))  # sample size, timestamps, dim
        embedded_layer = Embedding(input_dim=self.input_dim+2, output_dim=25)(input1)

        # Build LSTM. Two layers        
        Bandit_lstm_layer1 = LSTM(25, return_sequences=True)(embedded_layer)
        Bandit_lstm_layer2 = LSTM(10)(Bandit_lstm_layer1)
        temp_layer = Bandit_lstm_layer1

        #  Neural Net for Deep-Q learning Model
        DQN_layer1 = Dense(24, activation='relu')(Bandit_lstm_layer2)
        DQN_layer2 = Dense(24, activation='relu')(DQN_layer1)
        DQN_layer3 = Dense(self.action_size, activation='linear')(DQN_layer2)
        model = Model(input1, DQN_layer3)
        model.compile(optimizer='adam', loss='mse')
        return model


def remember(self, state, _action, _reward, next_state, _done):
    self.memory.append((state, _action, _reward, next_state, _done))


def memory_reset(self):
    self.memory = []


def act(self, _a, _mode, _r):
    if np.random.rand() <= self.epsilon:
        _a = random.randrange(self.action_size)
    _action = self.action0
    _action[_a] = 1
    _action[-1] = _r
    _action[-2] = _mode
    q_values = self.model.predict(_action)
    return np.argmax(q_values[0]), q_values  # returns action


def load(self, name):
    self.model.load_weights(name)


def save(self, name):
    self.model.save_weights(name)


def replay(self, _action, _q_list, _mode, _reward):
    n = len(_action)
    q_target = []
    # q_target = [q_list[i] if mode[i] == 0 else (reward[i] + self.gamma * q_list[i + 1]) for i in range(n - 1)]
    for i in range(n - 1):
        if _mode[i] == 0:
            q_target.append(_q_list[i])
        else:
            _q_list[i, _action[i]] = _reward[i] + self.gamma * max(_q_list[i + 1])
            q_target.append(_q_list[i])

    action_r = np.zeros((n - 1, self.action_size))
    for i in range(n - 1):
        action_r[i, _action[i]] = 1

    input_r = np.concatenate(action_r[:-1, :], _mode[:-1], _reward[:-1])
    self.model.fit(input_r, q_target, batch_size=n - 1)


if __name__ == "__main__":
    # You can overwrite environment arguments here:
    env_1_args['random_seed'] = 42

    # Initialize the gym for the first time by calling .make() and .init_gym()
    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)
    env.reset_random_seed()

    # Change product number here
    num_products = 500
    agent = DQNAgent(Configuration({
        **env_1_args,
        'num_products': num_products,
    }))
    env.reset()
    num_offline_users = 100000
    num_clicks = 0
    num_events = 0

    for _ in range(num_offline_users):
        # Reset env and set done to False.
        env.reset()

        observation, _, done, _ = env.step(None)
        reward = None
        done = None
        q_list = []
        action = []
        mode = []
        reward = []
        a = 0
        q = []
        while not done:
            if observation:
                for item in observation.sessions():
                    action.append(item['v'])
                    reward.append(0)
                    mode.append(0)
                    a, q = agent.act(item['v'], 0, 0)
                    q_list.append(q)

            mode.append(1)
            action.append(a)
            observation, r, done, info = env.step(a)
            reward.append(r)
            a, q = agent.act(a, 1, r)
            q_list.append(q)

            num_clicks += 1 if reward == 1 and reward is not None else 0
            num_events += 1
        agent.replay(action, q_list, mode, reward)

        print(num_clicks, num_events)

    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent.load("./save/cartpole-dqn.h5")

    '''for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)'''
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")
