# -*- coding: utf-8 -*-
import random
from collections import deque
import gym
import numpy as np
from keras.layers import Dense, LSTM, Input, Add, Embedding
from keras.models import Model

EPISODES = 1000


class DQNAgent:
    def __init__(self, _state_size, _action_size, input_dim, output_dim):
        self.state_size = _state_size
        self.action_size = _action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.session = [1, 0]  # first stands for Organic, second stands for Session

    def _build_model(self):

        # Build embedding layers
        input1 = Input(batch_shape=(None, None, self.input_dim))  # sample size, timestamps, dim
        embedded_layer = Embedding(output_dim=25)(input1)

        # Build Organic LSTM. Two layers
        Organic_lstm_layer1, self.organic_hidden1, self.organic_cell1 = LSTM(25, stateful=True, return_state=True,
                                                                             return_sequences=True)(embedded_layer)
        Organic_lstm_layer2, self.organic_hidden2, self.organic_cell2 = LSTM(self.output_dim, stateful=True,
                                                                             return_sequences=True,
                                                                             return_state=True)(Organic_lstm_layer1)

        # Build Bandit LSTM. Two layers
        Bandit_lstm_layer1, self.bandit_hidden1, self.bandit_cell1 = LSTM(25, stateful=True, return_sequences=True,
                                                                          return_state=True)(embedded_layer)
        Bandit_lstm_layer2, self.bandit_hidden2, self.bandit_cell2 = LSTM(self.output_dim, stateful=True,
                                                                          return_sequences=True,
                                                                          return_state=True)(Bandit_lstm_layer1)
        Total_layer = Add()([Bandit_lstm_layer2 * self.session[1], Organic_lstm_layer2 * self.session[0]])

        # Neural Net for Deep-Q learning Model
        DQN_layer1 = Dense(24, activation='relu')(Total_layer)
        DQN_layer2 = Dense(24, activation='relu')(DQN_layer1)
        DQN_layer3 = Dense(self.action_size, activation='linear')(DQN_layer2)
        model = Model(input1, DQN_layer3)
        model.compile(optimizer='adam', loss='mse')
        return model

    def Organic_2_Bandit(self):
        self.model.layers[1].state[0] = self.organic_hidden1
        self.model.layers[1].state[1] = self.organic_cell1
        self.model.layers[2].state[0] = self.organic_hidden2
        self.model.layers[2].state[1] = self.organic_cell2

    def Bandit_2_Organic(self):
        self.model.layers[1].state[0] = self.bandit_hidden1
        self.model.layers[1].state[1] = self.bandit_cell1
        self.model.layers[2].state[0] = self.bandit_hidden2
        self.model.layers[2].state[1] = self.bandit_cell2

    def remember(self, _state, _action, _reward, _next_state, _done):
        self.memory.append((_state, _action, _reward, _next_state, _done))

    def act(self, _state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(_state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, _batch_size):  # remove state and next state
        minibatch = random.sample(self.memory, _batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
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
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
