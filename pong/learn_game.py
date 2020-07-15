from game_env import GameEnv
import tensorflow as tf
import numpy as np
import random
from collections import deque


class GameLearner:
    def __init__(self):
        self.game_env = GameEnv()
        self.max_memory_size = 50000
        self.memory = deque(maxlen=self.max_memory_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

        self.new_model = self.create_model()
        self.model = self.create_model()


    def create_model(self):
        model = tf.keras.Sequential()

        dim_states = len(self.game_env.state)           # state dimension
        num_action = self.game_env.num_actions         # number of possible actions

        model.add(tf.keras.layers.Dense(200, input_dim=dim_states, activation="relu"))
        model.add(tf.keras.layers.Dense(100, activation="relu"))
        model.add(tf.keras.layers.Dense(50, activation="relu"))
        model.add(tf.keras.layers.Dense(num_action, activation="linear"))
        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        return model

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.game_env.num_actions)
        else:
            state = np.reshape(state, (1, len(state)))
            y = self.model.predict(state)[0]
            action = np.argmax(y)

        return action

    def remember(self, state, action, reward, new_state, terminal):

        # if memory is full, delete the oldest data
        if len(self.memory) >= self.max_memory_size:
            self.memory.popleft()

        self.memory.append([np.array(state), action, reward, np.array(new_state), terminal])

    def replay(self):
        min_batch_size = 32

        if len(self.memory) >= min_batch_size:

            batch_size = max(min_batch_size, int(len(self.memory) * 0.2))
            #batch_size = min_batch_size

            samples = random.sample(self.memory, batch_size)

            samples = np.array(samples)

            #states = samples[:, 0]
            actions = samples[:, 1]
            rewards = samples[:, 2]
            #new_states = samples[:, 3]
            terminals = samples[:, 4]

            dim_state = len(samples[0, 0])

            states = np.zeros((min_batch_size, dim_state))
            new_states = np.zeros((min_batch_size, dim_state))

            for i in range(0, min_batch_size):
                for j in range(0, dim_state):
                    states[i, j] = samples[:, 0][i][j]
                    new_states[i, j] = samples[:, 3][i][j]

            targets = self.model.predict(states)
            q_futures = np.max(self.model.predict(new_states), 1)

            for i in range(0, min_batch_size):
                a = actions[i]

                if terminals[i]:
                    targets[i][a] = rewards[i]
                else:
                    targets[i][a] = rewards[i] + (q_futures[i] * self.gamma)

            self.new_model.fit(states, targets, epochs=1, verbose=0)

    def update_model(self):
        new_weights = self.new_model.get_weights()
        weights = self.model.get_weights()
        for i in range(len(weights)):

            if len(weights[i].shape) == 2:
                for r in range(weights[i].shape[0]):
                    for c in range(weights[i].shape[1]):
                        weights[i][r][c] = new_weights[i][r][c]
            else:
                for r in range(weights[i].shape[0]):
                    weights[i][r] = new_weights[i][r]

        self.model.set_weights(weights)

    def save_model(self, fn):
        self.model.save(fn)
        self.model.save_weights(fn + ".w")

if __name__ == "__main__":
    learner = GameLearner()

    trials = 1000
    trial_len = learner.game_env.max_steps
    steps = []

    for trial in range(trials):
        learner.game_env.init()
        learner.decay_epsilon()

        cur_state = np.copy(learner.game_env.state)

        print("Trial {0}, eps {1}".format(trial, learner.epsilon))

        sum_reward = 0

        for step in range(trial_len):
            action = learner.act(cur_state)

            learner.game_env.act(action)
            learner.game_env.step()

            new_state = np.copy(learner.game_env.state)
            reward = learner.game_env.reward

            sum_reward += reward

            if learner.game_env.terminal or random.randint(0, 10) < 3:
                learner.remember(cur_state, action, reward, new_state, learner.game_env.terminal)

            learner.replay()
            learner.update_model()

            cur_state = np.copy(new_state)
            if learner.game_env.game_clear == 1 or learner.game_env.game_over == 1:
                print("\r{0} / {1}".format(step, trial_len), end='')
                print(" memory size: {0} ".format(len(learner.memory)))
                break

        if learner.game_env.game_over == 1:
            print(" Failed to complete in trial {}, r:{}".format(trial, (sum_reward / step)))
            if trial % 10 == 0:
                learner.save_model("models/trial-{}.model".format(trial))
        else:
            print(" Complted in {} trials".format(trial))
            learner.save_model("models/success.model")
            break


            





