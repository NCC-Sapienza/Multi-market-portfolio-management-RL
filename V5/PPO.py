import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import gym
import time
from Actor_Critic_LSTM_NN import ActorNetwork, CriticNetwork


class PPO_buffer:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        idx = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(idx)
        batches = [idx[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions),\
            np.array(self.probs), np.array(self.vals), np.array(self.rewards),\
            np.array(self.dones), batches

    def store_memory(self,state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []




class PPO_agent:
    def __init__(self, Actor, Critic, epochs, actor_opt, critic_opt, gamma = 0.99, lr = 0.0003, batch_size = 64, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.KLDLOSS = nn.KLDivLoss()
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = PPO_buffer(batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Actor = Actor.to(self.device)
        self.Critic = Critic.to(self.device)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.Actor.save_dict()
        self.Critic.save_dict()

    def load_models(self):
        print("... loading models ...")
        self.Actor.load_dict()
        self.Critic.load_dict()

    def choose_action(self, obs):
        state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        dist = self.Actor(state)
        value = self.Critic(state)
        action = dist
        print(action)

        probs = torch.squeeze(torch.log(action)).cpu().detach().numpy()
        action = torch.squeeze(action).cpu().detach().numpy()
        value = torch.squeeze(value).item()

        return action, probs, value

    def step(self):

        for _ in range(self.epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            values = vals_arr
            adv = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 0.99
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * \
                          (1- int(done_arr[k])) - values[k])
                adv[t] = a_t
            adv = torch.tensor(adv).to(self.device)

            values = torch.tensor(values).to(self.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist = self.Actor(states)
                critic_value = self.Critic(states).squeeze()

                new_probs = torch.log(dist)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weight = adv[batch] * prob_ratio
                kdl_loss = self.KLDLOSS(new_probs, old_probs)  #check the dim


                actor_loss = -torch.min(weight, kdl_loss).mean()

                returns = adv[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()
        self.memory.clear_memory()

"""
env = gym.make("CartPole-v0")
N_values = [20, 30, 50]
batch_size_values = [5, 10, 20]
epochs_values = [4, 10, 20]
lr_values = [0.0003, 0.0001]
games_values = [400, 600, 800]
results = {}
cont = 1
# iterate over all combinations of hyperparameters
for N in N_values:
    for batch_size in batch_size_values:
        for epochs in epochs_values:
            for lr in lr_values:
                for games in games_values:
                    s = time.time()
                    key = (N, batch_size, epochs, lr, games)
                    action_dim = env.action_space.n
                    space_dim = env.observation_space.shape[0]
                    Actor = ActorNetwork(space_dim, action_dim)
                    actor_opt = torch.optim.Adam(Actor.parameters(), lr)
                    Critic = CriticNetwork(space_dim)
                    critic_opt = torch.optim.Adam(Critic.parameters(), lr)


                    PPO = PPO_agent(Actor, Critic, epochs, actor_opt, critic_opt)
                    score_hist = []
                    iters = 0
                    best_score = env.reward_range[0]
                    avg_score = 0
                    n_steps = 0

                    for i in range(games):

                        obs = env.reset()
                        done = False
                        score = 0
                        while not done:
                            action, prob, val = PPO.choose_action(obs)
                            obs_ , reward, done, info = env.step(action)
                            n_steps += 1
                            score += reward
                            PPO.remember(obs, action, prob, val, reward, done)
                            if n_steps % N == 0:
                                PPO.step()
                                iters += 1
                            obs = obs_
                        score_hist.append(score)
                        avg_score = np.mean(score_hist[-100:])

                        if avg_score > best_score:
                            best_score = avg_score
                            PPO.save_models()

                        print("episode ", i, "score %.1f" % score, "avg score %.1f" % avg_score,
                              "time steps", n_steps, "learning steps", iters)
                    print("tempo impiegato: ", (time.time() - s)/60)
                    print(f"iter : {cont}/162")
                    cont += 1
                    results[key] = avg_score

results = pd.DataFrame(results)
results.to_excel("ris_grid_PPO.xlsx")


"""















