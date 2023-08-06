import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PPO import PPO_agent
from Env import Env
from Actor_Critic_LSTM_NN import ActorNetwork, CriticNetwork, LSTM_model
import time




# Change the assets limited to (4x5) matrix
ASSETS = {
    0   : ["7203.T", "9984.T", "6758.T", "6861.T", "9983.T"],
    1   : ["BRK-A", "JNJ", "PG", "V", "JPM"],
    2   : ["AAPL", "AMZN", "MSFT", "GOOGL", "NVDA"],
    3   : ["ULVR.L", "HSBA.L", "BATS.L", "DGE.L", "BP.L"]
}

Markets_DIC = {0: "TSE", 1:"NYSE", 2:"NASDAQ", 3:"LSE"}

def main():
    env = Env(ASSETS, LSTM_model(5,300,2,1))
    N_values = [20]
    batch_size_values = [5]
    epochs_values = [4]
    lr_values = [0.0003]
    games_values = [300]
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
                        action_dim = env.action_space
                        space_dim = env.action_space
                        Actor = ActorNetwork(space_dim, action_dim)
                        actor_opt = torch.optim.Adam(Actor.parameters(), lr)
                        Critic = CriticNetwork(space_dim)
                        critic_opt = torch.optim.Adam(Critic.parameters(), lr)


                        PPO = PPO_agent(Actor, Critic, epochs, actor_opt, critic_opt)
                        score_hist = []
                        iters = 0
                        best_score = 0
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
if __name__ == "__main__":
    main()