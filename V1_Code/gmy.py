import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  
import os
from MLProject.agent import Agent
from simtech import restart, next
import matplotlib.pyplot as plt 


write_data = False
def write_array_to_file(array, filename):
    if write_data:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                for item in array:
                    f.write(str(item))
                    f.write("\n")

            #os.chmod(filename, stat.S_IREAD)  # Change file permissions to read-only

    else:
        print(f"File {filename} already exists. No action taken.")


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    
    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)




if __name__ == "__main__":
    
    write_data = True ##HYPERPARAMETERS
    agent = Agent(gamma=1, epsilon=10.0,
                  batch_size=128, n_actions=5, eps_end=0.01,
                  input_dims=[5], lr=0.003)
    scores, eps_history = [], []
    n_games = 10000

    for i in range(n_games):
        
        time = 0 
        score = 0
        done = False
        #observation = env.reset()
        observation = restart()
        print("obs:")
        print(observation)
        cords = []

        while not done:
            action = agent.choose_action(observation) # <---
            #observation_, reward, done, info = env.step(action)
            observation_, reward, done, info = next(action)
            score += reward 
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            x, y = observation[0], observation[1]
            cords.append((x, y))





            observation = observation_

            if time > 1000:
                print("TIME OUT")
                done = True
            time += 1
            #print("______________")
        

        if done and info: 
                write_array_to_file(cords, f"./funcionals/cords{i}.txt")


        scores.append(score)
        eps_history.append(agent.epsilon)
        print("----")
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % np.mean(scores[-1000:]),
              'epsilon %.2f' % agent.epsilon,)#
        print("----")
    # x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, scores, eps_history, "./testNOW3.png")