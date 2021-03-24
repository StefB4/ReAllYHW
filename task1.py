import gym
import numpy as np
import ray
import os
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space, alpha = 0.2, gamma = 0.85):
        self.action_space = action_space
        self.height = h
        self.width = w
        self.q_vals = np.zeros((self.action_space,self.height, self.width))
        self.alpha = alpha
        self.gamma = gamma
        pass

    def __call__(self, state):
        output = {}
        print("INPUT STATE: ", state)
        if type(state) is tuple:
            #print("Typeof coordTupel: ", type(state[0]), type(state[1]))
            logits = self.q_vals[:, state[0], state[1]]
            logits = np.reshape(logits, (1,4))
            output["q_values"] = logits
        else:
            #print("Type of coordArray: ", type(state[0][0]),type(state[0][1]))
            logits = self.q_vals[:, int(state[0][0]), int(state[0][1])]
            logits = np.reshape(logits, (1,4))
            output["q_values"] = logits
        return output

    def get_weights(self):
        return self.q_vals

    def set_weights(self, q_vals):
        self.q_vals = q_vals
        
    def adjustQ(self, trajectory, update):
        if len(trajectory) > 0:
            mostRecent = trajectory.pop()
            state, action, reward = mostRecent
                        
            updated = self.q_vals[action, int(state[0][0]), int(state[0][1])] + self.alpha * (reward + (self.gamma * update) - self.q_vals[action, int(state[0][0]), int(state[0][1])])
            self.q_vals[action, int(state[0][0]), int(state[0][1])] = updated
            adjustQ(self, trajectory, updated)
        pass
    # def save(full_path):
    #     mkdir(fullpath)
    #     cd fullpath
    #     mkdir variabels
    #     get qvals
    #     write qvals to file
        

if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        "env_kwargs" : env_kwargs,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": 1

        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "/progress_test"


    print("Testing before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    episodes = 40
    saving_after = 5
    max_steps = 20 
    
    
    from collections import deque

    agent = manager.get_agent()
    for e in range(episodes):
        
        #new_state = env.reset()
        new_state = np.expand_dims(env.reset(), axis=0)
        
        trajectory = deque()
        #Or while not done?
        for t in range(max_steps):
            env.render()
            
            #Calls the model, gets the q-values, chooses action
            action = agent.act(new_state)
            print("Action: ", action)
            print("Type: ", type(action))
            savedState = new_state
            new_state, reward, done, info = env.step(action[0])
            new_state = np.expand_dims(new_state, axis=0)

            
            if done:
                #Update the q-values
                TabularQ.adjustQ(trajectory, reward)
                print("Episode finished after {} timesteps".format(t+1))
                break
            else:
                trajectory.append((savedState, action, reward)) 
            
        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)
    
    env.close()
        
        


        