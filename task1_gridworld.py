import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import os 
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):


        self.action_space = action_space # action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

        ## # TODO:

        # idea: pad playing field such that states alongside the edges exist 
        # GridWorlds won't actually go to these states, but training is easier that way 
        self.real_h = h
        self.real_w = w 
        self.q_vals = np.zeros((h+2,w+2,action_space)) # init weights, make sure q values from terminals are 0  

        self.alpha = 0.2
        self.gamma = 0.85
        self.reward_position_padded = (1,4) # real position (0,3)
        self.block_position_padded = (2,2) # real position (1,1)

    def __call__(self, state):

        # state has shape [[height_pos width_pos]]
        # print("call with state: " + str(state))

        ## # TODO:
        output = {}
        #output["q_values"] = np.random.normal(size=(1, self.action_space))
        #print(np.shape(output["q_values"]))

        # Full algo 
        # Loop for each episode:
            # Initialize S 
            # loop for each step of episode 
                # choose A from S using policy derived from Q (e.g. epsilon-greedy)
                # take action A, observe R,S' 
                # Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
                # S <- S' 
            # until S is terminal 

        # get initial q value in current state 
        # offset +1 because of padding around the playing field edges
        initial_q = self.q_vals[int(state[0][0]) + 1][int(state[0][1]) + 1][:]

        # Init updated q value with size of action space  
        updated_q = [None]*self.action_space

        # Update q value for each action 
        for action in range(self.action_space):
         
            # reset start state for each action
            new_state = state 

            # perform action, state has values height, width 
            if action == 0:
                new_state[0][0] += 1
            elif action == 1:
                new_state[0][1] += 1
            elif action == 2:
                new_state[0][0] -= 1
            elif action == 3: 
                new_state[0][1] -= 1   
            

            # environment takes care of not walking into walls
            # additionally, due to padding q values can be accessed just as usual 
            '''
            # clamp new state 
            if new_state[0][0] < 0:
                new_state[0][0] = 0 
            if new_state[0][1] < 0:
                new_state[0][1] = 0
            if new_state[0][0] >= self.h:
                new_state[0][0] = self.h - 1
            if new_state[0][1] >= self.w:
                new_state[0][1] = self.w - 1   
            '''

        
            
            # reward only when target point is hit 
            reward = 0
            if (new_state[0] == np.array(self.reward_position_padded)).all():
                print("Debug: next state reached goal")
                reward = 100
            
            # environment takes care of not walking into block 
            '''
            # big negative reward for blocking block 
            if (new_state[0] == np.array(self.block_position_padded)).all():
                print("Debug: next state hit block")
                reward = -9001 # it's over 9000! 
            '''


            # update q value  
            updated_q[action] = initial_q[action] + self.alpha * (reward + self.gamma * np.max(self.q_vals[int(new_state[0][0]) + 1][int(new_state[0][1]) + 1][:]) - initial_q[action])
            print(updated_q)


        # save update q_vals 
        self.q_vals[int(state[0][0])][int(state[0][1])][:] = updated_q
        #q_vals_copy = np.array(self.q_vals,copy=True)
        #q_vals_copy[int(state[0][0]) + 1][int(state[0][1]) + 1][:] = updated_q
        #self.set_weights(q_vals_copy) 

        # return updated q 
        output["q_values"] = np.reshape(updated_q, (1,4))  # np.reshape(self.q_vals[int(state[0][0]) + 1][int(state[0][1]) + 1][:], (1,4))
        
        ##### output["action"] = action # used in agent.py  
        
        #print("call returns: " +  str(output["q_values"]))
    
        return output


    # # TODO:
    def get_weights(self):
        return self.q_vals

    def set_weights(self, q_vals):
        self.q_vals = q_vals

    def save(self, path):
        print(path)
        np.savetxt(path + "_1.txt",self.q_vals[:][:][1])
        print("save dummy")

    def set_reward_position(self,pos):
        self.reward_position = pos 

    # what else do you need?


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
        "block_position": (1,1), # added default 
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
        # and more

        # probably not all needed 
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    '''
    print("test before training: ")
    manager.test(
        max_steps=10, #100,
        test_episodes=1, #10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    '''
    

    # do the rest!!!!

    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    '''
    buffer_size = 5000
    test_steps = 1000
    sample_size = 1000
    optim_batch_size = 8
    
    '''
    episodes = 20
    saving_after = 5
    max_steps = 20 



    # get initial agent
    agent = manager.get_agent()



    for e in range(episodes):

        # training core

        
        state_new = np.expand_dims(env.reset(), axis=0)
        #if return_reward:
        #    reward_per_episode = []

        for t in range(max_steps):
            if True: #render:
                env.render()
            state = state_new
            action = agent.act(state)
            # check if action is tf
            #if tf.is_tensor(action):
            #    action = action.numpy()
            if True: #self.kwargs['discrete_env']:
                action = int(action)
            state_new, reward, done, info = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)
            #if return_reward:
            #    reward_per_episode.append(reward)
            if done:
                #if return_time:
                #    time_steps.append(t)
                #if return_reward:
                #    rewards.append(np.mean(reward_per_episode))
                break
            if t == max_steps - 1:
                #if return_time:
                #    time_steps.append(t)
                #if return_reward:
                #    rewards.append(np.mean(reward_per_episode))
                break


        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    