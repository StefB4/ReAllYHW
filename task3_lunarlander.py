### 
# Additional changes in lunar_lander.py: Line 250, comment out assert (https://github.com/openai/gym/issues/219)
# In agent.py: Line 124, output["action"] = list(action.flatten()) (otherwise lunar_lander.py breaks)
### 
      

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  
from really.utils import (
    dict_to_dict_of_datasets,
)  
from scipy.stats import norm


class DDPG(tf.keras.Model):
    def __init__(self, input_units=8, output_actions=2):

        super(DDPG, self).__init__()

        # init history 
        self.history_policy = tf.keras.callbacks.History()
        self.history_q = tf.keras.callbacks.History()


        # build policy model (state of dim 8 in, output mu and sigma for each action)
        self.policy_input = tf.keras.Input(shape=(input_units,), name="policy_state_input")
        self.policy_layer2 = tf.keras.layers.Dense(16, activation='tanh')(self.policy_input)
        self.policy_layer3 = tf.keras.layers.Dense(16, activation='tanh')(self.policy_layer2)
        self.policy_mu_output = tf.keras.layers.Dense(output_actions, activation=None, use_bias=False)(self.policy_layer3)
        self.policy_sigma_output = tf.keras.layers.Dense(output_actions, activation="relu", use_bias=False)(self.policy_layer3) # relu for pos values only 
        self.policy_model = tf.keras.Model(self.policy_input, [self.policy_mu_output, self.policy_sigma_output], name="policy") 
        self.policy_model.compile(loss=self.custom_policy_loss, optimizer='adam') 
        self.policy_model.summary()


        # build q model (state of dim 8 and actions of dim 2 in, output q value for action-pair of dim 2)
        self.q_input_state = tf.keras.Input(shape=(input_units,), name="q_state_input")
        self.q_input_action = tf.keras.Input(shape=(output_actions,), name="q_action_input")
        self.q_input_combined = tf.keras.layers.concatenate([self.q_input_state, self.q_input_action])
        self.q_layer2 = tf.keras.layers.Dense(16, activation='tanh')(self.q_input_combined)
        self.q_layer3 = tf.keras.layers.Dense(16, activation='tanh')(self.q_layer2)
        self.q_output = tf.keras.layers.Dense(1, activation=None, use_bias=False)(self.q_layer3) 
        self.q_model = tf.keras.Model([self.q_input_state,self.q_input_action], self.q_output, name='q')
        self.q_model.compile(loss='mean_squared_error', optimizer='adam') 
        self.q_model.summary()


    def custom_policy_loss(self,y_actual,y_predicted):
        return y_predicted

    def call(self, state_in): # returns policy pred 
        output = {}
        policy_pred = self.policy_model(state_in)
        output["mu"], output["sigma"] = policy_pred        
        return output # ignore q network as we only need info about action here; q for training 

    def call_q(self,state,action):
        q_pred = self.q_model([state,action])
        return q_pred

    def train_q(self,data_dict,batch_size):
        self.q_model.fit([data_dict['state'],data_dict['action']], data_dict['q_target'], epochs=1, batch_size=batch_size, callbacks=[self.history_policy], verbose=0) 

    def train_policy(self,data_dict,batch_size):
        
        # create bridge distribution between action and mus, sigmas  
        mus = data_dict["action"]
        sigmas = tf.constant(0.001, shape=(data_dict["action"].shape))
        dummy_state = tf.constant(0.001, shape=(100,8))

        # fit using custom loss function 
        self.policy_model.fit(dummy_state,[mus,sigmas],epochs=1,batch_size=batch_size,callbacks=[self.history_q],verbose=0) # dummy_state is ignored by loss  


if __name__ == "__main__":

    # constants 
    GAMMA = 0.99

    buffer_size = 5000
    test_steps = 300
    epochs = 20
    sample_size = 500 #1000
    optim_batch_size = 8
    saving_after = 5

    kwargs = {
        "model": DDPG,
        "environment": "LunarLander-v2",
        "num_parallel": 8,
        "total_steps": 400, # 100
        "action_sampling_type": "continuous_normal_diagonal", # expects "mu" and "sigma" in network output 
        "num_episodes": 40 # per runner
    }

    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "/progress_ddpg"


    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, test_episodes=10, do_print=True, render=True)

    # get initial agent
    agent = manager.get_agent()
    #print(agent.model.trainable_variables)

    # keep track of buffer size and time to update
    number_of_elems_in_buffer = 0
    time_to_update = False 


    for e in range(epochs):


        print("\nStarting epoch " + str(e+1))


        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)
        number_of_elems_in_buffer += len(data['reward'])
        if (number_of_elems_in_buffer < buffer_size):
            print("Now " + str(number_of_elems_in_buffer) + " elements in buffer. Waiting until buffer is filled before sampling for optimization.")
        else:
            print("Buffer full (" + str(buffer_size) +  " elements), saw additional " + str(number_of_elems_in_buffer - buffer_size) + " elements. Sampling now for optimization.")

            # Update only every second epoch 
            if epochs % 2 == 0:
                time_to_update = True

            # update with batch 
            if time_to_update:

                time_to_update = False 
                
                # sample data from buffer; batch size is equivalent to "however many updates"
                sample_dict = manager.sample(sample_size)

                # sample actions for next states
                mu_sigma_tplus1 = agent.model.call(sample_dict["state_new"])
                mus, sigmas = mu_sigma_tplus1["mu"].numpy(), mu_sigma_tplus1["sigma"].numpy()
                a_tplus1 = norm.rvs(mus, sigmas)

                # sample q net response 
                q_tplus1 = agent.model.call_q(sample_dict["state_new"],a_tplus1)

                # compute targets 
                q_target = sample_dict["reward"] + GAMMA * sample_dict["not_done"] * q_tplus1 # y 
                sample_dict["q_target"] = q_target

                # train q network on target qs 
                agent.model.train_q(sample_dict,optim_batch_size)

                # gradient ascent 
                agent.model.train_policy(sample_dict, optim_batch_size)


                # Get last losses
                losses = agent.model.history_policy.history['loss']
                losses.append(agent.model.history_q.history['loss'])

                # Get optimized weights and update agent
                new_weights = agent.model.get_weights()
                manager.set_agent(new_weights)
                agent = manager.get_agent()

                # update aggregator
                time_steps = manager.test(test_steps)
                manager.update_aggregator(loss=losses, time_steps=time_steps)
                print(f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses])}   avg env steps ::: {np.mean(time_steps)}")


                if e % saving_after == 0:
                    manager.save_model(saving_path, e)



    # test after training
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)




'''

# Action is two floats [main engine, left-right engines].
# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
# Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off


The state. Attributes:
s[0] is the horizontal coordinate
s[1] is the vertical coordinate
s[2] is the horizontal speed
s[3] is the vertical speed
s[4] is the angle
s[5] is the angular speed
s[6] 1 if first leg has contact, else 0
s[7] 1 if second leg has contact, else 0


'''