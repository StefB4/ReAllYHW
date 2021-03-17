import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class DQN(tf.keras.Model):
    def __init__(self, input_units=4, output_units=2):

        super(DQN, self).__init__()
        '''
        self.layer = tf.keras.layers.Dense(input_units)
        self.layer2 = tf.keras.layers.Dense(16)
        self.layer3 = tf.keras.layers.Dense(16)
        self.layer4 = tf.keras.layers.Dense(output_units)
        '''

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(16, input_dim=input_units, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(16, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(output_units, activation=None, use_bias=False))

        # compile 
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def call(self, x_in):

        # state makeup (from the Gym docs)
        # Observation:
        # Type: Box(4)
        # Num     Observation               Min                     Max
        # 0       Cart Position             -4.8                    4.8
        # 1       Cart Velocity             -Inf                    Inf
        # 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        # 3       Pole Angular Velocity     -Inf                    Inf

        output = {}
        '''
        x1 = self.layer(x_in)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        '''
        pred = self.model(x_in)

        output["q_values"] = pred
        return output


    def train(self,data_dict,batch_size):

        self.model.fit(data_dict['state'], data_dict['q_target'], epochs=1, batch_size=batch_size) #, verbose=0)


if __name__ == "__main__":

    kwargs = {
        "model": DQN,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 1000, # 100
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 100 #1000
    optim_batch_size = 8
    saving_after = 5

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
    #manager.test(test_steps, do_print=True, render=False)

    # get initial agent
    agent = manager.get_agent()
    print(agent.model.trainable_variables)

    GAMMA = 0.9
    EPSILONSTART = 0.99
    EPSILONDECAYRATE = 0.005
    EPSILONMIN = 0.01

    for e in range(epochs):

        print("Starting epoch " + str(e+1))

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        #for key, value in data.items() :
        #    print (str(key) + str(np.shape(data[key])))
        # returns: action(390,) state(390, 4) reward(390,) state_new(390, 4) not_done(390,) and 390 increases per episode

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        #data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)


        print("optimizing...")

        #for key, value in sample_dict.items() :
        #    print (str(key) + str(np.shape(sample_dict[key])))
        # returns: state(1000, 4) action(1000,) reward(1000,) state_new(1000, 4) not_done(1000,) 1000 is probably from sample size
        # data_dict holds batchformatted data 

        # TODO: iterate through your datasets

        
        
        # use samples from buffer to compute q_target
        if True:            

            # infer Q(s_t+1,a_t+1)
            q_state_new = agent.model.call(sample_dict["state_new"])['q_values']

            # compute target 
            q_target = sample_dict["reward"] + GAMMA * np.amax(q_state_new, 1)
            sample_dict["q_target"] = q_target

            # fit model 
            agent.model.train(sample_dict,optim_batch_size)



        

        # TODO: optimize agent

        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress
        #print(
        #    f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        #)

        # yeu can also alter your managers parameters
        new_epsilon = EPSILONSTART*(1-EPSILONDECAYRATE)**(e+1)
        if (new_epsilon < EPSILONMIN):
            new_epsilon = EPSILONMIN
        print("Updating epsilon to " + str(new_epsilon))
        manager.set_epsilon(epsilon=new_epsilon)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
