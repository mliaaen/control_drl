# main function that sets up environments
# and performs training loop

from collections import deque
from maddpg_agent import Agent
import numpy as np
import os
import time
import torch
from unityagents import UnityEnvironment

N_EPISODES = 2000
SOLVED_SCORE = 0.5
CONSEC_EPISODES = 100
PRINT_EVERY = 10
ADD_NOISE = True




class Runner():
    
    def __init__(self, config):
        """

        :rtype: object
        """
        super(Runner, self).__init__()
        self.n_episodes = config.get("n_episodes", N_EPISODES)
        self.solved_score = config.get("solved_score", SOLVED_SCORE)
        self.conseq_episodes = config.get("conseq_episodes", CONSEC_EPISODES)
        self.seed = config.get("seed", 1)

        print(self.n_episodes)

        self.agents = []

        self.env = UnityEnvironment(file_name='Tennis.app')

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)

        # examine the state space
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like: \n{}\n'.format(states[0]))

        # initialize agents
        self.agents = [Agent(self.state_size, self.action_size, 1, self.seed, config=config) for i in range(self.num_agents)]
        print (self.agents)

    def __del__(self):
        #self.env.reset(train_mode=True)[self.brain_name]
        #self.env.close()
        print("cleaning up run-environment getting ready for next round")

    def seeding(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def get_actions(self, states, add_noise):
        #print("get_actions", states, type(states), self.num_agents, self.action_size)
        actions = [agent.act(states, add_noise) for agent in self.agents]
        # flatten action pairs into a single vector
        #print("return actions", actions)
        return np.reshape(actions, (1, self.num_agents*self.action_size))

    def reset_agents(self):
        for agent in self.agents:
            agent.reset()

    def learning_step(self, states, actions, rewards, next_states, done):
        for i, agent in enumerate(self.agents):
            agent.step(states, actions, rewards[i], next_states, done, i)

    ## Training loop


    def training_loop(self):
        # start environment

        # initialize scoring
        scores_window = deque(maxlen=CONSEC_EPISODES)
        scores_all = []
        moving_average = []
        best_score = -np.inf
        best_episode = 0
        already_solved = False
        self.seeding()


        for i_episode in range(1, self.n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]      # reset the environment
            states = np.reshape(env_info.vector_observations, (1,self.num_agents*self.state_size)) # flatten states
            #print("joho",states, self.num_agents, self.state_size)
            self.reset_agents()
            scores = np.zeros(self.num_agents)
            while True:
                actions = self.get_actions(states, ADD_NOISE)           # choose agent actions and flatten them
                env_info = self.env.step(actions)[self.brain_name]           # send both agents' actions to the environment
                next_states = np.reshape(env_info.vector_observations, (1, self.num_agents*self.state_size)) # flatten next states
                rewards = env_info.rewards                         # get rewards
                done = env_info.local_done                         # see if the episode finished
                #print("learning", next_states, actions, done)
                self.learning_step(states, actions, rewards, next_states, done)  # perform the learning step
                scores += np.max(rewards)                          # update scores with best reward
                states = next_states                               # roll over states to next time step
                #print(actions)
                if np.any(done): # exit loop if episode finished
                    #print ("done")
                    break

            ep_best_score = np.max(scores)                         # record best score for episode
            scores_window.append(ep_best_score)                    # add score to recent scores
            scores_all.append(ep_best_score)                       # add score to histor of all scores
            moving_average.append(np.mean(scores_window))          # recalculate moving average

            # save best score
            if ep_best_score > best_score:
                best_score = ep_best_score
                best_episode = i_episode

            # print results
            print(i_episode)
            if i_episode % PRINT_EVERY == 0:
                print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                    i_episode-PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))

            # determine if environment is solved and keep best performing models
            if moving_average[-1] >= SOLVED_SCORE:
                if not already_solved:
                    print('<-- Environment solved in {:d} episodes! \
                    \n<-- Moving Average: {:.3f} over past {:d} episodes'.format(
                        i_episode-CONSEC_EPISODES, moving_average[-1], CONSEC_EPISODES))
                    already_solved = True
                    # save weights
                    torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
                    torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
                    torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
                    torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
                elif ep_best_score >= best_score:
                    print('<-- Best episode so far!\
                    \nEpisode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                    i_episode, ep_best_score, moving_average[-1]))
                    # save weights
                    torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
                    torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
                    torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
                    torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
                # stop training if model stops improving
                elif (i_episode-best_episode) >= 200:
                    print('<-- Training stopped. Best score not matched or exceeded for 200 episodes')
                    break
                else:
                    continue
        self.env.close()
        return best_score


## Helper functions
def get_hyperparams(**args):
    import itertools
    for k, v in args.items():
        print("%s = %s" % (k, v))

    param_list = []
    param_list_dict = []
    #print(arg[j] for j in range(len(arg))
    print(args)
    for i in itertools.product(args["n_episodes"], args["seeds"]):
        param_list.append(i)
    for config in param_list:
        param_list_dict.append({"n_episodes":config[0], "seeds":config[1]} )

    return param_list_dict


config = {"n_episodes":300, "seed":100}

episodes = [10, 20]
seeds=[1]

hyper_configs = get_hyperparams(n_episodes=episodes, seeds=seeds)



scores = []
for config in hyper_configs:

    print("Now running with config:", config)
    runner = Runner(config)
    best_score = runner.training_loop()
    scores.append(best_score)
    del(runner)
    time.sleep(5)
print (scores)

i = score.index(max(scores))
print ("best config %i %s"%(i, hyper_configs[i]))
