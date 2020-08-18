# Udacity deep-reinforcement-learning Navigation project
Navigation problem using Deep Reinforcement Learning Agent

# 1. Project Details

## Unity ML-Agents
**Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. In this project I use Unity's rich environments to design, train, and evaluate a deep reinforcement learning algorithms. 

You can learn more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

### Overview
For this project, I will train an agent to navigate (and collect bananas!) in a large, square world.

<p align="center">
  <img src="https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif" />
</p>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Completion criteria

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 
over 100 consecutive episodes.


## Agent

The class `Agent`is the well-known class implementing the following mechanisms:

* Two Q-Networks (local and target) using the a neural network.
* Replay memory (using the class ReplayBuffer)
* Epsilon-greedy mechanism
* Q-learning, i.e., using the max value for all possible actions
* Computing the loss function by MSE loss
* Minimize the loss by gradient descend mechanism using the ADAM optimizer
* Soft update from local QNetwork parameters to target QNetwork parameters

## Model Q-Network

Both Q-Networks (local and target) are implemented by the class `QNetwork`, a simple neural network with 3 fully-connected layers and 2 rectified nonlinear layers. This `QNetwork` class is implemented in the framework of Python package `PyTorch`. The number of neurons of the fully-connected layers are 
as follows:

 * Layer fc1,  number of neurons: _state_size_ x _fc1_units_, 
 * Layer fc2,  number of neurons: _fc1_units_ x _fc2_units_,
 * Layer fc3,  number of neurons: _fc2_units_ x _action_size_,
 
where _state_size_ = 37, _action_size_ = 8, _fc1_units_ and _fc2_units_ are the input params.

## Deep-Q-Network Algoritm

We run several training sessions and we included in `Navigation.ipynb` the best one.  We did that using a `agent` with different parameters and we run the *Deep-Q-Network* procedure `dqn` as follows:

```
  agent = Agent(state_size=state_size, action_size=action_size, seed=1, fc1_units=fc1_nodes, fc2_units=fc2_nodes)       
  scores, episodes epsilon_list = dqn(n_episodes = 1800, eps_start = 1)  
```  

We experience the following parameters:  

* `fc1_units` : Number of nodes for the first fully connected layer.
* `fc2_units` : Number of second for the first fully connected layer.

The obtained weights for the best training session is sabed into the file `model.pt`.

The _Deep-Q-Network_ procedure `dqn` performs  a double loop. External loop is executed till the number of episodes reached the maximal number or the completion criteria is executed `np.mean(scores_window) >=13`, where `scores_window` is the array of the type `deque` realizing  the shifting window of length `<= 100`. The elements `scores_window[i]` and `epsilon_list[i]` contains the `score` and `epsilon` respectively, achieved by the algorithm on the episode `i`.

## Output of training
This is the output of one of our training sessions that indicates, for a given DQN architecture (printed), the number of episodes required to train the model:
```
DQN architecture of the Agent:
QNetwork(
  (fc1): Linear(in_features=37, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=40, bias=True)
  (fc3): Linear(in_features=40, out_features=4, bias=True)
)
Episode: 450, Avg.Score: 13.03,  Score 15.0, Scores >= 13: 59, Epsilon: 0.07
 terminating at episode : 450 ave reward reached +13 over 100 episodes
```
We can plot the evolution of the `score` and `epsilon` during training:
 ![score.png][score] 
 ![epsilon.png][epsilon] 


# 2. Content of this repository

*  `Navigation.ipynb`: file with fully functional code (all code cells are executed and displaying output).
*  `Report.pdf`: project report.
*   `model.pt`:file with the saved model weights of the successful agent.

# 3. Report
To know more details of this project you can download this [Report](/Report.md)

