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

### Training sessions

We run several training sessions and we included in `Navigation.ipynb` the best one.  We did that using a **agent** with different parameters and we run the *Deep-Q-Network* procedure **dqn** as follows:

```
  agent = **Agent**(state_size=state_size, action_size=action_size, seed=1, fc1_units=fc1_nodes, fc2_units=fc2_nodes)       
  scores, episodes epsilon_list = **dqn**(n_episodes = 1800, eps_start = 1)  
```  

We experience the following parameters:  

* _fc1_units_ : Number of nodes for the first fully connected layer.
* _fc2_units_ : Number of second for the first fully connected layer.

The obtained weights for the best training session is sabed into the file `model.pt`.

# 2. Content of this repository

*  `Navigation.ipynb`: file with fully functional code (all code cells are executed and displaying output).
*  `Report.pdf`: project report.
*   `model.pt`:file with the saved model weights of the successful agent.

## 3. Report
To know more details of this project you can download this [Report](/Report.md)

