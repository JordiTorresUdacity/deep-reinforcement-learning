# Udacity deep-reinforcement-learning Navigation project
Navigation problem using Deep Reinforcement Learning Agent

# Project Details

![](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)
## Unity ML-Agents
**Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. In this project I use Unity's rich environments to design, train, and evaluate a deep reinforcement learning algorithms. 

You can learn more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

## The Environment
For this project, I will train an agent to navigate (and collect bananas!) in a large, square world.

<p align="center">
  <img src="/images/banana.gif" />
</p>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Content of this repository

The `Navigation.ipynb file with fully functional code, all code cells executed and displaying output, and all questions answered. You can also download this via your workspace by clicking download as..
A README.md markdown file with a description of your code, much like this one.
An HTML or PDF export of the project report with the name Report.html or Report.pdf.
A file with the saved model weights of the successful agent, can be named something like model.pt.
Any additional images used for the project that were not supplied to you for the project. Please do not include the large banana, project data sets that you may download to work with. These files will make your project too large to submit.


## Train the Agent
```bash
python train.py
```

You can tune the model by changing the following hyperparameters in following files (default values below):

### train.py
* TOTAL_EPISODES = 500
* EPSILON_START = 1.0
* EPSILON_DECAY = .99
* EPSILON_END = 0.01
* DOUBLE_DQN = True
  * See paper [here](https://arxiv.org/abs/1509.06461)
* DUELING_DQN = True
  * See paper [here](https://arxiv.org/abs/1511.06581)
### agent.py
* LR = 5e-4 (learning rate)
* BUFFER_SIZE = 100.000
* BATCH_SIZE = 64
* GAMMA = .99 (discount factor)

* UPDATE_EVERY = 4 (How many steps to wait before update the target QNetwork)
* TAU = 1e-3 (soft update from local QNetwork parameters to target QNetwork parameters)

## Report
To see more details in this [Report](/Report.md)

