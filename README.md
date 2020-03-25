# INF581-project
Advanced topics in Artificial Intelligence - Third year course at Ecole polytechnique

Repo for our final project: training an autonomous agent in the Gym Car-Racing environment. 

**Requirements**

The project needs gym box2D environments. To install it:
- you must have installed swig (http://www.swig.org/download.html), which might need Visual Studio Tools for C++ if you're on Windows
- then do "pip install box2d-py" and it should works fine

**Files**

- play.py : allows to play the game (up for accelerating, left or right for steering). When a trial is succesful, the data is stored in data>data.pkl.zip. You can change the path at your convenience in the play.py file if necessary.
- dqn.ipynb: contains the DQN, DDQN and DP (neural network policy) agents. Only the DQN works for the moment and obtains an average score of 814.0 with the model dqn_imitation_learning.h5 that is in the repo.
You can train yourself the model by using agent.train(memory) where the memory comes from the function read_data() that reads the file data>data.pkl.zip
- policy_gradient.ipynb: implements a policy gradient algorithm REINFORCE (see report for further documentation).
- evolutional.ipynb: implements an evolutionary optimisation process.

**Performances**

Here is a video showing the behavior of the agent after a training period of 1000 epochs of 100 training samples. (~30min training)
<div align=center><img src="resources/car_racing_demo.gif"/></div>

**Useful links:**
- Easy CHair for submissions and reviews: https://easychair.org/my/conference?conf=inf5812020
- The car racing environment we are using: https://gym.openai.com/envs/CarRacing-v0/
- Link to the moodle: https://moodle.polytechnique.fr/mod/assign/view.php?id=42619
- If you have not attended the courses: https://fr.wikipedia.org/wiki/Processus_de_d%C3%A9cision_markovien (just read this page and then try to do the PC)
- An interesting stanford paper. Contains describtion of the problem ! https://web.stanford.edu/class/aa228/reports/2018/final150.pdf
- The repository where we found the file play.py (later simplified a little bit) https://github.com/gui-miotto/DeepLearningLab/blob/master/Assignment%2003/DL%20Lab%20-%20Assignment%2003%20-%20Guilherme%20Miotto.pdf
- An interesting post on DQN : https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
- final report on overleaf: https://www.overleaf.com/project/5e5cd6a3be38d60001360ea1
