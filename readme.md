# An Efficient Stochastic Game Framework for Energy Management in Microgrid Networks

This repository contains all the code for the paper <b>"An Efficient Stochastic Game Framework for Energy Management in Microgrid
Networks"</b>. The code is organized as follows:
- The 3-agents folder contains all the code and analysis files for Setup1 (both normal and stochastic ADL).
- The 8-agents folder contains all the code and analysis files for Setup2 and Setup3 (both normal and stochastic ADL).

The file **requirements.txt** contains all the requirements needed to run the code. You can also use **install.sh** to set up a virtual environment and install all the dependencies automatically.

To install all the dependencies follow the following steps:
~~~
pip install -r requirements.txt
~~~
or for creating a virtual environment and then installing the dependencies:
~~~
chmod +x install.sh
./install.sh
~~~
Make sure you have **virtualenv** package installed. Check [this](https://gist.github.com/frfahim/73c0fad6350332cef7a653bcd762f08d) guide to install virtualenv.

### Steps to run the code
- Go to any folder which contains code. Each of the folders will contain:
    - main.py (contains all the code to run the experiment).
    - DQN_agent.py (contains classes and functions for the agents).
    - analysis notebook (analysis of all the results).
- Create two folders, 'logs' and 'saved', where all the results of the experiments and the trained models will be saved
- In the terminal type the following to run the code:
~~~
python main.py
~~~
- All the results will be stored in the 'logs' folder and all the trained models will be stored in the 'saved' folder.
- **All the logs and saved models for all the experiments that we have performed can be found in this [link](https://drive.google.com/open?id=1HFCvPiJ6llfMvZfTlDOlref2URTrNOSv)**.
