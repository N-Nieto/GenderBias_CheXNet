# GenderBias_CheXNet
In this Readme you will find all the steps you will need to do in order to reproduce the experiments showing gender bias.

Optional Step: If is your fist time coding, you will have to install Python. We recommend to install Anaconda.

- Open a terminal and type:
  >>sudo install anaconda

Step 1 - Download the data:

Download the Chext14 from: 

TODO: add a simple command for download from Terminal.

Step 2 - Download this repository

- Clone and download GenderBias_cheXNet.

In this repository you will find all the scripts we use to run the experiments.

Step 3 - Create a Python enviroment

  1- Open a Terminal in the path if the repository, or where you have the enviroment.ylm file.
  2- >>conda env create -your_env_name -enviroment

Step 4 - Activate the enviroment

- Open a new terminal
  >>source activate your_env_name

Step 5 - Train the network

Fist, you need yo open the "config_bias" file and change the path where you have all the images.

In the same terminal you have the enviroment activated type.

(yout_env_name)>> python3 training.py

When the training process is finish, you will find the folder /output that conteins the trained weigths of the network.

Step 6 - Testing the network

On a Terminal with your_env_name activated run

  (your_env_name)>>python3 testing.py
  
When the testing is over, you will find the predictions made for the network in the same /output folder.

Step 7 - Visualize your predictions

In order to ilustrate the network performance.

On a Terminal with your_env_name activated run:

   (your_name_env)>>jupyter notebook Performance_evaluated.jp
   
When the notebook is load click "Run All Cells", in order to reproduce the images in the paper.

