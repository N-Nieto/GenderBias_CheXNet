# GenderBias_CheXNet
In this Readme you will find all the steps you will need to do in order to reproduce the experiments showing gender bias.

Step 0: If is your fist time coding, you will have to install Python. We recommend to install Anaconda Distribution.

You could find some easy explained instuctions in the following tutorial:
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart


Step 1 - Download this repository

- Clone and download GenderBias_cheXNet.

In this repository you will find all the scripts we use to run the experiments.

Step 2 - Download the data (If you already have the data downloaded skip this step) :

- Open a Terminal
- Set the terminal path in the unzip GenderBias_CheXNet
>>python batch_download_zips.py

This may take a while. 

If you rather prefer to download the data by your own, you could find all the files here:
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737


Step 3 - Create a Python enviroment

  1- Open a Terminal in the path if the repository, or where you have the enviroment.ylm file.
  2- >>`pip3 install -r requirements.txt`

Step 4 - Activate the enviroment

- Open a new terminal
  >>source activate your_env_name

Step 5 - Train the network

Fist, you need yo open the "config_bias" file and change the path where you have download the dataset.

In the same terminal you have the enviroment activated type.

(yout_env_name)>> python3 training.py

When the training process is finish, you will find the folder /output that conteins the trained weigths of the network.

Step 6 - Testing the network:

Now we have your model trained, it is time to generate predictions in unseen data

On a Terminal with your_env_name activated run

  (your_env_name)>>python3 testing.py
  
When the testing is over, you will find the predictions made for the network in the same /output folder.

Step 7 - Visualize your predictions

In order to ilustrate the network performance.

On a Terminal with your_env_name activated run:

   (your_name_env)>>jupyter notebook Performance_evaluated.ipynb
   
When the notebook is load click on top "Cell->Run All", in order to reproduce the images in the paper.

