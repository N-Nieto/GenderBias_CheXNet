# GenderBias_CheXNet
In this tutorial you will find all the steps and instructions you need in order to reproduce the experiments showing gender bias.

### Step 0: If it is your first time coding in Python 3, you will have to install it. We recommend to install Anaconda Distribution:

You could find some straigthforward instructions in the following tutorial:

https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart


### Step 1 - Download the GenderBias_CheXNet repository:

In this repository you will find all the scripts needed to repoduce our experiments.

### Step 2 - Download the X-ray images (If you already have the dataset skip this step):

- Open a Terminal

- Set the terminal path in the unzip GenderBias_CheXNet

(base)>> `python batch_download_zips.py`

This may take a while. 

If you rather prefer to download the data by your own, you could find all the files here:

https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

### Step 3 - Create a Python enviroment:

  1- Open a Terminal in the repository's path.
  
  2- Run the following command:
 
  (base)>>`conda env create --name your_env_name  --file requirements.yml`

### Step 4 - Activate the environment with the following command:

  (base)>>`source activate your_env_name`
  
  You will see your environment name in the command line
  
  (your_env_name)>>
  

### Step 5 - Training the network:

First, make sure that in "config_file.ini" the image_source_dir contains the path where you have download the dataset.

Run the training script with the following command:

(yout_env_name)>> `python3 training.py`

When the training process finished, you will find the "/output" folder that contains the trained weights of the network.

### Step 6 - Testing the network:

Now that you have your model trained, it is time to generate predictions in unseen data

Run the testing script with the following command:

  (your_env_name)>>`python3 testing.py`
  
When the testing is over, you will find the network predictions in the "/output" folder. 

As an example, for the fold 0, training with only male images and testing on female set you will find:

`y_pred_run_0_train_0%_female_images_test_female.csv`
