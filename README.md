# Pytorch Lightning Jumpstart Guide

## Step #1: Log onto server
If you aren't connected to campus internet make sure you are connected to the universities VPN. After that you need to ssh into the server by running `ssh username@server_address` and then put your password into it.

Bonus step. It is good practice to run htop and nvidia-smi to get a feel for if anyone else is currently using the server and not hog all the resources.

## Step #2: Download files
Change directories to where you want to start this project. Then run `git clone https://github.com/tjkerby/pytorch_lightning_jumpstart.git`. 

## Step #3: Set up working environment
Run `docker build -t name_of_image .`
This create a docker image from the Dockerfile and installs the listed packages in the Dockerfile. 

Next run `docker run -itd --shm-size 16G --gpus all -v $(pwd)/:/usr/src/app name_of_image`
This creates a container with a shared volume between your working directory and /usr/src/app on the docker image. So all your files/folders in the current working directory will be available there and any additions, deletions, and alterations of the files/folders in /usr/src/app will be carried over after to the current working directory on the server. --gpus all is telling it to make all available gpus visible. You could change this to be a subset of your gpus if you want or need to share with others. Finally, --shm-size 16G is telling docker how much memory to share between the gpus. If you don't add this you will get errors like the ones found here https://github.com/pytorch/pytorch/issues/2244 if you don't add it and try to do multi-gpu training.

Now we can view the list of active containers by running `docker ps`. Under the IMAGE column you should see your specified name_of_image that we built and its associated CONTAINER ID.

Finally we can enter into the docker container by running `docker exec -it container_id /bin/bash` and insert the container id found when running docker ps. This will take you into the docker container where you have gpu access, your evironment setup, and the list of files needed to run your analysis.

## Step #4: Run your training script
Prior to starting model training lets edit config.py and edit the parameters for this particular run. Now all we need to do is run  `python mnist.py` to start training our model. 

## Step #5: Look at logs
Open vs code and connect it to the server where the python script is running. This may require adding to your ~/.ssh/config file something similar to the following:
Host shorcut_name
    HostName server_address
    User username
Once this is added you should be able to user VS Codes remote explorer and connect to it. Now open up the view_tensorbaord.ipynb file and run the script to connect tensorboard and look at the metrics tensorboard logged. You can use this to edit hyperparameters and finetune model training.
