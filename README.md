# Automating Object Detection : Zero TO One

## Requirements:

1. This project is supposed to be run in a docker with tensorflow-serving setup in it.
   (To save the trouble, you can pull my docker container with all requirements installed: sannzay/tf_serve:version4)
2. tensorflow (both in docker and local)
3. utils (both in docker and local)
4. grpcio (only in local)
5. grpcio-tools (only in local)
6. matplotlib (both in docker and local)
7. pillow (both in docker and local)
8. scipy (only in local)
9. tensorflow-serving-api (only in local)
10. protobuf-complier (both in docker and local)
11. numpy (both in docker and local)
12. pandas (both in docker and local)


## Instructions to run:

Run the following comand in the docker terminal:
```
python3 objectdetection.py --class=1 --label=label.pbtxt --steps=62 --eval=10 --train=train --test=test --model="faster_rcnn_inception_v2_pets.config"
```
Change the following values as per your needs before running the script.
1. Class (number of classes in the training set)
2. label (file containing all the labels  of your dataset, one per each line)
3. steps (number of steps it has to be trained)
4. eval (number of images you have for validation)
5. train (path to the training dataset)
6. test (path to the test dataset)
7. model (architecture you want to use for training)

**NOTE:** This project can also be executed without docker, but the hosting part of the model won't be achieved. (In that case comment the last line (os.system command) in the objectdetection.py file) 

## Detailed Zero to One description about this project 

1. SSD-inception using Tensorflow object detection api
	- 1.1 Dataset preparation
	- 1.2 Downloading and installation of tensorflow api and its dependencies
	- 1.3 Setting up the network architecture as per our needs
	- 1.4 Training and Testing
	- 1.5 Results
2. Yolo v2 object detection using Darkflow
	- 2.1 Dataset preparation
	- 2.2 Downloading and installation of tensorflow api and its dependencies
	- 2.3 Setting up the network architecture as per our needs
	- 2.4 Training and Testing
	- 2.5 Results
3. Fast RCNN using Tensorflow object detection api
	- 3.1 Dataset preparation
	- 3.2 Downloading and installation of tensorflow api and its dependencies
	- 3.3 Setting up the network architecture as per our needs
	- 3.4 Training and Testing
	- 3.5 Results
4. Tensorflow serving
	- 4.1 Generating model files and structuring them as per the tf_serving needs
	- 4.2 Installing the docker
	- 4.3 Building the tensorflow_serving docker image
	- 4.4 Exporting and Running the model
	- 4.5 Accessing the service using the client file



## 1. SSD-inception using Tensorflow object detection api

### 1.1 Dataset preparation

Create a Folder Tensorflow
```
cd Tensorflow/train1

mkdir addons
cd addons 
```
Install labelImg
```
git clone https://github.com/tzutalin/labelImg.git


sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
cd ..
mkdir images
cd images
```
Insert test and train image datasets
```
cd ..
mkdir annotations #stores all csv files and tfrecords of the datasets
```
Generate xml images for all the training and testing datasets and place them in the corresponding folders

Create a label_map.pbtxt and place inside the train1/annotaion folder  and train1/training folder

```
item {
	id: 1
	name: ‘car’
}
```
#Pre-processing

Create a folder with the name “preprocessing” to keep all the scripts used to convert the data to the desired format.

#converting all xml files to csv files
Use the script xml_to_csv.py to convert all the csv files to the train.csv and test.csv
```
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train.csv
```
put them in tain1/annotaions folder

#converting csv files to tfrecord files
Use the script generate_tfrecord.py to convert the csv files to the tfrecord files
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record
put them in tain1/annotaions folder



### 1.2 Downloading and installation of tensorflow api and its dependencies

Install pillow, lxml, jupyter, matplotlib, opencv 
```
git clone https://github.com/tensorflow/models.git 

cd Tensorflow

cd models/research/

protoc object_detection/protos/*.proto –python_out=.

export PYTHONPATH=$PYTHONPATH:/content/gdrive/'My Drive'/TensorFlow/models/research/object_detection

python3 setup.py build

python setup.py install

export PYTHONPATH=$PYTHONPATH:content/gdrive/'My Drive'/TensorFlow/models/research:content/gdrive/'My Drive'/TensorFlow/models/research/slim
```
```
cd /Tensorflow
mkdir workspace
cd workspace
mkdir train1
cd train1
```

### 1.3 Setting up the network architecture as per our needs

Lets use transfer learning for training.
Download the required model from the tensorflow detection model-zoo github diectory (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
Here we will use ssd_mobilenet_v1_coco_2017_11_17.tar.gz
```
mkdir train1/pre-trained-model 
```
Unzip the contents and put them inside the train1/pre-trained-model

Get the corresponding config file from the models/research/object_detection/samples/config folder

Here it is ssd_inception_v1_coco.config

Change the number of classes we are training on. Here it is only 1 class (car)

Change the IOU threshold to 0.3 which is the optimum value used in papers.

Change the threshold _score = 0.7, which not only decrease the number of computations for IOU processing but also helps to filter out the predictions which have a minimum probability of 70%. 

Mention the path to the train.record and test.record files that we have created.

Mention the path to the label_map.pbtxt which contains the class names and their indices.

Mention the path to the trained weights of the model that we have downloaded

Set the learning rate and evaluation configurations as per our needs here the learning rate is 0.004

Optionally, we can change the number of training steps to suit our needs eg: 30 or 100 (so less mentioned, as we are just updating the already trained network)  


### 1.4 Training and Testing

Now lets go to the models/research/object_detection/legacy/ and copy the train.py from there and copy it into the train1/ folder

Make sure that all the dependencies are exported to the python path (Orelse you will have to copy the nets and deployment folders to the train1 folder)

Now train the network using the train.py, mentioning the corresponding config file and the training directory where our trained models will be saved
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v1_coco.config
```

You will soon see the training process updates on the screen.


### 1.5 Results
```
Training set contains: 61 images
Testing set contains: 10 images
Dataset contains car images with only one class, car
learning rate:0.0004
Time taken for a single image inference(including post processing): 0.667 seconds
```
NOTE: Due to the high variance condition due to the less amount of data used for training, loss values may be misleading, instead refer mAP values and the output images.

Below results are evaluated on the test dataset.
```
Results till 40 steps:

classification_loss: 0.177876
localization_loss: 0.073772
mAP@0.5IOU: 0.312029 

last step loss: 0.6
Time taken: 446 seconds
Output images:
```
```
Results till 62 steps:

classification_loss: 0.323334
localization_loss: 0.223708
mAP@0.5IOU: 0.666667

last step loss: 1.3
Time taken: 656 seconds
Output images:
```
```
Results till 100 steps:

classification_loss: 0.301499
localization_loss: 0.206459
mAP@0.5IOU: 0.666667 

last step loss: 1.3
Time taken: 992 seconds
Output images:
```
```
Results till 120 steps:

classification_loss: 0.293540
localization_loss: 0.157667
mAP@0.5IOU: 0.666667 

last step loss: 0.6
Time taken: 1190 seconds
Output images:
```

## 2. Yolo v2 object detection using Darkflow

2.1 Dataset preparation
2.2 Downloading and installation of tensorflow api and its dependencies
2.3 Setting up the network architecture as per our needs
2.4 Training and Testing

### 2.5 Results
```
Training set contains: 61 images
Testing set contains: 13 images
Dataset contains car images with only one class, car
learning rate:0.001
Time taken for a single image inference(including post processing): 0.378 seconds
```
Below results are evaluated on the test dataset.
```
Results till 875 steps:

mAP@0.5IOU: 0.1259
 

last step loss: 1.3
Time taken: 14244.76 seconds = 3 hr 57 min
Output images:

Results till 1000 steps:
```
```
mAP@0.5IOU: 0.1667

last step loss: 1.1
Time taken: 16278 seconds
Output images:

Results till 1125 steps:
```
```
mAP@0.5IOU: 0.1667

last step loss: 0.92
Time taken: 18313.7 seconds
Output images:

Results till 1250 steps:
```
```
mAP@0.5IOU: 0.2222

last step loss: 0.85
Time taken: 20348 seconds
Output images:
```
## 3. Fast RCNN using Tensorflow object detection api

### 3.1 Dataset preparation

Refer 2.1 

### 3.2 Downloading and installation of tensorflow api and its dependencies

Refer 2.2

### 3.3 Setting up the network architecture as per our needs

Lets use transfer learning for training.
Download the required model from the tensorflow detection model-zoo github diectory (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
Here we will use faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
mkdir train2/pre-trained-model 
```
Unzip the contents and put them inside the train1/pre-trained-model

Get the corresponding config file from the models/research/object_detection/samples/config folder

Here it is faster_rcnn_inception_v2_pets.config

Change the number of classes we are training on. Here it is only 1 class (car)

Change the IOU threshold to 0.45 which is the optimum value used in papers.

Change the threshold _score = 0.7, which not only decrease the number of computations for IOU processing but also helps to filter out the predictions which have a minimum probability of 70%. 

Mention the path to the train.record and test.record files that we have created.

Mention the path to the label_map.pbtxt which contains the class names and their indices.

Mention the path to the trained weights of the model that we have downloaded

Set the learning rate and evaluation configurations as per our needs here the learning rate is 0.0002 

Optionally, we can change the number of training steps to suit our needs eg: 30 or 100 (so less mentioned, as we are just updating the already trained network)  

Now save this file in the train2/training folder

### 3.4 Training and Testing

Now lets go to the models/research/object_detection/legacy/ and copy the train.py from there and copy it into the train1/ folder

Make sure that all the dependencies are exported to the python path (Orelse you will have to copy the nets and deployment folders to the train2 folder)

Now train the network using the train.py, mentioning the corresponding config file and the training directory where our trained models will be saved
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

You will soon see the training process updates on the screen.

### 3.5 Results
```
Training set contains: 61 images
Testing set contains: 10 images
Dataset contains car images with only one class, car
learning rate:0.0002
Time taken for a single image inference(including post processing): 2.67 seconds
```
NOTE: Due to the high variance condition due to the less amount of data used for training, loss values may be misleading, instead refer mAP values and the output images.

Below results are evaluated on the test dataset.
Results till 40 steps:
```
classification_loss: 0.453891
localization_loss: 0.753596
mAP@0.5IOU: 0.377063
mAP@.75IOU: 0.011551 

last step loss: 1.6
Time taken: 259 seconds
Output images:
```

Results till 62 steps:
```
classification_loss: 0.734477
localization_loss: 0.461803
mAP@0.5IOU: 0.811881
mAP@.75IOU: 0.295766

last step loss: 1.5
Time taken: 405 seconds
Output images:
```
Results till 100 steps:
```
classification_loss: 0.485620
localization_loss: 0.265662
mAP@0.5IOU: 0.732673
mAP@.75IOU: 0.577228 

last step loss: 0.75
Time taken: 621 seconds
Output images:
```

## 4. Tensorflow serving

### 4.1 Generating model files and structuring them as per the tf_serving needs

Present version of Tensorflow serving doesn’t accept the direct accessing of the model and its weights. We need to get the saved model to be served in the session which contains both the weights and graph structure and also freeze (used for the inference) the model before we export it to the serving engine. That is, create the saved model (.pb file) from the available weight and structure files (.ckpt files). Note that saved model is not a serializable one and can be used for retraining whereas once its frozen it becomes serializable and cannot be used for training.

Go to the models/research/object_detection folder, where you will find export_inference_graph.py file which helps to create the desired models. Mention the path to the config file and checkpoints used for training the model while running it.
```
python3 -u export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path=/home/convo/Downloads/faster_rcnn_inception_v2_pets.config \
  --trained_checkpoint_prefix=tensorflow-obj/cfast100/model.ckpt-100 \
  --output_directory=model_output/cfast100
```
Once the command is executed you will find the saved model, frozen model and checkpoints files at the mentioned output location. 

Grab the contents in the saved_model folder and put inside a new folder named “1” outside the working directory. Create a new folder with the name of your own neural network, lets say “fastcarmodel” and put the folder “1” inside it. 

### 4.2 Installing the docker

Open the terminal and execute the following commands in the sequence provided:
```
sudo apt update
sudo apt upgrade

sudo apt install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt update
sudo apt install docker-ce
```
By now you should be able to see the running updates of the docker by executing the following command:
```
sudo systemctl status docker 
```


### 4.3 Building the tensorflow_serving docker image

First we have to clone the tensorflow serving github repository before proceeding to the building of the docker image.
```
git clone --recursive https://github.com/tensorflow/serving
```
Enter into the tensorflow serving project folder through terminal
```
cd serving
```
Inside /tools/docker/ you will find the Dockerfile.devel which lists all the dependencies required for us to build the tensorflow_serving service. Through docker we can create a specific container which is unaffected by other system files and loaded with all the dependencies mentioned.

This can be achieved by executing the following command from serving/:
```
sudo docker build --pull -t $USER/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
```
Once it is build, run it by the following command:
```
docker run --name=tensorflow_container -it $USER/tensorflow-serving-devel
```
Check for the contents inside the container. Some versions come with tensorflow_serving already configured inside it. If it is the case, proceed to the next step. Orelse manually clone the tensorflow_serving repo and configure it using the following command:
```
git clone --recursive https://github.com/tensorflow/serving 
```
Now we need to build the tensorflow_serving project using Google’s bazel from inside the container. It downloads and manages all the dependencies required for the serving.
```
bazel build -c opt tensorflow_serving/...
```

### 4.4 Exporting and Running the model

Once the building is finished completely, get back to the directory where we saved our ready-made final graph folder, “fastcarmodel” through terminal.

Now we have to copy that folder into the container we built by executing the following command:
```
sudo docker cp ./fastcarmodel x:/serving  (x is the contained id {eg: root@x})
```
Now get back to the container and check whether the model is exported. Once it is done, we are all set to host our model on the server.

Use the following command from inside the container to host the model using using tensorflow-serving (Mention the port number, model name, log file and desired path for the model ):
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/serving/fastcarmodel &> fastcarmodel_log & 
```
You can check the updates of the server in the fastcarmodel_log as mentioned in the above command.

4.5 Accessing the service using the client file

Before leaving the container, install the required grpc tools by the following command:
```
pip3 install grpcio grpcio-tools
```
Now, get back to the directory where the client.py is saved and again install the grpc tools here.
```
pip3 install grpcio grpcio-tools
```
Get the IP address of our container by using the following command: 

sudo docker network inspect bridge | grep Ipv4Address

Once it is executed, you will see the address where our service is hosted. Note it.

Now get the inference by executing the client.py file, mentioning the address and the image path that we want to analyze, as follows:
```
python3 client.py --server=172.17.0.2:9000 –image=./test.jpg
```
You will see a json response from the model explaining the details of detection (number of detections with the bounding boxes)

 
