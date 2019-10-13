# object-detection

## Requirements:

1. This project is supposed to be run in a docker with tensorflow-serving setup in it.
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

python3 objectdetection.py --class=1 --label=label.pbtxt --steps=62 --eval=10 --train=train --test=test --model="faster_rcnn_inception_v2_pets.config"

Change the following values as per your needs before running the script.
1. Class (number of classes in the training set)
2. label (file containing all the labels  of your dataset, one per each line)
3. steps (number of steps it has to be trained)
4. eval (number of images you have for validation)
5. train (path to the training dataset)
6. test (path to the test dataset)
7. model (architecture you want to use for training)

**NOTE:** This project can also be executed without docker, but the hosting part of the model won't be achieved. (In that case comment the last line (os.system command) in the objectdetection.py file) 