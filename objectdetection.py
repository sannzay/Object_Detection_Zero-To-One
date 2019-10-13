#  python3 objectdetection.py --class=1 --label=label.pbtxt --steps=62 --eval=10 --train=train --test=test --model="faster_rcnn_inception_v2_pets.config"

import os
import fileinput
import argparse
import time 


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True, help="Number of classes used for training")
ap.add_argument("-l", "--label", required=True, help="path to the filename consisting all label names of all the classes")
# ap.add_argument("-r", "--train", required=True, help="path to the train records")
# ap.add_argument("-e", "--test", required=True, help="path to the test records")
ap.add_argument("-m", "--model", required=True, help="Name of the config file")
ap.add_argument("-r", "--train", required=True, help="path to the train images")
ap.add_argument("-e", "--test", required=True, help="path to the test images")
ap.add_argument("-s", "--steps", required=True, help='Number of training steps')
ap.add_argument("-v", "--eval", required=True, help='Number of images to be evaluated in test data')
ap.add_argument("-i", "--iou", required=False, default=0.45, help='Number of images to be evaluated in test data')
ap.add_argument("-o", "--score", required=False, default=0.7, help='Number of images to be evaluated in test data')
args = vars(ap.parse_args())

dir_path = os.path.dirname(os.path.realpath(__file__))
labels = [line.rstrip() for line in open(args['label'])]
os.system('mkdir Tensorflow Tensorflow/addons Tensorflow/workspace Tensorflow/scripts Tensorflow/scripts/preprocessing Tensorflow/workspace/train1 Tensorflow/workspace/train1/pre-trained-model Tensorflow/workspace/train1/annotations Tensorflow/workspace/train1/images Tensorflow/workspace/train1/training Tensorflow/workspace/train1/evaluation')
os.chdir('Tensorflow')
os.system('git clone https://github.com/tensorflow/models.git')
os.chdir('../')
# os.system('mv models-master models')
print("Your training labels are: ")
print(labels)
with open("Tensorflow/workspace/train1/annotations/label_map.pbtxt", "w") as f:
	for i, l in enumerate(labels):
		f.write("item {\n\tid: %d\n\tname: '%s'\n}\n" % ((i+1), l))
# train_rec = 'cp '+args['train']+' Tensorflow/workspace/train1/annotations/'
# test_rec = 'cp '+args['test']+' Tensorflow/workspace/train1/annotations/'
# os.system(train_rec)
# os.system(test_rec)

os.system('cp xml_to_csv.py Tensorflow/scripts/preprocessing/')
os.system('cp generate_tfrecord.py Tensorflow/scripts/preprocessing/')

os.system(f'cp -rf {args["train"]} Tensorflow/workspace/train1/images/')
os.system(f'cp -rf {args["test"]} Tensorflow/workspace/train1/images/')


os.system('wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz')
os.system('tar -zxvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz')
os.system('cp -rf faster_rcnn_inception_v2_coco_2018_01_28/* Tensorflow/workspace/train1/pre-trained-model/')
## os.chdir('Tensorflow/models/research/object_detection/utils')
## os.chdir('../../../../../')
with fileinput.FileInput('Tensorflow/models/research/object_detection/utils/object_detection_evaluation.py', inplace=True) as file:
	for line in file:
		print(line.replace('unicode(', 'str('), end='')
os.chdir('Tensorflow/models/research')
os.system('protoc object_detection/protos/*.proto --python_out=.')
os.system(f'export PYTHONPATH=$PYTHONPATH:{dir_path}/Tensorflow/models/research/object_detection')
os.system('python3 setup.py build')
os.system('python3 setup.py install')
os.system(f'export PYTHONPATH=$PYTHONPATH:{dir_path}/Tensorflow/models/research:{dir_path}/Tensorflow/models/research/slim')
os.system(f'export PYTHONPATH=$PYTHONPATH:{dir_path}/Tensorflow/models/research/slim/nets')
os.chdir('../../../')
os.system(f'cp Tensorflow/models/research/object_detection/samples/configs/{args["model"]} Tensorflow/workspace/train1/training/{args["model"]}')
os.system('cp Tensorflow/models/research/object_detection/legacy/train.py Tensorflow/workspace/train1/train.py')
os.system('cp Tensorflow/models/research/object_detection/legacy/eval.py Tensorflow/workspace/train1/eval.py')
os.system(f'cp -rf {dir_path}/Tensorflow/models/research/slim/nets {dir_path}/Tensorflow/workspace/train1/')
os.system(f'cp -rf {dir_path}/Tensorflow/models/research/slim/deployment {dir_path}/Tensorflow/workspace/train1/')
os.system(f'cp -rf {dir_path}/Tensorflow/models/research/slim/nets {dir_path}/Tensorflow/models/research/object_detection/')
os.system(f'cp -rf {dir_path}/Tensorflow/models/research/slim/deployment {dir_path}/Tensorflow/models/research/object_detection/')

os.system('python3 Tensorflow/scripts/preprocessing/xml_to_csv.py -i Tensorflow/workspace/train1/images/train/ -o Tensorflow/workspace/train1/annotations/train_labels.csv')
os.system('python3 Tensorflow/scripts/preprocessing/xml_to_csv.py -i Tensorflow/workspace/train1/images/test -o Tensorflow/workspace/train1/annotations/test_labels.csv')
os.system(f'python3 Tensorflow/scripts/preprocessing/generate_tfrecord.py --label=car --csv_input=Tensorflow/workspace/train1/annotations/train_labels.csv  --img_path=Tensorflow/workspace/train1/images/train --output_path=Tensorflow/workspace/train1/annotations/train.record')
os.system(f'python3 Tensorflow/scripts/preprocessing/generate_tfrecord.py --label=car --csv_input=Tensorflow/workspace/train1/annotations/test_labels.csv  --img_path=Tensorflow/workspace/train1/images/test --output_path=Tensorflow/workspace/train1/annotations/test.record')

os.system('git clone https://github.com/cocodataset/cocoapi.git')
os.chdir('cocoapi/PythonAPI')
# For python3 uncomment the following snippet
with fileinput.FileInput('Makefile', inplace=True) as file:
	for line in file:
		print(line.replace('python','python3'), end='')
os.system('make')
os.system(f'cp -r pycocotools {dir_path}/Tensorflow/workspace/train1/')
os.system(f'cp -r pycocotools {dir_path}/Tensorflow/models/research/')
os.chdir('../../')
model = f'{args["model"]}'
if model == "faster_rcnn_inception_v2_coco.config":
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:
			print(line.replace('num_classes: 90',f'num_classes: {args["class"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:
			print(line.replace('PATH_TO_BE_CONFIGURED/model.ckpt',f'{dir_path}/Tensorflow/workspace/train1/pre-trained-model/model.ckpt'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100',f'{dir_path}/Tensorflow/workspace/train1/annotations/train.record'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010',f'{dir_path}/Tensorflow/workspace/train1/annotations/test.record'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt',f'{dir_path}/Tensorflow/workspace/train1/annotations/label_map.pbtxt'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('num_steps: 200000',f'num_steps: {args["steps"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('num_examples: 8000',f'num_examples: {args["eval"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('score_threshold: 0.0',f'score_threshold: {args["score"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('iou_threshold: 0.7',f'iou_threshold: {args["iou"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_coco.config', inplace=True) as file:
		for line in file:		
			print(line.replace('iou_threshold: 0.6',f'iou_threshold: {args["iou"]}'), end='')

elif model == "faster_rcnn_inception_v2_pets.config":
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:
			print(line.replace('num_classes: 37',f'num_classes: {args["class"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:
			print(line.replace('PATH_TO_BE_CONFIGURED/model.ckpt',f'{dir_path}/Tensorflow/workspace/train1/pre-trained-model/model.ckpt'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/pet_faces_train.record-?????-of-00010',f'{dir_path}/Tensorflow/workspace/train1/annotations/train.record'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/pet_faces_val.record-?????-of-00010',f'{dir_path}/Tensorflow/workspace/train1/annotations/test.record'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt',f'{dir_path}/Tensorflow/workspace/train1/annotations/label_map.pbtxt'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('num_steps: 200000',f'num_steps: {args["steps"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('num_examples: 1101',f'num_examples: {args["eval"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('score_threshold: 0.0',f'score_threshold: {args["score"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('iou_threshold: 0.7',f'iou_threshold: {args["iou"]}'), end='')
	with fileinput.FileInput('Tensorflow/workspace/train1/training/faster_rcnn_inception_v2_pets.config', inplace=True) as file:
		for line in file:		
			print(line.replace('iou_threshold: 0.6',f'iou_threshold: {args["iou"]}'), end='')	

os.system(f'cp Tensorflow/workspace/train1/training/{args["model"]} Tensorflow/workspace/train1/pre-trained-model/{args["model"]}')

start = time.time() 

os.system(f'python3 Tensorflow/workspace/train1/train.py --logtostderr --train_dir={dir_path}/Tensorflow/workspace/train1/training/ --pipeline_config_path={dir_path}/Tensorflow/workspace/train1/training/{args["model"]}')

end = time.time()
print('Total training time is: ')
print(end-start)

print('Freezing the graph and saving the model...')
os.chdir('Tensorflow/models/research/object_detection')
os.system(f'''python3 -u export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path={dir_path}/Tensorflow/workspace/train1/training/{args["model"]} \
  --trained_checkpoint_prefix={dir_path}/Tensorflow/workspace/train1/training/model.ckpt-{args["steps"]} \
  --output_directory={dir_path}/trained-model{args["steps"]}''')
os.chdir('../../../../')

os.system('mkdir servablemodel servablemodel/1')
os.system(f'cp -rf trained-model{args["steps"]}/saved_model/* servablemodel/1/')
print('Hosting the model on the server...')
os.system('/tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=fastRCNN --model_base_path=/tensorflow-serving/objectdetection/servablemodel &> servablemodel_log &')

print('Evaluation on the test data:\n')
os.system(f'python3  Tensorflow/workspace/train1/eval.py --logtostderr  --checkpoint_dir={dir_path}/Tensorflow/workspace/train1/training --eval_dir={dir_path}/Tensorflow/workspace/train1/evaluation/ --pipeline_config_path={dir_path}/Tensorflow/workspace/train1/training/{args["model"]}')
