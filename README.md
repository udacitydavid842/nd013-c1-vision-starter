# Object detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

## Structure

The data in the classroom workspace will be organized as follows:
```
/home/backups/
    - raw: contained the tf records in the Waymo Open format. (NOTE: this folder only contains temporary files and should be empty after running the download and process script)

/home/workspace/data/
    - processed: contained the tf records in the Tf Object detection api format. (NOTE: this folder should be empty after creating the splits)
    - test: contain the test data
    - train: contain the train data
    - val: contain the val data
```

The experiments folder will be organized as follow:
```
experiments/
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training
    - experiment0/....
    - experiment1/....
    - experiment2/...
    - pretrained-models/: contains the checkpoints of the pretrained models.
```

## Prerequisites

### Local Setup

For local setup if you fhave your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. However, you will need to login to Google Cloud using the following command:
```
gcloud auth login
```
This command will display a link that you need to copy and paste to your web browser. Follow the instructions. You can check if you are logged correctly by running :
```
gsutil ls gs://waymo_open_dataset_v_1_2_0_individual_files/
```
It should display the content of the bucket.

## Instructions

### Download and process the data

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you will need to implement the `create_tf_example` function. This function takes the components of a Waymo Tf record and save them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file. 

Once you have coded the function, you can run the script at using
```
python download_process.py --data_dir /home/workspace/data/ --temp_dir /home/backups/
```

You are downloading XX files so be patient! Once the script is done, you can look inside the `/home/workspace/data/processed` folder to see if the files have been downloaded and processed correctly.


### Exploratory Data Analysis

Now that you have downloaded and processed the data, you should explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation). 


### Create the splits

Now you have become one with the data! Congratulations! How will you use this knowledge to create the different splits: training, validation and testing. There are no single answer to this question but you will need to justify your choice in your submission. You will need to implement the `split_data` function in the `create_splits.py` file. Once you have implemented this function, run it using:
```
python create_splits.py --data_dir /home/workspace/data/
```

NOTE: Keep in mind that your storage is limited. The files should be <ins>moved</ins> and not copied. 

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf). 

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `training/pretrained-models/`. 

Now we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/processed/train/ --eval_dir /home/workspace/data/processed/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Create a folder `training/reference`. Move the `pipeline_new.config` to this folder. You will now have to launch two processes: 
* a training process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
* an evaluation process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```

NOTE: both processes will display some Tensorflow warnings.

To monitor the training, you can launch a tensorboard instance by running `tensorboard --logdir=training`. You will report your findings in the writeup. 

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup. 

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it. 


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:
```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/reference/experiment2/pipeline.config --trained_checkpoint_dir training/reference/experiment2/ckpt-50 --output_directory training/experiment0/exported_model/
```

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py -labelmap_path label_map.pbtxt --model_path training/reference/experiment2/exported_model/saved_model --tf_record_path /home/workspace/data/processed/test/segment-9758342966297863572_875_230_895_230_with_camera_labels.tfrecord --config_path training/reference/experiment2/pipeline_new.config --output_path animation.mp4
```

## Submission Template

### Project overview
In this project, we are building a model that would help a self-driving car better predict and detect objects (cars, pedestrians, and cyclists). 

We are made available the [Waymo Open dataset](https://waymo.com/open/) and we use 100 tf record files on which we would develop a model to train on then. Object detection systems are very important for self-driving cars so that they can correctly determine which object surrounds them and make circulation safer.

At the end of this project, we create a [video](./animation.mp4) of the model's inferences for a tf record file `home/workpace/data/processed/test/segment-12027892938363296829_4086_280_4106_280_with_camera_labels.tfrecord`. 

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.
To set up and run this project, we make sure we use a system equipped with Nvidia GPU running with the latest drivers, install tensorflow-2.3.0-gpu, python 3.6 for the basic setup. 

#### Environment set up
Assuming you will be using an Nvidia gpu enabled system,

- We clone this project [repository](https://github.com/udacitydavid842/nd013-c1-vision-starter.git) from github into your local system.
- Navigagte to the repo's root dir from your terminal.
- You can use the provided Dockerfile and requirements in the [build directory](./build) to build a docker image.
- Run the Dockerfile with the command (run this command from the within the [build directory](./build)):
```
docker build -t project-dev -f Dockerfile.gpu .
```
- Create a docker container to run the created docker image: 
```
docker run -v C:\Users\mypc\nd013-c1-vision-starter\:/app/project/ -p 8888:8888 -p 6006:6006 --shm-size=16gb -ti project-dev bash
```
- Once in container, you will need to install gsutil, which you can easily do by running:
```
curl https://sdk.cloud.google.com | bash
```
- Once gsutil is installed and added to your path, you can auth using:
```
gcloud auth login
```
- We ensure to downgrade/upgrade tensorflow-gpu to version 2.3.0:
```
pip3 install tensorflow-gpu==2.3.0
```
- The project works with numpy v.1.17
```
pip install --upgrade numpy==1.17
```
- install python3-tk, which is compactible version for this project
```
apt-get install python3-tk
```
- install seaborn for data visualition
```
pip install seaborn
```

#### File Structure
```
/home/backups/
    - raw: contained the tf records in the Waymo Open format. (NOTE: this folder only contains temporary files and it is emptied after running the download and process [script](./download_process.py) )
```
```
/home/workspace/data/
    - processed: contained the tf records in the Tf Object detection api format. (NOTE: this folder will contain the train, val, and test data folders after creating the splits)
```
```
/home/workspace/data/processed/
    - test: contain the test data
    - train: contain the train data
    - val: contain the val data
```
```
experiments/
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training
    - pretrained-models/: contains the checkpoints of the pretrained models.
```
```
training/
    - reference/
        - experiment1/...: should contain the pipeline config file and information for the training of the model without augmentation.
        - experiment2:/...: should contain the pipeline config file with modifications and information for the training of the model with augmentation.
    - pretrained-model/: should contain the checkpoints of the pretrained models.
```
```
solution/
    - pipeline_new.config: new config file with modifications
```

### Dataset
#### Dataset analysis

The provided data contains images with objects (cars, pedestrians, and cyclists) and we have to annotate. We need to properly explore the data such that we can know how to correctly split our data into training and validation sets to minimize the test error bias. 

The dataset contains images with different variations, clear images, images in bad weather conditions, images at night, images with distant objects, etc... Below are a few images we plot with colored bounding boxes around objects such as (vehicles - red, pedestrians - blue and cyclist green)

![img1](screenshots/img1.png)![img2](screenshots/img2.png)![img3](screenshots/img3.png)![img4](screenshots/img4.png)![img5](screenshots/img5.png)![img6](screenshots/img6.png)![img7](screenshots/img7.png)![img8](screenshots/img8.png)![img9](screenshots/img9.png)![img10](screenshots/img10.png)

Further analysis of the dataset shows that most images contain vehicles and pedestrians (majority vehicles), and very few sample images have cyclists in them. The chart below shows the distribution of classes (cars, pedestrians, and cyclists), over a collection of 100 random images in the dataset.``        

![distribution](screenshots/chart.png)

We can see these visulizations in the `Exploratory Data Analysis.ipynb` file.

#### Cross validation
Here are dataset consists of 100 tfrecord files. We split them up into training, validation, and testing sets. 

We give our training set 75% of the data, 15% for validation, and the remaining 10% for testing. This splitting is such that we have enough data for training as well as reserve data for test and validation. Since we have just 100 tfrecord files to deal with we need to minimize the test error and overfitting, thus we use 75% of the data for training so that we can have 15% for cross-validation which is a good number in this case.

### Training 
#### Reference experiment
I performed the training over a GPU with 8 cores thus I used a batch size of 8 and validation was run along side the training but over the avalable CPU cores. I first of all ran the training and validation based on the configurations without augmentation of the Restnet50 [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and got the folowing results.

![loss](screenshots/exprt1-loss.JPG)
training loss

The loss charts as shown in the image above shows that the model is overfitting as the validation loss (in blue) does not generalize well with the training loss (in orange). This is observed as the blue lines are clearly above the orange lines, showing that the error in classifying is high.

![presision](screenshots/exprt1-precision.JPG)
Precision

![recall](screenshots/exprt1-recall.JPG)
Recall

We observe from the Databoxes_precision and Databoxes_recall charts above that the values  for precision and recall are low and increases slowly as with increasing training steps. 

To conclude, the performance for this algorithm is generally poor and will need some modifications to improve on the model.

#### Improve on the reference

To improve on this model, The next experimet I ran was with augmetation by adding augmentatinos to the data such as, converting the image to gray scale with a probability of 2%, setting the contrast of the image with a min_delta of 0.6 and a max_delta of 1.0, again, I adjusted the brightness of the image  with a max_delta of 0.3. This augmentations are reflected in the pipline configuration file `solution/pipeline_new.config`.

I used the configuration to run the notebook `Explore augmentations.ipynb`. I was able to observer its performance on the following images

![gray scale](screenshots/grayscale.PNG)
Gray scale image

![foggy](screenshots/foggy.PNG)
Foggy weather

![night](screenshots/night.PNG)
Night

![bright](screenshots/bright.PNG)
Bright

![bright](screenshots/contrast.PNG)
High Contrast

We get the the following charts below after training the model with the new augmentations.
![loss2](screenshots/exprt2-loss.JPG)
training loss with augmentation

![presision2](screenshots/exprt2-precision.JPG)
Precission with augmemtaion

![recall2](screenshots/exprt2-recall.JPG)
recall with augmentation

Generally, the loss of the modified model is lower that of the original model, this shows that it is performing better. Thus to improve on the model further we should train with additional images with varying brightness and contrast and also converting to gray scale is necessary.

![joinned loss](screenshots/joined_loss.JPG)
The image above shows the combination of the training/validation loss of the experiment without augmentation (experiment1) and the training/validation loss for the experiment with augmentation. 

Finally, we could improve generally by adding more data that has a reasonable amount of cyclist to pedestrians and vehicles ratio so that the training will not be biased to only vehicle objects. Again it is difficult to recognize distant objects thus this will be a challenge as even the human eye can not recognize objects a mile away. Find model inference video [here](./animation.mp4)

### Running Project
With the project setup as describedin the ***Environment set up*** section above, 

- 1: navigate to the project root folder from your docker container
- 2: download an process data:
```
python download_process.py --data_dir /home/workspace/data/ --temp_dir /home/backups/
```
- 3: run all cells on the `Exploratory Data Analysis.ipynb` notebook.
- 4: create splits of the data for training, validation, and testing:
```
- python create_splits.py --data_dir /home/workspace/data/processed/
```
- 5: download download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to training/pretrained-models/.
- 6: Edit [pipeline.config](./pipeline.config) file.:
```
python edit_config.py --train_dir /home/workspace/data/processed/train/ --eval_dir /home/workspace/data/processed/val/ --batch_size 8 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
- 7: Run training for non augmentation pipeline:
```
python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
- 8: Run validation on a separate terminal just after launch the training:
```
CUDA_VISIBLE_DEVICES="" python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```
- 9: while training is going on, launch tensorboard from another terminal:
```
tensorboard --logdir=training/reference/ --host 0.0.0.0
```
- 10: view tensorboard dashboard from your browser:
```
localhost:6006
```
- 11: when the training/validation is done, kill the terminals
- move the data in `training/reference` to `training/reference/experiment1`
- 12: modify the `pipeline_new.config` cereated in step 6 with more augmentations

![augmented](screenshots/augmentation.JPG)
- repeat steps 7 and 8
- 13: move the data in `training/reference` to `training/reference/experiment2` 
- 14: observe performance on tensorboard
- 15: save the new model:
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/experiment2/pipeline_new.config --trained_checkpoint_dir training/reference/experiment2 --output_directory training/refrence/experiment2/exported_model/
```
- create a video of your model's inferences for any tf record file in the test data directory.
```
python inference_video.py -labelmap_path label_map.pbtxt --model_path training/experiment2/exported_model/saved_model --tf_record_path /home/workspace/data/processed/test/segment-9758342966297863572_875_230_895_230_with_camera_labels.tfrecord --config_path training/reference/experiment2/pipeline_new.config --output_path animation.mp4
```
