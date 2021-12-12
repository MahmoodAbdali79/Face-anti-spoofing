# Face-anti-spoofing

Face anti spoofing is a challenging task in computer vision. It is the task of preventing false facial verification by using a photo, video, mask or a different substitute for an authorized person's face. in other word we're finding liveness in films that captured from humans. For this task, lots of models have been provided and we selected [one](https://github.com/emadeldeen24/face-anti-spoofing) of them to retrain on dataset and getting better result.

## origin datasets

We use OULU-NPU as our training dataset to reach a better result .This dataset contains two types of attack presentation ( `print-attack` and `reply-attack`)  and real film in some series with two different backgrounds. more details about this dataset -> [here](https://sites.google.com/site/oulunpudatabase/).

## functions

`frames_`: this function gets all the frames of a video (used for train dataset)

  the output frames will be in train/1/image/{name} .

`eval_frames`:  this function select some frames from a frame file, add the face coordinates

   and name of the video as the name of the file and save it in test/frame for

   further processes. 

`get_rppg_pred`:   this function is used to return the RPPG predictions of images.

`make_pred`: this function predict a single image recording to its RPPG prediction and the image itself.

`images_pred`: this function predict frames of a specific video.

`images_evaluate`: this function is used to evaluate some images in a specific directory.

`get_generator`: this is the main generator that is used to train the model

  the files in train/1 are used to train the model.

`prediction_test`: this function simply returns prediction of frames in the test/frame directory.

`labels_test`: this function simply returns labels of frames in test/frame directory.

## load model

For loading models and their weights, we use the following code. following script is used for loading the model, loading weights and compiling the model.

```python
# load model
json_file = open('../RGB_rPPG_merge_softmax_.json', 'r')  
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#load weights and compile
model.load_weights("../RGB_rPPG_merge_softmax_.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## result of models on the origin dataset

As you can see below, the results show that our models accuracy has been improved on the test data.

|          Model          | HTER  |
| :---------------------: | :---: |
| RGB_rPPG_merge_softmax_ | 0.142 |
|    Balance_30_300_7     | 0.099 |
|      Test_30_180_1      | 0.102 |
