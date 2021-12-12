# benchmark
After train models on main dataset and getting good result , it's time to test our models on an other data base that surly has difference than previous dataset. so we did this on  [CASIA FASD](https://ieeexplore.ieee.org/document/6199754).  it's a brelient dataset for face anti spoofing with good variety on types of spoofing attack presentation.

## functions

In the `evaluation.ipynb` there are some functions for extracting and predict frames 

- #### extraction

  . So `extract_frames`  extract frames (each 5 frames) from videos in this format: `numberfolder_filemname_x,y,w,h_label.jpg` . 

  `ExtractFrameBasedOnClass` and `extract_frame` are used for extracting frames based on classes.

  And also `seprate_HR` extract frames based on high rezolution and low rezolution. 

  

- #### prediction

   `prediction_test` is used for  predict frames that was created by `extract_frames`.

  We have some frames frame that extracted based on types of attacks and for predict them `PredictSepratedClass` is used .

  `prediction_HRNOHR_test` predicts frames extracted by `seprate_HR`

And there are some basic functions that are used for basic operation that explained in [here](https://github.com/MahmoodAbdali79/Face-ani-spoofing#functions).

## load model

For loading models and their weight, we use the following code. So here can see code of loading model, load weight and compile them.

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
## result of models on origin dataset

After predict models on each types of extracted frames, we found out that our model isn't very good on this benchmark. actually it's as worse as first model on benchmark and we found some reason for this    

- ### why this happened?

  The main reason refers to differences between these tow dataset. benchmark involves an other type attack (eye blink attack) that is not in the train dataset.

  an other reason refers to resolution of benchmark that divided to high resolution and low resolution. Also cameras were used in benchmark is different from train dataset cameras.

  and every difference  between these dataset could be effective that you can find them in detail : 
  > main dataset : [OULU-NPU](https://sites.google.com/site/oulunpudatabase/)
  > benchmark : [CASIA FASD](https://ieeexplore.ieee.org/document/6199754)

## how can improve model

> The cure for pain is in the pain.  **Rumi**

actually, because the benchmark dataset has more variety than train dataset, it's good idea to train  benchmark as train dataset. or every way that complain the train dataset in variety.

As mentioned above about camera, using same camera for both train dataset and benchmark could be helpful. And  also using newer `RPPG` and better face detection model could be helpful for having better result. 
