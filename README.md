# prednet

Code and models accompanying [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104) by Bill Lotter, Gabriel Kreiman, and David Cox.

The PredNet is a deep recurrent convolutional neural network that is inspired by the neuroscience concept of predictive coding (Rao and Ballard, 1999; Friston, 2005).
**Check out example prediction videos [here](https://coxlab.github.io/prednet/).**

The architecture is implemented as a custom layer<sup>1</sup> in [Keras](http://keras.io/).
Code and model data is now compatible with Keras 2.0.
Specifically, it has been tested on Keras 2.1.3 with Theano 0.9.0, Tensorflow 1.5, and Python 3.
The provided weights were trained with the Theano backend.
For previous versions of the code compatible with Keras 1.2.1, use fbcdc18.
To convert old PredNet model files and weights for Keras 2.0 compatibility, see ```convert_model_to_keras2``` in `keras_utils.py`.
<br>

This is the package that is compatible with IBM's DLaaS.

## DLaaS Setup
Complete the [one-time setup tutorial](https://github.com/mypublicorg/DLaaS-Getting-StartedTutorial/blob/master/onetimesetup.md). Our tutorial steps below are heavily based on the [DLaaS demo tutorial](https://github.com/mypublicorg/DLaaS-Getting-StartedTutorial/blob/master/demo.md).

## KITTI Demo

Code is included for training the PredNet on the raw [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset.
We include code for downloading and processing the data, as well as training and evaluating the model.
The preprocessed data and can also be downloaded directly using `download_data.sh` and the **trained weights** by running `download_models.sh`.
The model download will include the original weights trained for t+1 prediction, the fine-tuned weights trained to extrapolate predictions for multiple timesteps,  and the "L<sub>all</sub>" weights trained with an 0.1 loss weight on upper layers (see paper for details).

### Steps
1. **Download/process data**
	```bash
	$ ./download_data.sh
	```
	Execute download_data.sh. This will download processed data (~3 GB) and dump it into the correct format using Pickle.
	<br>
	<br>
	
2. **Upload the dataset to your bucket**

	Assign the name of the bucket to a shell variable
	```
	$ bucket_name=<your_bucket_name>
	$ bxaws s3 cp kitti_data/  s3://$bucket_name/kitti_data/ --recursive
	```
	(optional) Verify that the data was successfully uploaded using this comand.

	```
	$ bxaws  s3 ls s3://$bucket_name/kitti_data/
	```
3. **Edit your manifest file, `tensorflow-prednet-kitti.yml`**

	This yaml file should hold all the information needed for executing the job, including what bucket, ml framework, and computing instance to use.

4. **Update the template manifest:**

	Edit `tensorflow-prednet-kitti.yml`:
	Add your author info and replace the values of `aws_access_key_id`, `aws_secret_access_key`, and `bucket` 
	with your storage instance credentials and bucket name.
	This should be done for both the data input reference (`training_data_reference`) 
	and the output reference (`training_results_reference`). 
	Notice that you may use the same bucket for both input and output, but this is not required.

```yaml
model_definition:
  framework:
#framework name and version (supported list of frameworks available at 'bx ml list frameworks')
    name: tensorflow
    version: 1.5
#name of the training-run
  name: MYRUN
#Author name and email
  author:
    name: JOHN DOE
    email: JOHNDOE@EMAIL.COM
  description: This is running kitti training on the prednet model
  execution:
#Command to execute -- see script parameters in later section !!
    command: python3 kitti_train.py; python3 kitti_evaluate.py
    compute_configuration:
#Valid values for name - k80/k80x2/k80x4/p100/p100x2/v100/v100x2
      name: v100x2
training_data_reference:
  name: training_data_reference_name
  connection:
    endpoint_url: "https://s3-api.us-geo.objectstorage.service.networklayer.com"
    aws_access_key_id: < YOUR SAVED ACCESS KEY >
    aws_secret_access_key: < YOUR SAVED SECRET ACCESS KEY >
  source:
    bucket: < mybucketname >
  type: s3
training_results_reference:
  name: training_results_reference_name
  connection:
    endpoint_url: "https://s3-api.us-geo.objectstorage.service.networklayer.com"
    aws_access_key_id: < YOUR SAVED ACCESS KEY >
    aws_secret_access_key: < YOUR SAVED SECRET ACCESS KEY >
  target:
    bucket: < mybucketname >
  type: s3
```

Notice that under `execution` in the yaml file, we specified a command that will be executed 
when the job starts execution at the server. (make sure you give right path to data)

```
python3 kitti_train.py; python3 kitti_evaluate.py
```

This will train a PredNet model for t+1 prediction.
	**To download pre-trained weights**, run `download_models.sh`
	<br>
The evaluation will output the mean-squared error for predictions as well as make plots comparing predictions to ground-truth.

5. **Zip all the code and models into a .zip file:**
	```
	$ zip model.zip ./*
	```
6.**Send your code and manifest to IBM Watson Studio:** <br>
	```
	$ bx ml train model.zip tensorflow-prednet-kitti.yml
	```
	The command should generate a training ID for you, meaning the prednet model has started training on Watson!

7. **Monitor the training**
	We can check the status of all training using the command:
	```
	$ bx ml list training-runs
	```
	Continuously monitor a training run by using the `bx ml monitor` command:
	```
	$ bx ml monitor training-runs < trainingID >
	```
### Feature Extraction
Extracting the intermediate features for a given layer in the PredNet can be done using the appropriate ```output_mode``` argument. For example, to extract the hidden state of the LSTM (the "Representation" units) in the lowest layer, use ```output_mode = 'R0'```. More details can be found in the PredNet docstring.

### Multi-Step Prediction
The PredNet argument ```extrap_start_time``` can be used to force multi-step prediction. Starting at this time step, the prediction from the previous time step will be treated as the actual input. For example, if the model is run on a sequence of 15 timesteps with ```extrap_start_time = 10```, the last output will correspond to a t+5 prediction. In the paper, we train in this setting starting from the original t+1 trained weights (see `kitti_extrap_finetune.py`), and the resulting fine-tuned weights are included in `download_models.sh`. Note that when training with extrapolation, the "errors" are no longer tied to ground truth, so the loss should be calculated on the pixel predictions themselves. This can be done by using ```output_mode = 'prediction'```, as illustrated in `kitti_extrap_finetune.py`.

### Additional Notes
When training on a new dataset, the image size has to be divisible by 2^(nb of layers - 1) because of the cyclical 2x2 max-pooling and upsampling operations.

<br>

<sup>1</sup> Note on implementation:  PredNet inherits from the Recurrent layer class, i.e. it has an internal state and a step function. Given the top-down then bottom-up update sequence, it must currently be implemented in Keras as essentially a 'super' layer where all layers in the PredNet are in one PredNet 'layer'. This is less than ideal, but it seems like the most efficient way as of now. We welcome suggestions if anyone thinks of a better implementation.  
