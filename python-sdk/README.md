# Python Data Science Integration SDK for Fusion 5.0

## Overview

With Fusion 5.0, data scientists and machine learning engineers can deploy end-user trained Python machine learning models to Fusion, offering real-time prediction and seamless integration with query and index pipelines.  

Example use cases include:

* Using SpaCy to extract named entities and indexing results into a Solr collection
* Using a Keras model to perform query intent classification at query-time
* Use pretrained word embeddings to generate synonyms for a query

Benefits include:

* Extension points for data scientists to plug in customized Python modeling code
* Client libraries to ease the development and testing of Python plugins
* API driven and dynamic, runtime loading and updating of plugins

## User Guide

The following is a guide for how a data scientist can deploy his/her own models to Fusion.

Prerequisites:

* Running instance of Fusion 5.0 with AI license

* Local development environment with Python 3.6.x on Mac OS or Linux

The high level steps required to deploy a model to Fusion are:

1. Install the Data Science Integration SDK locally
2. Train a custom model
3. Create and test plugin
4. Bundle plugin as a zip file
5. Testing and Deployment

### Install the Data Science Integration SDK locally

The Data Science Integration (DSI) SDK includes helper libraries that will streamline the process of training local models and deploying them to Fusion.  To install the SDK:

1. Clone the repository:

   ```
   % git clone https://github.com/lucidworks/fusion-data-science-toolkit.git
   ```

2. Create and activate a virtual environment using virtualenv or pyenv

3. Install pip dependencies:

   ```
   % cd python-sdk
   % pip install -r requirements.txt
   ```

4. Optionally, if you'd like to use Jupyter to train your models:

   ```
   % pip install jupyter
   ```

### Train a custom model

Train a machine learning model using the virtual environment in any tool of your choice (i.e. Jupyter).

The `requirements.txt` included with this SDK contains all the libraries that are available in Fusion's Machine Learning Service's runtime environment. This environment includes the most popular libraries used by data scientists today (i.e. scikit-learn, Tensorflow, XGBoost, etc.). Restricting your training and inference code to these libraries will ensure that your model will work with Fusion without any additional setup steps.

If you need to use an additional library that is not available, you'll need to create a custom `Dockerfile` which extends the ML service's base image, and add the additional commands necessary to install additional pip dependencies.  Refer to Appendix A for more detailed instructions.

### Create and test plugin

To create a plugin, you will need to create a file called `predict.py` that contains these two functions:

* `def init(bundle_path: str)`
  * This function is called by the ML service when the model is invoked for the first time.  Place one-time initialization here, like loading a serialized model from disk
  * `bundle_path` is the path to the unzipped bundle. 
* `def predict(model_input: dict) -> dict`
  * This function contains the code necessary to generate a prediction from a single input.
  * Single input parameter `model_input` is a `dict` representing the input to your model
  * Returns a `dict` of (key, value) pairs, representing model output. Dictionary keys must be `str`, an dictionary values must be one of the following types:
    *  `numbers.Number` (`float`, `int`, etc.)
    * `str`
    * `list` or `ndarray` of `str`
    * `list` or `ndarray` of `numbers.Number`

Here's a simple example of a model that simple outputs the input:

```python
def init(bundle_path: str):
    """
    One-time initialization here.  For example, loading a serialized model from disk.
    Any objects created here will need to be made module global in order for it to be
    accessible by predict(), i.e. global my_keras_model
    
    :param bundle_path: Path to unzipped bundle, used to construct file path to bundle 
    contents, i.e. os.path.join(bundle_path, "model.pkl")
    """
    print("Initializing the model!")
    
def predict(model_input: dict) -> dict: 
    """
    Generate prediction.
    
    Return value is a dict where keys must be `str`, and values must be one of the following types:
      - `numbers.Number` (`float`, `int`, etc.)
      - `str`
      - `list` or `ndarray` of `str`
      - `list` or `ndarray` of `numbers.Number`
    
    :param model_input: a dict containing model input
    :return: model output dict. 
    """
    if 'input' not in model_input:
        raise ValueError("Input must contain the key 'input'")
    
    return {
        "output": model_input['input']
    }
```



Here's an example of wrapping a simple sentiment analysis Keras model:

```python
import os
import pickle
from keras.models import load_model
from keras import preprocessing
import keras

INPUT_LENGTH = 500

def init(bundle_path: str):
    """
    One-time initialization here.  For example, loading a serialized model from disk.
    Any objects created here will need to be made module global in order for it to be
    accessible by predict(), i.e. global my_keras_model
    
    :param bundle_path: Path to unzipped bundle, used to construct file path to bundle 
    contents, i.e. os.path.join(bundle_path, "model.pkl")
    """
    global tokenizer
    keras.backend.clear_session()
    with open(os.path.join(bundle_path, 'tokenizer.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)
        global model
        model = load_model(os.path.join(bundle_path, 'sentiment.h5'))
    
def predict(model_input: dict) -> dict: 
    """
    Generate prediction.
    
    Return value is a dict where keys must be `str`, and values must be one of the following types:
      - `numbers.Number` (`float`, `int`, etc.)
      - `str`
      - `list` or `ndarray` of `str`
      - `list` or `ndarray` of `numbers.Number`
    
    :param model_input: a dict containing model input
    :return: model output dict. 
    """
    if 'input' not in model_input:
        raise ValueError("Required field 'input' not defined.")

    samples = [ model_input['input'] ]
    idx_sequence = tokenizer.texts_to_sequences(samples)
    padded_idx_sequence = preprocessing.sequence.pad_sequences(idx_sequence, maxlen=INPUT_LENGTH)

    y = model.predict(padded_idx_sequence)
    label = "positive" if y[0][0] > 0.5 else "negative"

    return {
        "sentiment": label,
        "score": y[0][0]
    }
```

### Packaging plugin

To create a model bundle, simply create a zip file containing `predict.py` and dependent serialized objects. For example, if your current working directly contains:

```
predict.py
tokenizer.pickle
sentiment.h5
```

On a Mac, you can create a zip using:

```
zip -r /path/to/model.zip .
```

### Testing and Deployment

The following requires the `fusion-machine-learning-client` library installed in your virtualenv.

#### Testing Locally

The client libraries contain the `LocalBundleRunner`, which runs your plugin in your local interpreter.  This allows you to quickly test and debug your plugin locally without needing to interact with Fusion.

Usage:

```python
from lucidworks.ml.sdk import LocalBundleRunner

runner = LocalBundleRunner("/path/to/model.zip")
output = runner.predict({
	"input": "Hello World!"
})
```

class `LocalBundleRunner(bundle_zip)` - Loads the bundle zip file and invokes `predict.py#init()`

`LocalBundleRunner.predict(model_input)` - Calls `predict.py#predict()`and returns model output `dict`, ensuring that model output satisfies datatype requirements.

If your model produces expected output without any errors, you are ready to deploy your model to Fusion.

#### Deploy model bundle to Fusion

Use `MLServiceSDKFusionClient` to deploy and test your model in Fusion.  The `model_id` is simply a unique ID you assign to your model which you use to reference when interacting with the model.

```python
from lucidworks.ml.sdk import MLServiceSDKFusionClient
from requests.auth import HTTPBasicAuth

model_id = 'echo'
app_name = '<Fusion App Name>'
fusion_api_url = '<Fusion API URL>'

client = MLServiceSDKFusionClient(fusion_api_url, 
                                  app_name,
                                  auth=HTTPBasicAuth('<Username>', '<Password>'))
```

To upload your model to Fusion:

```
client.upload_model('/path/to/model.zip', model_id)
```

To test your uploaded model using Fusion:

```python
output = client.predict({
	"input": "Hello World!"
})
```

**IMPORTANT**: The  `MLServiceSDKFusionClient.predict()` function in intended only for development and testing purposes, not for production use.  To use your model in production, use in conjunction with query and index pipelines.

If your model produces expected output without any errors, congratulations, you have successfully deployed your model to Fusion!  You can integrate your model with query and index pipelines using the Machine Learning stages.

## Appendix A: Building a custom ML service image

In order to use Python libraries that are not included in the default ML service image, you will need to build a custom image that includes commands to install these additional libraries.  In order to do this, you will need:

* Docker installed in a local build machine
* Docker Hub account with read access to official Lucidworks Fusion images
* Write access to a Docker registry to publish your custom image

### Create Image

First, create a Dockerfile based off this template:

```dockerfile
FROM lucidworks/fusion-ml-python-image:5.0

USER root

# Add any additional commands here, i.e. pip install gensim

USER 8764
```

Login to Docker Hub with your credentials:

```
docker login
```

Build the image:

```
docker build -t my-org/fusion-ml-python-image .
```

Publish the image to Docker Hub or your internal Docker registry.

