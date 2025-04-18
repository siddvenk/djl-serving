# DJL Serving Operation Modes

## Overview

DJL Serving is a high-performance serving system for deep learning models. DJL Serving supports models with:

1. [Python Mode](#python-mode)
2. [Java Mode](#java-mode)
3. [Binary Mode](#binary-mode)

Also see the options for [model configurations](configurations_model.md).

## Python Mode

This section walks through how to serve Python-based model with DJL Serving.

### Define a Model


To get started, implement a python source file named `model.py` as the entry point. DJL Serving will run your request by invoking a `handle` function that you provide. The `handle` function should have the following signature:

**For Python Sync Mode:**

Python sync mode is the default, and can be controlled with the configuration `option.async_mode=<false|true>`.
Omitting `option.async_mode`, or setting `option.async_mode=false` will run the Python engine in sync mode.

```
def handle(inputs: Input) -> Optional[Output]:
```

**For Python Async Mode:**

Python async mode allows inference handlers to leverage asyncio. 
This is useful for writing handler code at the single-request level and leveraging asyncio to handle multiple concurrent requests.

To enable async mode, you must set `option.async_mode=true` and define a `model.py` file with a `handle` function. The `handle` function must have the following signature:

```
async def handle(inputs: Input) -> Optional[Output]:
```

If there are other packages you want to use with your script, you can include a `requirements.txt` file in the same directory with your model file to install other dependencies at runtime. 
A `requirements.txt` file is a text file that contains a list of items that are installed by using pip install. You can also specify the version of an item to install.

If you don't want to install packages over the network at runtime, you can bundle the package wheel in the model directory and install the package from model directory:

```
./local_wheels/ABC-0.0.2-py3-none-any.whl
```

### Packaging

DJL Serving supports model artifacts in model directory, .zip or .tar.gz format.

To package model artifacts in a .zip:

```
cd /path/to/model
zip model.zip *
```

To package model artifacts in a .tar.gz:

```
cd /path/to/model
tar -czvf model.tar.gz *
```

### Serving Example

Let's run an example where we load a model in Python mode and run inference using the REST API.

#### Step 1: Define a Model

In this example, we will use the resnet18 model in the djl-demo repo.

The example provides a [model.py](https://github.com/deepjavalibrary/djl-demo/blob/master/djl-serving/python-mode/resnet18/model.py) that implements a `handle` function.

```
def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
```

It also provides a `requirements.txt` that loads torchvision 0.12.0:

```
torchvision==0.12.0
```

To get this model, clone the djl-demo repo if you haven't done so yet. Then, we can package model artifacts in .zip or .tar.gz.

```
git clone https://github.com/deepjavalibrary/djl-demo.git
cd djl-demo/djl-serving/python-mode/
zip -r resnet18.zip resnet18
```

#### Step 2: Start Server

Next, start DJL Serving and load this model at startup.

##### Linux/macOS

```
djl-serving -m resnet::Python=file://$PWD/resnet18.zip
```

Or we can load directly from model directory:

```
djl-serving -m resnet::Python=file://$PWD/resnet18
```

##### Windows

```
path-to-your\serving.bat -m "resnet::Python=file:///%cd%\resnet18.zip"
```

Or we can load directly from model directory:

```
path-to-your\serving.bat -m "resnet::Python=file:///%cd%\resnet18"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `resnet` with the model.

After the model is loaded, we can start making inference requests.

#### Step 3: Inference

To query the model using the prediction API, open another session and run the following command:

##### Linux/macOS

```
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST "http://127.0.0.1:8080/predictions/resnet" -T "kitten.jpg"
```

On Windows, you can just download the image and use `Postman` to send POST request.

This should return the following result:

```json
[
  {
    "tabby":0.4552347958087921,
    "tiger_cat":0.3483535945415497,
    "Egyptian_cat":0.15608155727386475,
    "lynx":0.026761988177895546,
    "Persian_cat":0.002232028404250741
  }
]
```

## Java Mode

This section walks through how to serve model in Java mode with DJL Serving.

### Translator

The `Translator` is a Java interface defined in DJL for pre/post-processing.

You can use a built-in DJL TranslatorFactory by configuring `translatorFactory` in `serving.properties`.

Or you can build your own custom Translator. Your Translator should have the following signature:

```java
public void prepare(TranslatorContext ctx);

public NDList processInput(TranslatorContext ctx, I input) throws Exception;

public String processOutput(TranslatorContext ctx, NDList list) throws Exception;
```

### Provide Model File

Next, you need to include a model file. DJL Serving supports model artifacts for the following engines:

- MXNet
- PyTorch (torchscript only)
- TensorFlow
- ONNX

You can also include any required artifacts in the model directory. For example, `ImageClassificationTranslator` may need a `synset.txt` file, you can put it in the same directory with your model file to define the labels.

### Packaging

To package model artifacts in a .zip:

```
cd /path/to/model
zip model.zip *
```

To package model artifacts in a .tar.gz:

```
cd /path/to/model
tar -czvf model.tar.gz *
```

### Serving Example (NLP)

Let's run an example where we load a NLP model in Java mode and run inference using the REST API.

#### Step 1: Download Model File

In this example, we will use the HuggingFace Bert QA model.

First, if you haven't done so yet, clone the DJL repo.

DJL provides a [HuggingFace model converter](https://github.com/deepjavalibrary/djl/tree/master/extensions/tokenizers#use-djl-huggingface-model-converter-experimental) utility to convert a HuggingFace model to Java:

```
git clone https://github.com/deepjavalibrary/djl.git
cd djl/extensions/tokenizers
python -m pip install -r src/main/python/requirements.txt
python src/main/python/model_zoo_importer.py -m deepset/bert-base-cased-squad2
```

This will generate a zip file into your local folder:

```
model/nlp/question_answer/ai/djl/huggingface/pytorch/deepset/bert-base-cased-squad2/0.0.1/bert-base-cased-squad2.zip
```

The .zip file contains a `serving.properties` file that defines the `engine`, `translatorFactory` and so on.

```
engine=PyTorch
option.modelName=bert-base-cased-squad2
option.mapLocation=true
translatorFactory=ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory
includeTokenTypes=True
```

#### Step 2: Start Server

Next, start DJL Serving and load this model at startup.

##### Linux/macOS

```
djl-serving -m bert-base-cased-squad2=file://$PWD/bert-base-cased-squad2.zip
```

##### Windows

```
path-to-your\serving.bat -m "bert-base-cased-squad2=file:///%cd%\bert-base-cased-squad2.zip"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `bert-base-cased-squad2` with the model.

#### Step 3: Inference

To query the model using the prediction API, open another session and run the following command:

```
curl -k -X POST http://127.0.0.1:8080/predictions/bert-base-cased-squad2 -H "Content-Type: application/json" \
    -d '{"question": "How is the weather", "paragraph": "The weather is nice, it is beautiful day"}'
```

This should return the following result:

```
nice
```

### Serving Example (CV)

Let's run an example where we load a CV model in Java mode and run inference using the REST API.

#### Step 1: Download Model File

In this example, we will use a PyTorch resnet18 model.

```
curl -O https://mlrepo.djl.ai/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/resnet18.zip
```

The .zip file contains a `serving.properties` file that defines the `engine`, `translatorFactory` and so on.

```
application=cv/image_classification
engine=PyTorch
option.modelName=resnet18
width=224
height=224
centerCrop=True
applySoftmax=true
option.mapLocation=true
translatorFactory=ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory
```

#### Step 2: Start Server

Next, start DJL Serving and load this model at startup.

##### Linux/macOS

```
djl-serving -m resnet::PyTorch=file://$PWD/resnet18.zip
```

##### Windows

```
path-to-your\serving.bat -m "resnet::PyTorch=file:///%cd%\resnet18.zip"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `resnet` with the model.

#### Step 3: Inference

To query the model using the prediction API, open another session and run the following command:

##### Linux/macOS

```
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST "http://127.0.0.1:8080/predictions/resnet" -T "kitten.jpg"
```

This should return the following result:

```json
[
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.5183261632919312
  },
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.1956063210964203
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.1955675184726715
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.03224767744541168
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.02553771249949932
  }
]
```

### Serving Example (Custom Translator)

Let's run an example where we load a model in Java mode and run inference using the REST API.

#### Step 1: Download Model File

In this example, we will use a PyTorch resnet18 model.

```
mkdir resnet18 && cd resnet18
curl https://resources.djl.ai/test-models/traced_resnet18.pt -o resnet18.pt
curl -O https://mlrepo.djl.ai/model/cv/image_classification/ai/djl/pytorch/synset.txt
```

#### Step 2: Define a Custom Translator

Next, we need to prepare a custom Translator. In this example, we will use [CustomTranslator.java](https://github.com/deepjavalibrary/djl-demo/blob/master/djl-serving/java-mode/devEnv/src/main/java/CustomTranslator.java) in the djl-demo repo.

We need to copy the Translator to the libs/classes folder.

```
mkdir -p resnet18/libs/classes
git clone https://github.com/deepjavalibrary/djl-demo.git
cp djl-demo/djl-serving/java-mode/devEnv/src/main/java/CustomTranslator.java resnet18/libs/classes
```

#### Step 3: Start Server

Next, start DJL Serving and load this model at startup.

##### Linux/macOS

```
djl-serving -m resnet::PyTorch=file://$PWD/resnet18
```

##### Windows

```
path-to-your\serving.bat -m "resnet::PyTorch=file:///%cd%\resnet18"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `resnet` with the model.

#### Step 4: Inference

To query the model using the prediction API, open another session and run the following command:

##### Linux/macOS

```
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST "http://127.0.0.1:8080/predictions/resnet" -T "kitten.jpg"
```

This should return the following result:

```json
[
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.5183261632919312
  },
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.1956063210964203
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.1955675184726715
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.03224767744541168
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.02553771249949932
  }
]
```

## Binary Mode

This section walks through how to serve model in binary mode with DJL Serving.

Binary mode doesn't support pre-processing and post-processing. DJLServing only accept Tensor (NDList/npy/npz) as input and output.

### Provide Model File

For Binary Mode, you just need to place the model file in a folder.

DJL Serving supports model artifacts for the following engines:

- MXNet
- PyTorch (torchscript only)
- TensorFlow
- ONNX

### Packaging

To package model artifacts in a .zip:

```
cd /path/to/model
zip model.zip *
```

To package model artifacts in a .tar.gz:

```
cd /path/to/model
tar -czvf model.tar.gz *
```

### Serving Example

Let's run an example where we load a model in binary mode and run inference using the REST API.

#### Step 1: Download Model File

In this example, we will use a PyTorch resnet18 model.

```
mkdir resnet18 && cd resnet18
curl https://resources.djl.ai/test-models/traced_resnet18.pt -o resnet18.pt
```

We can package model artifacts in .zip or .tar.gz. In this example, we package the model in a .zip file.

```
cd resnet18
zip resnet18.zip *
```

#### Step 2: Start Server

Next, start DJL Serving and load this model at startup.

#### Linux/macOS

```
djl-serving -m resnet::Python=file://$PWD/resnet18.zip
```

Or we can load directly from model directory:

```
djl-serving -m resnet::PyTorch=file://$PWD/resnet18
```

#### Windows

```
path-to-your\serving.bat -m "resnet::PyTorch=file:///%cd%\resnet18.zip"
```

Or we can load directly from model directory:

```
path-to-your\serving.bat -m "resnet::PyTorch=file:///%cd%\resnet18"
```

This will launch the DJL Serving Model Server, bind to port 8080, and create an endpoint named `resnet` with the model.

#### Step 3: Inference

DJLServing in binary mode currently accepting NDList/Numpy (.npz) encoded input data. The returned data is always falls into NDList encoding.

You can use DJL API to create `NDList` and serialize the `NDList` to bytes as the input.

##### Direct Inference

```
# download a sample ndlist encoded data
curl -O https://resources.djl.ai/benchmark/inputs/ones_1_3_224_224.ndlist
curl -X POST "http://127.0.0.1:8080/predictions/resnet" \
    -T "ones_1_3_224_224.ndlist" \
    -H "Content-type: tensor/ndlist" \
    -o "out.ndlist"
```

##### Python Client Inference

You can also define a Python client to run inference.

From [inference.py](https://github.com/deepjavalibrary/djl-demo/blob/master/djl-serving/binary-mode/inference.py), the following is a code snippet to illustrate how to run inference in Python.

```
data = np.zeros((1, 3, 224, 224), dtype=np.float32)
outfile = TemporaryFile()
np.savez(outfile, data)
_ = outfile.seek(0)

response = http.request('POST',
	'http://localhost:8080/predictions/resnet',
	headers={'Content-Type':'tensor/npz'},
	body=outfile.read())
```

Run the `inference.py` to see how it interacts with the server in python:

```
python inference.py
```

Users are required to build their own client to do encoding/decoding.
