# Sample - Text Generation

Text generation can be thought of as a prediction problem.
Given N tokens, what is the next token.
As text is a categorical variable, this in turn can be thought of as a classification problem.

The goal of this repository is to demonstrate simple techniques in both R and Python for performing text generation using deep learning. 

## Scripts

The scripts below are available for both R and Python.
The names of the files will be identical except for the extension, either `.r` or `.py`.
The code is broken up into different files to help highlight the language differences while maintaining the structural elements of the process.

1. `~/code/preprocess_corpus.py` loops through the corpus and vectorizes it.
   Most DNN need to operate over vectorized values (i.e. integers).
   This applies to both word embedings and one-hot encodings.
   **NOTE**: There are built in TensorFlow functions that can perform this task.
   However, we elected to spell out the process because this is a sample.
2. `~/code/create_model.py` creates the model and saves it to disk.
   This allows us to decouple the creation, training, and prediction aspects.
   In this sample we have chosen to create a simple process: a single `LSTM` followed by a `Dense`.
   In a _real_ setting the network would have more layers.
   Unfortunately, hyper-tuning a multi-layer network is as much _art_ as science.
   To keep this example on-point, the simplest version was chosen.
* `~/code/load_corpus.{r|py}` supports loading a multi-document corpus.
  Please note: these functions are identical in internet, but the low-level tokenization is _slightly_ different.
  I will work out the delta when I get more time, but the difference is minor enough to still be fit for purpose.
* `~/code/utils.{r|py}` contains helper functions around generating the samples, batching, and one-hot encoding.
* `~/code/generate_text.{r|py}` defines the whole process for the example.
  Load data, then creating the model, then training the model, and finally producing the sample text.

## Environment

Tensor Flow is super fickle when it comes to setup.
The code used in this paper uses the v1 guidance found [here](https://github.com/MindMimicLabs/getting-started/blob/master/setup-your-environment.md)
