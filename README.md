# Sample - Text Generation

Text generation can be thought of as a prediction problem.
Given N tokens, what is the next token.
As text is a categorical variable, this in turn can be thought of as a classification problem.

The goal of this repository is to demonstrate simple techniques in both [R](https://www.r-project.org/) and [Python](https://www.python.org/) for performing text generation using deep learning. 

## Scripts

The scripts below are available for both R and Python.
The names of the files will be identical except for the extension, either `.r` or `.py`.
The code is broken up into different files to help highlight the language differences while maintaining the structural elements of the process.

1. `~/code/preprocess_corpus.py` loops through the corpus and vectorizes it.
   Most DNNs need to operate over vectorized values (i.e. integers).
   This applies to both word embedings and one-hot encodings.
   **NOTE**: There are built in TensorFlow functions that can perform this task.
   However, we elected to spell out the process because this is an example.
2. `~/code/create_model.py` creates the model and saves it to disk.
   This allows us to decouple the creation, training, and prediction aspects.
   In this example we have chosen to create a simple process: a single `LSTM` followed by a `Dense`.
   In a _real_ setting the network would have more layers.
   Unfortunately, hyper-tuning a multi-layer network is as much _art_ as science.
   To keep this example on-point, the simplest version was chosen.
3. `~/code/train_model.py` uses the vectorized corpus to train the model.
   It uses small batches to produce the training results.
   In this example we have chosen to _not_ randomize the batch order, electing to keep order the same both inside the document and accross epochs.
   In a _real_ setting this leads to training bias.
   To keep this example on-point, the simplest version was chosen.
4. `~/code/model_predict.py`
5. `~/code/utils.py` contains helper functions around generating the samples, batching, and one-hot encoding.

## Sugested Use

Copy/Paste the R or Python code into your repo and run it with your corpus.
_Always_ try a straight copy/paste first.
_After_ you have verified that the basic code works on your computer, _then_ modify the code to suite your process.

This work is intended to be educational.
If you use this code as the basis of your code in some academic publication, only a hyperlinked footnote to this repository is requested.

## Environment

Tensor Flow is super fickle when it comes to setup.
The code used in this paper uses the v1 guidance found [here](https://github.com/MindMimicLabs/getting-started/blob/master/setup-your-environment.md)
