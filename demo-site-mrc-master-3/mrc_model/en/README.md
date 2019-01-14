# English MRC

This folder contains the code for the English version of the demo website for MRC.

## Train

The first step is to train a model on English SQuAD dataset.

In order to get good results, it's better to use BERT-Large. But to train efficiently such a big model, we need TPU.

Please refer to this [Colab notebook](https://colab.research.google.com/drive/1MEgLS4C6jP2eoVvNRnDDLP7hXg6Wa742) for the training phase on TPU. You can also use your own Cloud TPU, the code is the same.

## Evaluate

After training on TPU, we have a saved checkpoint for the trained model.
Now let's run it on our local GPU server, and evaluate how good it is. This can be done easily by using the script `test_tpu_model.sh`, in the `script` directory.

## API

Starting the Flask server running the MRC API is simple. First, go to the right folder : `demo-site-mrc/mrc_model/`. Then simply run the command :

`python -m en.mrc_api`

---

The Flask application will load the configuration file `flask_config.json` located in the `mrc_model/en` directory, and use the given configuration to run the model. 

_Note : if you didn't evaluate your trained model using the script `test_tpu_model.sh`, you need to manually download your trained model on the GPU server and specify its location in the configuration file._

### Configuration

| Item | Type | Example | Description |
|---|---|---|---|
| `bert_config_file` | `str` | `../script/bert_model/bert_config.json` | Location of the configuration file for the BERT model. |
| `bert_vocab_file` | `str` | `../script/bert_model/vocab.txt` | Location of the vocabulary file used by the BERT model. |
| `bert_checkpoint_file` | `str` | `../script/bert_model/model.ckpt-32579` | Location of the model checkpoint to use for inference. |
| `lower_case` | `bool` | `True` | Case mode used by the BERT model : <ul><li>`True` : Uncased</li><li>`False` : Cased</li></ul> |
| `tmp_dir` | `str` | `./tmp` | Location of the folder where to store temporary files. |
| `context_max_len` | `int` | `384` | Maximum length allowed for the passage to infer. |
| `doc_stride` | `int` | `128` | Document stride length to use by the model, when the passage is too big. |
| `question_max_len` | `int` | `64` | Maximum length allowed for the question to infer. |
| `answer_max_len` | `int` | `42` | Maximum length allowed for an inferred answer. |
| `gpu` | `str` | `1` | String describing which GPU to use for inference. Since there is 4 GPU, number outside of the range [0 ~ 3] are not valid. Let empty in order to use all GPU. Specify `-1` to infer on CPU. |
| `batch_size` | `int` | `8` | Batch size to use for inference (for inference batches). |
| `n_best` | `int` | `1` | Number of best predictions to keep at inference time. |
| `host` | `str` | `0.0.0.0` | IP to use to host the Flask application. Using `0.0.0.0` makes Flask listen on all public IP. |
| `port` | `int` | `4243` | Port on which Flask will listen for incoming requests. |

## Unit tests

The interface of the API is tested through unit tests. To run the tests, first go to the right folder : `demo-site-mrc/mrc_model/`. Then simply run the command :

`python -m en.unit_test -v`

_Note : It is normal if all tests are not passing. Indeed, some test expect the answer to be right, but the model used is not perfect and will likely do some mistake. The goal of these tests is to see how the system reacts to the question. A human should judge if the test is OK or not._

---

You can also run a specific test by specifying its name on the command line. For example, to execute only `test_response_format` :

`python -m en.unit_test TestInterface.test_response_format -v`

---

Tests are using the configuration file `unit_test_config.json` to load the right model. This configuration file is similar to `flask_config.json`, there is just no Flask-specific parameter. Please refer to previous section for more details on the configuration file.

## Files

* **run_squad_42maru_single.py** : File containing the Tensorflow model definition.
* **inference.py** : Modified version of `run_squad_42maru_single.py`, to infer an answer from a given passage and question. This file is actually not used (it was used as a model but that's all).
* **func2.py** : Helpful functions used in `run_squad_42maru_single.py`.
* **interface.py** : File containing the definition of the user-friendly wrapper around the Tensorflow model. It allows to load a model and predict an answer from a question and a passage with a simple method call.
* **mrc_api.py** : Define the server hosting the JSON API, using Flask.
* **unit_test.py** : File containing the unit tests.