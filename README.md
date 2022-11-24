# Problem

For this test we are asking you to develop your own recommendation model. More specifically we are expecting you to implement a transformer model that is able do deal with sequential recommendation problem where the model is trained to predict the next item interacted by the user according to its past activities.

For your implementation you can freely use tensorflow or pytorch however we are working mostly with the tensorflow ecosystem (that's why we'd appreciate to see your skill with this framework). Your implementation of the model must be done using the functional api of the framework you choose most as possible. We don't want to see the use of external libraries. Simple layers like Linear or Convolution layers can be picked up directly from the framework you use however we would like to see at least a clean implementation of layers like the one doing the scale dot product computation. The same is true for the preprocessing and the loss function.

To help you, you can find different books, web pages or papers on this topic. However, we advise you to follow this paper which describes well the problem we are trying to solve with this test.
[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer - arXiv](https://arxiv.org/pdf/1904.06690.pdf)

To train your model you can use data provided with the test. These data contain a tensorflow dataset of user session where a session can be viewed as a sequence of activities done sequentially by a user and during a predefined range of time. Each session are defined by the following fields:
-  userIndex: index of the user doing the session
-  movieIndices: index of movies liked by the user sorted using timestamps (we are recommending movies !)
-  timestamps: sorted date in seconds where the activity has been done

Movies are indexed from 0 to the total number of movies. Furthermore you could find a movie_title_by_index.json file containing a mapping movieIndex  -> movieTitle.

The dataset can be parsed using the following proto schema:

```
    schema = {
        "userIndex": tf.io.FixedLenFeature([], tf.int64),
        "movieIndices": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        "timestamps": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64)
    }
```

Don't hesitate to export your code within a google collab notebook to test your model implementation. The model is easily runnable on GPU machine.
However, we expect the output to be a git repository with all indication allowing to run and test the code.

Due to the limitation of time, we are not expecting a perfect result. We mostly want to see your coding skills and your ability to understand and master what you are doing.
Don't hesitate to ask to us if you have any questions or meet issues with data.

Enjoy your test!

# How to use the repo
## Steps to get started:
1 - Create an  environment:
    - Option 1: you can just run the makefile to generate the env. The makefile supports both conda and virtualenv
    - Option 2: create your environment with your own python environment manager. 

2 - Activate your python environment 
```
conda activate bert4rec
```
3 - Install requirements by running
```
pip install -r requirements.txt
```

## Use of the bert4rec library: 
The repo is structured as following:

├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├─  bert4rec           <- Source code for use in this project.
│   └─── __init__.py    <- Makes src a Python module
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


### Data
As mentionned above there are three data folders: raw, interim, processed.
* The raw folder contains the raw data.
* The interim folder contains the preprocessed data.
* The processed folder contains the training and test datasets.

The data is already processed no need to run the notebooks: `1.0 - Explore raw data and prepare it.ipynb` and `2.0 - prepare datas.ipynb`

### Notebooks
Testing and running the library is done using notebooks. There are three notebooks:
```
1.0 - Explore raw data and prepare it.ipynb
2.0 - prepare datast.ipynb
3.0 - Training.ipynb
```
By default we turn the project into a Python package bert4rec (see the setup.py file). You can import your code and use it in notebooks with a cell like the following:
```
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in bert4rec, it gets loaded
%autoreload 2

from bert4rec.bert import BertModel 
```


## References

### Code for bert4rec
* [x] The paper in [paperswithcode](https://paperswithcode.com/paper/bert4rec-sequential-recommendation-with)
* [x] Youtube [video](https://www.youtube.com/watch?v=4pYHEzwTa78) explaining the paper
