# How to use the repo
## Steps to get started:
1 - Create an  environment:
  * Option 1: you can just run the makefile to generate the env. The makefile supports both conda and virtualenv
  * Option 2: create your environment with your own python environment manager. 

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

```
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
```

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
