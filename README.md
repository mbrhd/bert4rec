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
Create an  environment
    - you can just run the makefile to generate the env

Add the conda env to ipykernel
```
conda activate bert4rec
conda install -c conda-forge nb_conda_kernels
conda install ipykernel
```

Use the notebook to do data preparation and training


## References

### Code for bert4rec
* [x] The paper in [paperswithcode](https://paperswithcode.com/paper/bert4rec-sequential-recommendation-with)
* [x] Youtube [video](https://www.youtube.com/watch?v=4pYHEzwTa78) explaining the paper
