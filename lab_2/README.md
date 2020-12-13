# Reproducing

Run `./repro_script.sh` inside the right conda environment
(in particular with `nltk` installed). This will download and train
everything and produce all the plots. To save the training logs,
containing among other things the final test accuracies and their
standard deviations, use `./repro_script.sh > training_log.txt`.

If the SST dataset and the word2vec embeddings are already downloaded,
the corresponding lines can be commented out.

Our trained models, training logs and other artefacts are also available
in the [GitHub repository](https://github.com/TomFrederik/nlp1_uva).

To train the models on the LISA cluster, the jobs in `./lisa_jobs/` can
be used, after downloading the dataset and pretrained models.
