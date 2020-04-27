# Insincere-questions-detection(2019/5/21)
I use CNN as a try to classify Quora sincere and insincere questions. The convolutional neural network structure is based on Y. Kim. Convolutional neural networks for sentence classifi-cation. I make some changes so that the network performs better. The word embedding is Glove. 

The ipynb file is run over the platform Colab. You need to create a new fold 'project' and upload the ipynb, data and word embedding file to the fold.

The data is largely unbalanced. I use two ways to treat it. One is oversampling by repeating the data of smaller number. The other one is using weighted loss function. All the results are shown in the notebook.

The preprocessing part is contributed by my partner Gus Yang.

Data: https://www.kaggle.com/c/quora-insincere-questions-classification/data
