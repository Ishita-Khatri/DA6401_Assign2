Part A involves training a CNN model from scratch on the iNaturalist dataset.

It is implemented across two files: `Assign2.ipynb` and `Assign2A4.ipynb`. `Assign2.ipynb` contains the code for Questions 1, 2, and 3, while `Assign2A4.ipynb` contains the code for Question 4.

I first define a CNN model as a class, making it flexible to accept different numbers of filters, filter sizes, activation functions, and dense layers. A model summary using a dummy input is also provided to verify the architecture.

Next, I load the iNaturalist dataset from the provided zip file. I use `StratifiedShuffleSplit` from `sklearn` to split the training dataset into training and validation sets, ensuring each class is equally represented in the validation set. A custom `Counter` function is also used to count the number of samples in each class within the dataloaders to confirm the correctness of the split.

Using the CNN model and dataloaders, I train the model on the dataset. I then perform a brute-force hyperparameter sweep using a large search space with the Bayesian optimization method to find the best configuration. However, I observe that many underperforming configurations can be identified early based on their initial loss/accuracy trends. To reduce unnecessary computations, I adopt two strategies to find the best configuration more efficiently:

1. Search space reduction: Based on the analysis of Sweep Phase 1, I reduce the search space by enabling batch normalization, fixing the activation function to ReLU, and using the momentum optimizer.

2. Early termination: I implement early stopping for underperforming configurations by monitoring their validation accuracy trends in the early stages.

Finally, I obtain the best hyperparameter configuration, achieving a validation accuracy of 40% after 20 epochs. I retrain the model using this configuration, which gives a validation accuracy of 36.75% and a test accuracy of 36.40%.

A 10x3 grid of test images, along with their true and predicted labels, is also displayed.

The Weights & Biases (wandb) report provides a detailed summary of the hyperparameters used, the sweep ranges, the best configuration found, performance graphs, and insights from the sweep experiments.
