Part B involves fine-tuning a pre-trained model on the iNaturalist dataset. I use **ResNet50** as the pre-trained model. The file `Assign2B.ipynb` contains the code to: Load the pre-trained model along with its weights, modify it to match the input and output size of the iNaturalist dataset, freeze certain layers to make training feasible on a large model, and finally, fine-tune it on the iNaturalist dataset.

First, I load the ResNet50 architecture and its pre-trained weights using `torchvision`.
Since the iNaturalist dataset has 10 classes, I replace the final fully connected layer with a new one that has output size 10, while keeping the input size the same as the original.

Next, I freeze all layers of the network by setting `param.requires_grad = False` for all model parameters. Then, I unfreeze the last `k` layers by enabling gradient tracking for them. I start with k = 1, i.e., freezing all layers except the last one, and run a sweep to find the best values for the optimizer, batch size, and learning rate. After that, I repeat the process for k = 2, k = 3, and k = 4, unfreezing more layers each time.

The Weights & Biases (wandb) report provides: The best hyperparameter values and validation accuracy trends across 20 epochs for each value of k (1, 2, 3, and 4).


