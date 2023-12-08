# airbus-ship-detection

loss: 0.2032 - dice_coef: 0.3429 - binary_accuracy: 0.9916 - val_loss: 0.1628 - val_dice_coef: 0.4184 - val_binary_accuracy: 0.9935

private score: 0.75324
public score: 0.56209


One of the challenges of this competition is strong data unbalance. Even if only images with ships are considered, the ratio of mask pixels to the total number of pixels is ~1:1000. If images with no ships are included, this ratio goes to ~1:10000, which is quite tough to handle. Therefore, I drop all images without ships, that makes the training set more balanced and also reduces the time per each epoch almost by 4 times. In an independent run, when the dice of my model reached 0.895, I ran it on images without ships and identified ~3600 false positive predictions out ~70k images. The incorrectly predicted images were incorporated to the training set as negative examples, and training was continued. The problem of false positive predictions can be further mitigated by stacking U-net model with a classification model predicting if ships are present in a particular image (https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection - ~98% accuracy). I also noticed that in some kernels the dataset is tried to be balanced by keeping approximately the same number of images with 0, 1, 2, etc. ships. However, this strategy would be effective for such task as ship counting rather than training U-net or SSD. One possible way to balance the dataset is creative cropping the images that keeps approximately the same number of pixels corresponding to a ship or something else. However, I doubt that such approach will effective in this competition. Therefore, a special loss function must be used to mitigate the data unbalance

Loss function is one of the most crucial parts of the completion. Due to strong data unbalance, simple loss functions, such as Binary Cross-Entropy loss, do not really work. Soft dice loss can be helpful since it boosts prediction of correct masks, but it leads to unstable training. Winners of image segmentation challenges typically combine BCE loss with dice (http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/). Similar loss function is used in publicly available models in this completion. I would agree that this combined loss function works perfectly for Carvana completion, where the number of pixels in the mask is about half of the total number of pixels. However, 1:1000 pixel unbalance deteriorates training with BCE. If one tries to recall what is the loss function that should be used for strongly unbalanced data set, it is focal loss (https://arxiv.org/pdf/1708.02002.pdf), which revolutionized one stage object localization method in 2017. This loss function demonstrates amazing results on datasets with unbalance level 1:10-1000. In addition to focal loss, I include -log(soft dice loss). Log is important in the convex of the current competition since it boosts the loss for the cases when objects are not detected correctly and dice is close to zero. It allows to avoid false negative predictions or completely incorrect masks for images with one ship (the major part of the training set). Also, since the loss for such objects is very high, the model more effectively incorporates the knowledge about such objects and handles them even in images with multiple ships. To bring two losses to similar scale, focal loss is multiplied by 10. The implementation of focal loss is borrowed from https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c .

Notes:
Valuable thoughts and directions for improvement
https://www.kaggle.com/code/iafoss/unet34-dice-0-87
Decent baseline model and visuals
https://www.kaggle.com/code/hmendonca/u-net-model-with-submission/notebook
Some visualizations
https://www.kaggle.com/code/vladivashchuk/notebook6087f5277f/edit
Losses
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master
https://github.com/JunMa11/SegLossOdyssey