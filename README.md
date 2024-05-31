# DATA130051-PJ2
Midterm Assignment

Task 1:
Fine-tuning a Convolutional Neural Network Pre-trained on ImageNet for Bird Recognition

Basic Requirements:

Modify an existing CNN architecture (such as AlexNet, ResNet-18) for bird recognition by setting its output layer size to 200 to match the number of classes in the dataset, while using network parameters pre-trained on ImageNet for the remaining layers.
Train the new output layer from scratch on the CUB-200-2011 dataset, and fine-tune the remaining parameters using a smaller learning rate.
Observe the effects of different hyperparameters, such as the number of training steps and learning rates, as well as their various combinations, and strive to improve the model's performance as much as possible.
Compare the results obtained from fine-tuning the pre-trained model with those from training the network from scratch using randomly initialized parameters on the CUB-200-2011 dataset to observe the improvements brought by pre-training.
Submission Requirements:

Submit only a PDF format experiment report, which should include a basic introduction to the model, dataset, and experimental results, as well as training and validation loss curves and validation accuracy changes visualized using Tensorboard during training.
Submit the code to your own public GitHub repo. The repo's README should clearly indicate how to train and test the model. Upload the trained model weights to Baidu Cloud/Google Drive or other cloud storage. The experiment report should include the GitHub repo link where the experiment code is located and the download address for the model weights.
Task 2:
Train and Test Object Detection Models Faster R-CNN and YOLO V3 on the VOC Dataset

Basic Requirements:

Learn to use existing object detection frameworks—such as mmdetection or detectron2—to train and test the object detection models Faster R-CNN and YOLO V3 on the VOC dataset.
Select 4 images from the test set and visualize and compare the proposal boxes generated in the first stage of the trained Faster R-CNN and the final prediction results.
Collect three images not included in the VOC dataset that contain objects from VOC categories, visualize and compare the detection results of the two models trained on the VOC dataset on these three images (displaying bounding boxes, class labels, and scores).
Submission Requirements:

Submit only a PDF format experiment report, which should include an introduction to the models, dataset, and experimental results, as well as training and validation loss curves and validation mAP curves visualized using Tensorboard.
The report should provide detailed experimental settings, such as the division of training and test sets, network structure, batch size, learning rate, optimizer, iteration, epoch, loss function, evaluation metrics, etc.
Submit the code to your own public GitHub repo. The repo's README should clearly indicate how to train and test the models. Upload the trained model weights to Baidu Cloud/Google Drive or other cloud storage. The experiment report should include the GitHub repo link where the experiment code is located and the download address for the model weights.
Note: This assignment is a group assignment. The number of team members should be less than or equal to 2 people (same quality of work, 1 person completing independently will receive extra points). One member of the group should submit the experiment report, with the names and student numbers of all group members listed in the report.
