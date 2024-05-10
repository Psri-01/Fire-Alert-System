# Fire-Alert-System

Dataset used: https://drive.google.com/file/d/1ydVpGphAJzVPCkUTcJxJhsnp_baGrZa7/view

This project utilizes a diverse, publicly available dataset (Dataset) containing 2059 jpg images of which 300 real-world images of fire and smoke are chosen for robust model training. Data augmentation is performed for YOLOv5 by flipping and zooming the images to expand the dataset size and potentially enhance its performance and robustness. It has been manually annotated with the MakeSense platform, which allows efficient manual annotation, for YOLOv4-tiny and automatically annotated in the Roboflow platform for the YOLOv5 variants in two object detection formats: the YOLO darknet format and YOLOv5 pytorch format for training YOLOv4-tiny and YOLOv5 n,s,m variants respectively, by drawing a bounding box according to the respective class labels, fire and smoke. The annotations are stored as a .txt file corresponding to their respective images.

Following the data acquisition and augmentation steps, the annotated images are used to train models for comparative analysis. Training a YOLOv4-tiny model on a custom dataset is done as follows: The darknet repository is cloned onto the Google Colab virtual machine to obtain the model’s weights. The files needed for training are uploaded, including the classes, custom configuration file and process.py file which converts 90% of the image labels to training and testing labels (as a .txt file) respectively. The Makefile is changed to enable OpenCV and GPU, build darknet. The hyperparameters chosen for training are: IoU loss: CIoU (4); Batch size, i.e. the number of images processed at once is 32 (because of the way YOLO performs its convolutions); the number of maximum batches is 6000. Step sizes are 4800 and 5400. Number of filters in the convolutional layers above both yolo heads is (classes+5)*3 = 7*3=21. The detector is then trained for a few epochs, and its performance is evaluated using the mean average precision (mAP) metric. It is a popular metric used in evaluating object detection systems.

YOLOv5 aims for a sweet spot of accuracy and efficiency. Its lightweight variant, YOLOv5s, is perfect for resource-constrained devices like Raspberry Pi. For real-time applications, its precision and speed are perfectly balanced, like fire and smoke detection where speed is critical. YOLOv5s is trained on our custom data for tasks like distinguishing fire and smoke patterns. This process entails: clone the YOLOv5 ultralytics repository, setup GPU using torch. Then the dataset is accessed with Roboflow’s private user API key. The dataset information is stored into the data.yaml file. The model configuration is changed according to the YOLOv5 model version used. It is trained on the custom data for 100 epochs. The results are plotted as a .png after training is completed, and analysed. Using test photos, the trained model will draw conclusions. The weights are used for inference, and the inference source and model confidence are given (greater necessary confidence results in fewer predictions). The source could be an image directory, a single image, a video file, or the webcam port of a device.

The trained model that is compatible with Raspberry Pi 3b+, YOLOv5s in this case is deployed in it. A preliminary check is conducted with flame sensor and smoke gas (MQ-2) sensor to detect fire and smoke or flammable gases. If a fire or smoke is detected, three images are captured with the help of a Raspberry Pi camera v2 (8MP) and alerts are sent to the user via email. The frame rate is reduced to 10 fps to avoid computational overload in the resource-constrained edge device. Upon detection of fire or smoke through the sensors and successful validation with the captured image via the camera, email alerts are sent to the user to prevent false alarms, and for enabling timely responsive action. This energy efficient edge-based system is an efficient way to utilise the model's detection for providing remote alerts and maintaining a safe, controlled environment. Successful implementation paves the way for integrating the model into real-world fire detection systems for a more cost-efficient and convenient fire and smoke detection and alert system.

Integrating the YOLOv5 model into a laptop webcam for real-time fire and smoke detection involves configuring the webcam to capture video frames at a resolution it can process effectively. Here YOLOv5m is chosen as it has the highest accuracy. As the webcam feeds live video into the system, the YOLOv5m model continuously analyses each frame to detect and classify fire or smoke. This classification is performed with decent accuracy and confidence in identifying the correct categories. The results are then displayed directly on the live feed, providing immediate visual feedback. This setup allows for real-time analysis, which is crucial for fire detection and alarm systems. 

![image](https://github.com/Psri-01/Fire-Alert-System/assets/114862496/1a83161e-f998-4951-b8a8-7d84808fe59d)

The graphs from the YOLOv5m model output illustrate the training process across 100 epochs, focusing on three key loss metrics: box loss (localization loss), classification loss and obj loss (confidence loss). The box loss shows a significant reduction from an initial value of 0.1064 to 0.0437, indicating the model’s rapidly improving accuracy in bounding box predictions. The classification loss starts at 0.0284 but exhibits a sharp decline to 0.001869, suggesting the model’s increasing proficiency in correctly classifying objects within the bounding boxes. The obj loss starts at 0.03012 but presents an initial spike before descending from 0.03879 to 0.03117, which could reflect the model’s confidence in its predictions, specifically regarding the objects’ presence and the accuracy of the predicted bounding boxes. These trends are indicative of the model’s learning effectiveness and its potential for reliable object detection in various applications.

The ‘val/box_loss’ graph shows a steep decline in loss, showing a quick improvement in the model's accuracy in predicting bounding boxes. The ‘val/cls_loss’ graph demonstrates a dramatic decrease from a high initial value, reflecting the model’s enhanced capability in classifying objects within those boxes. The ‘val/obj_loss’ graph initially spikes, suggesting a potential issue that is quickly resolved as the loss decreases sharply, indicative of the model’s learning and adaptation in handling deformations or other complex features. 

Precision is the fundamental indicator of the model’s accuracy. It measures the proportion of true positives among all positive predictions, while recall quantifies the ability to detect all actual positives. The precision graph shows that the model maintains high precision, predominantly around 0.6, for a significant range of thresholds before experiencing a decline. This suggests that the model is capable of detecting nearly all actual positives. The recall graph indicates an initial rapid increase to high recall values, stabilizing near 0.5, which implies that the model is capable of detecting nearly all actual positives.

![image](https://github.com/Psri-01/Fire-Alert-System/assets/114862496/a9afc4fa-360b-4b5c-a8b4-fbe6e246b67f)

As soon as a fire or smoke is detected with the help of the Flame sensor and MQ-2 sensors respectively, the Raspberry Pi camera v2 captures three images and generates bounding boxes for the same. It has a resolution of 8 MP (mega pixels). Email alerts are obtained in Raspberry Pi, which helps in the user checking for a possibility of a false alarm. It helps the user validate the alert with the help of the captured image and taking the necessary actions and safety measures in order to prevent the fire from spreading. This setup not only allows for real-time analysis, but is crucial for fire detection systems that are portable and cost-effective. It displays the model’s capability to work on a complex detection such as that of fire and smoke and generalize well to it, thereby proving to be dynamic and highlighting its detection efficiency.

# Conclusion
This study has successfully demonstrated the potential of the YOLOv5m and YOLOv5s object detection models for identifying fire and smoke in various contexts. 300 images from an open-source dataset, evenly split into fire and smoke images, was labelled and annotated in the requisite format. The trained model that is most compatible with Raspberry Pi 3b+ is converted to a TensorFlow lite format (YOLOv5s) and deployed in it to detect fire and smoke or flammable gases. The system takes sensor outputs from Raspberry Pi and validates the same with the help of a Raspberry Pi camera by monitoring live feed. Upon detection, it sends an image capture of the fire/smoke via email, with the captured image, thereby preventing false alarms, and for enabling timely responsive action. This is an energy efficient low-cost edge-based alarm system, utilised for providing remote fire alerts and maintaining a safe environment.

Credits: https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-on-windows-5ec436100433
https://github.com/ultralytics/yolov5
