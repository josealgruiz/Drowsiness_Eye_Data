# Drowsiness_Eye_Data

The aim of this project was to develop an advanced computer vision model capable of detecting drowsiness in drivers using PERCLOS, blink percentage, and blink rate. To accomplish this, a comprehensive data collection process was conducted on the CARLA simulator and a custom map. Eye videos were recorded using the Dikabilis Glasses 3 eye tracker, capturing crucial eye movement information.

The collected eye videos were then subjected to a meticulous preprocessing stage, leveraging a custom-built code specifically designed to extract and determine important metrics such as PERCLOS, blink percentage, and blink rate. Each frame of the eye videos was meticulously labeled to accurately classify drowsiness levels.

Building upon this groundwork, a CRNN (Convolutional Recurrent Neural Network) model was constructed. This model combined the power of convolutional layers for feature extraction with recurrent layers for temporal analysis, allowing for the effective interpretation of the extracted PERCLOS, blink percentage, and blink rate features. The CRNN model demonstrated impressive capabilities in analyzing and classifying drowsiness levels in drivers, ultimately contributing to enhancing road safety.
