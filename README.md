# LSTM_seq2seq-for-water-quality
This is a Tensorflow implementation of LSTM-seq2seq model construction for paper "A Novel Operational Water Quality Mobile Prediction System with LSTM-Seq2Seq Model".<br/>
**utilies.py :** encapsulates the functions needed for model training and tuning.<br/>
**tune.py :**  model tuning code.<br/>
**train.py :** model training code.<br/>
**Jintan.csv :** data for testing which ncludes water temperature, pH, dissolved oxygen, turbidity, conductivity, permanganate value, ammonia nitrogen data from 8.31.2018 to 5.31.2021 for the Jintan station in the Tuojiang River Basin.<br/>
**environment.yml :** environment configuration required for model building.<br/>
**tmonitorsectionautopredict.zip :** Mini program back-end code packaging file.<br/>
**yc.zip :** Mini program front-end code packaging file.<br/>
# 1 Environments
The model worked on Python 3 with Tensorflow>=2.0. Within this repository we provide an environment files (environment.yml) that can be used with Anaconda to create an environment with all packages needed.<br/>
Simply run
```python
conda env create -f environment.yml
```
# 2 Usage
## 2.1 Tuning
This tuning code is mainly used to adjust the size of the different hidden layers. You can set your own input and output time steps and training data path to execute the tuning code.
## 2.2 Training
The training code trains the model with the optimal hidden layer size obtained from the tuning code, for different input and output lengths of various indicators. Similarly, you can set your own input and output time steps, and training data path to execute the training code.
