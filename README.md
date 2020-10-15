# DNN-Inference-Optimization



## Project Introduction

For the DNN model inference in the end-edge collaboration scenario, design the adaptive DNN model partition and collaborative inference scheme for chain and non-chain DNN model, which can effectively solve the problems of coarse granularity and simplification of existing schemes and obtain the approximate optimal strategy by using dynamic programming method. Moreover, design the computation latency prediction model of DNN model with consideration of device load and task characteristics, which adds the features related to device load changes such as idle CPU time and memory occupancy for the first time. And a variety of common regression models are evaluated, which can accurately predict the computation latency of the DNN model in the real scene of device load changes. In addition, an end-edge cooperative inference framework of DNN model is designed and implemented. The framework is used to verify the computation latency prediction, model partition and collaborative inference strategy.



## Environment configuration and description

This program consists of a server-side program and a client-side program. The development language and environment are different.

The server uses Python as the development language. It is recommended to use anaconda as the Python package management software. IDE recommends using PyCharm. The specific development environment configuration is as follows:

| **Environment**                     | **Description**                                           |
| ----------------------------------- | --------------------------------------------------------- |
| Develop operating  system           |                                                           |
| Integrated  Development Environment | PyCharm                                                   |
| Development  language               | Python3.7.4                                               |
| Dependent library                   | Keras,tensorflow,networksx,scilit-learn,    numpy, socket |

The client uses Android as the development language, and the IDE recommends using Android Studio. The specific development environment configuration is as follows:

| **Environment**                     | **Description**                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| Develop operating  system           |                                                              |
| Integrated  Development Environment | Android  Studio3.5.3                                         |
| Development  language               | Java1.8                                                      |
| Dependent library                   | Andriod  SDK8.0,gson2.6.2,  tensorflow-android1.5.0,openCVLibrary344 |



## Project code description

- Calculate the delay prediction model

  The DNN layer calculation delay prediction model code is located under the path ct_prediction_model.

  Data collection tool: Each type of DNN network layer needs to collect its own inference training data. The code for collecting data is: ct_prediction_model/xxx/RunxxxLatencyData.py. Generate xxx_train.csv file after execution.

- DNN model realization and export

  Model implementation: The 14 DNN model implementation codes used in the experiment are located in: xxxModel.py under the models/xxx/ folder. Execution method: python xxxModel.py.

  Model export: Model deployment is divided into two parts, one is the export of the model running on the server side, and the other is the export of the model on the Android phone. The implementation of the 14 DNN models and the model export code are all organized in the same way, and they are all TrainXxx.py under the models/xxx/ folder. Execution method: python TrainXxx.py. Among the model files generated after execution, the model.h5 file is a model running on the server side, which is read into executable inference with Python code; the xxx_model.pb file is a model file that can be run on an Android phone.

- End-edge node time calculation

  Edge node calculation time: The code is located in: xxxModel.py under the models/xxx/ folder. Execution method: python xxxTimeCount.py. Generate EdgeNodeComputeTime.txt file after execution.

  Mobile node calculation time: The code is located at: android/app/src/main/java/aflak/me/tensorflowlitexor/LayerComputeTimeActivity.java. Generate MobileNodeComputeTime.txt after execution.

  Upload transmission delay statistics for each layer of mobile devices: The code is located at: android/app/src/main/java/aflak/me/tensorflowlitexor/LayerUpTimeActivity.java. Generate MobileNodeUploadTime.txt after execution.

  Download transmission delay statistics for each layer of mobile devices: The code is located at: android/app/src/main/java/aflak/me/tensorflowlitexor/LayerDownTimeActivity.java. Generate MobileNodeDownloadTime.txt after execution.

- DNN partition algorithm

  The code locations of the DP-based non-chain DNN partitioning algorithm, DADS partitioning algorithm, and NeuroSurgeon partitioning algorithm are as follows:

  MyDNNPartition.py

  DADSPartition.py

  NeurosurgeonPartition.py



## Running and sample

- Running

  Socket communication between client and server.

  Client:

  The Android code is located in: android folder.

  The interface of the Android client is relatively simple, as shown below, important information is printed in the log, pay attention to the log. The corresponding relationship between each button in the interface and the Android code Activity file is as follows:

  ![img](README.assets/clip_image002.png)

  ​	The location of the corresponding Activity code: android/app/src/main/java/aflak/me/tensorflowlitexor/xxx.java.

  Server:

  Among the above functions on the Android side, [Statistics of upload transmission delay per layer of mobile devices], [Statistics of download transmission delay per layer of mobile devices] and [Cooperative reasoning execution], before running, the server program needs to be started in advance. The server program code location is as follows:

  TargetNetUpTime.py

  TargetNetDownTime.py

  server.py

- Sample

  The system has two inference modes: (1) Offine Mode, that is, when the mobile phone is disconnected from the Internet, it performs inference tasks independently, and "Time" is the inference time. (2) Online Mode, that is, the mobile phone and the server communicate through a network connection to coordinate inference tasks. "Local Time" is the time required for local inference tasks on the mobile phone, and "Server Time" is the data sent by the mobile phone. Time, server reasoning task time and mobile phone terminal receiving results. The sum of "Local Time" and "Server Time" is the time for collaborative reasoning between the mobile phone and the server.

  ![img](README.assets/clip_image002-1602646375471.png)

  ​	The following table shows the two modes of inference time and frame loss for different duration videos.

| Video duration | Offline   | Online        |           |      |
| -------------- | --------- | ------------- | --------- | ---- |
| InferenceTime  | DropFrame | InferenceTime | DropFrame |      |
| 5s             | 3033ms    | 36            | 3619ms    | 15   |
| 10s            | 9640ms    | 97            | 8147ms    | 23   |
| 15s            | 13505ms   | 149           | 10069ms   | 41   |
| 20s            | 17967ms   | 198           | 12658ms   | 59   |

 

## Team information

Name: AI-Maglev

Members:  Yuwei Wang, Sheng Sun, Kun zhao, Kang Li

Email: wangyuwei@ict.ac.cn