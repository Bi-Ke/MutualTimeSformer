## MutualTimeSformer

This is a short-term project, 'Mutual Action Recognition'. To well address the human-human interaction recognition problem, we propose a novel Mutual TimeSformer (MTimeSformer) to model the dynamic relationships between two people and to utilize the whole dynamic information from a video. 

![](imgs\Figure.png)

Specifically, in the multi-head attention module, we split q, k, and v into two groups q1, k1, v1 and q2, k2, v2 to represent each person. Across space, we only extract important inter-related dynamic information by computing q1, k2 and v2, without computing q1, k1 and v1, and vice versa. Across time, we compute q1, k1, v1 and q2, v2, k2 respectively, which is the same as the TimeSformer model.

We conducted experiments on the NTU-RGBD120 two-person mutual action videos. The dataset contains 25 thousand two-person mutual action videos that correspond to 26 different two-person interaction classes in a control environment. In order to locate each people and corresponding each person identity, we chose YOLOv5 object detection model to detect each person in a frame and then used DeepSort model to track each person and get individual identity. Then, we will apply the proposed MTimeSformer on two bounding boxes covering people.  



### Requirements

PyTorch >= 1.6



