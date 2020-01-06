# RingReduce
========================================================
Implementation of Ring AllReduce, introduced in Baidu's DeepSpeech
--------------------------------------------------------
This is a ground-up demo implementation of Ring Reduce using the PyTorch distributed package. Ring Reduce is extremely effective for efficient distributed deep learning. It works well for large-scale learning over multiple GPU clusters, where inter-GPU bandwidth is limited. 

_References_
[Andrew Gibiansky's blog post](http://andrew.gibiansky.com/)