#!/usr/bin/python

import random
import subprocess

svm_train = "/usr/local/bin/svm-train"
svm_predict = "/usr/local/bin/svm-predict"
problemfile = "problem.in"

svm_types = {"csvm": "0", "nusvm": "1"}
kernel_types = {"linear": "0", "poly": "1", "rbf": "2"}
probabilities = {"_prob": "1", "": "0"}

CLASSES = 8
SAMPLES_PER_CLASS = 4
ATTRIBUTES = 8

for svm_type in svm_types.keys():
    for kernel_type in kernel_types.keys():
        for probablity in probabilities.keys():
            s = svm_types[svm_type]
            t = kernel_types[kernel_type]
            b = probabilities[probablity]

            modelfile = f"m_{svm_type}_{kernel_type}{probablity}.libsvm"
            predictionfile = f"{modelfile}-predicted"

            subprocess.run([svm_train, "-s", s, "-t",
                            t, "-b", b, problemfile, modelfile])

            subprocess.run(
                [svm_predict, "-b", b, problemfile, modelfile, predictionfile])
