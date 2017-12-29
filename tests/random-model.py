#!/usr/bin/python

import random

SV_PER_CLASS = 64
ATTRIBUTES = 16

header = """svm_type c_svc
kernel_type rbf
gamma 1
nr_class 2
total_sv 1024
rho -2.90877
label 0 1
probA -1.55583
probB 0.0976659
nr_sv 64 64
SV
"""

def write_with_label(f, label):
    for i in range(SV_PER_CLASS):

        f.write(label + " ")

        for j in range(ATTRIBUTES):
            rest = str(j) + ":" + str(random.random()) + " "
            f.write(rest)

        f.write("\n")


with open('xxx.model', "w") as f:
    f.write(header)
    # 256 0:0.3093766 1:0 2:0 3:0 4:0 5:0.1764706 6:0 7:0 8:1 9:0.1137485
    # -256 0:0.3332312 1:0 2:0 3:0 4:0.09657142 5:1 6:0 7:0 8:1 9:0.09917226
    write_with_label(f, "256")
    write_with_label(f, "-256")
