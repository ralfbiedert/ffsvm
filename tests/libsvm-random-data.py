#!/usr/bin/python

import random

CLASSES = 8
SAMPLES_PER_CLASS = 4
ATTRIBUTES = 8


with open('problem.in', "w") as f:
    for c in range(CLASSES):

        # We'll write `SAMPLES_PER_CLASS * CLASSES` lines.
        for _ in range(SAMPLES_PER_CLASS):
            base = (c / CLASSES) + 0.0001

            f.write(str(c) + " ")

            for j in range(ATTRIBUTES):
                rest = str(j) + ":" + str(base +
                                          (base**3 * random.random())) + " "
                f.write(rest)

            f.write("\n")
