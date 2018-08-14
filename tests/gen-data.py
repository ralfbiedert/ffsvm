#!/usr/bin/python

import random

CLASSES = 8
SAMPLES_PER_CLASS = 4

DENSE_ATTRIBUTES = 8

SPARSE_ATTRIBUTES = 128
SPARSE_ATTRIBUTE_PROBABILITY = 0.1

with open('data_dense/problem.csv', "w") as csv:
    with open('data_dense/problem.in', "w") as f:
        for c in range(CLASSES):

            # We'll write `SAMPLES_PER_CLASS * CLASSES` lines.
            for _ in range(SAMPLES_PER_CLASS):
                base = (c / CLASSES) + 0.0001

                f.write(str(c) + " ")

                nums = []

                for j in range(DENSE_ATTRIBUTES):
                    num = base + (base**3 * random.random())
                    rest = str(j) + ":" + str(num) + " "
                    nums.append(str(num))
                    f.write(rest)

                csv.write(",".join(nums) + "\n")
                f.write("\n")


with open('data_sparse/problem.in', "w") as f:
    for c in range(CLASSES):

        # We'll write `SAMPLES_PER_CLASS * CLASSES` lines.
        for _ in range(SAMPLES_PER_CLASS):
            base = (c / CLASSES) + 0.0001

            f.write(str(c) + " ")

            for j in range(SPARSE_ATTRIBUTES):
                if random.random() >= SPARSE_ATTRIBUTE_PROBABILITY:
                    continue
                num = base + (base**3 * random.random())
                rest = str(j) + ":" + str(num) + " "
                f.write(rest)

            f.write("\n")
