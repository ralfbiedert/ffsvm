#!/usr/bin/python

import random

CLASSES = 8
SAMPLES_PER_CLASS = 4
ATTRIBUTES = 8

with open('data/problem.csv', "w") as csv:
    with open('data/problem.in', "w") as f:
        for c in range(CLASSES):

            # We'll write `SAMPLES_PER_CLASS * CLASSES` lines.
            for _ in range(SAMPLES_PER_CLASS):
                base = (c / CLASSES) + 0.0001

                f.write(str(c) + " ")

                nums = []

                for j in range(ATTRIBUTES):
                    num = base + (base**3 * random.random())
                    rest = str(j) + ":" + str(num) + " "
                    nums.append(str(num))
                    f.write(rest)

                csv.write(",".join(nums) + "\n")
                f.write("\n")
