import math
import matplotlib.pyplot as plt
import numpy as np

from_policy = "from_policy.txt"
policy_file = "regular_policy.txt"
to_policy = "to_policy.txt"

def read_data_pairs(file):
    f = open(file, "r")
    strData = f.read()
    strData = strData[:len(strData) - 1]
    strData = strData.split("\n")
    data = [(int(s.split(' ')[0]), int(s.split(' ')[1])) for s in strData]
    return data

def compute_mean(file):
    data = read_data_pairs(file)
    sum = 0
    for p in data:
        sum += p[1]
    mean = sum / len(data)
    dev = 0
    for p in data:
        dev += (p[1] - mean)**2
    dev /= (len(data) - 1)
    return (mean, dev, math.sqrt(dev))

def build_plots(files):
    for f in files:
        data = read_data_pairs(f)
        x = []
        y = []
        for p in data:
            x.append(p[0])
            y.append(p[1])
        plt.hist(y)
    plt.show()

print(compute_mean(policy_file))
build_plots([policy_file])


print(compute_mean(to_policy))
build_plots([to_policy])


print(compute_mean(from_policy))
build_plots([from_policy])


