import random

randn = [100, 200, 500, 1000, 2000, 5000, 10000]

def processA():
    if random.random() <= 0.85:
        return 1
    else:
        return 0

def processBC():
    if random.random() < 0.95 and random.random() < 0.90:
        return 1
    else:
        return 0

rel = []
for n in range (len(randn)):
    inn = randn[n]
    sum = 0.0
    for i in range (inn):
        if processA() or processBC():
            sum += 1
    rel.append(sum/inn)

print('reliability: ', rel)