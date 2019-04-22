import csv
import numpy
import pandas as pd
#from scipy.stats.stats import pearsonr

def moyenne(x):
    m = 0
    n = 0

    for i in range(0, len(x)):
        if (x[i] != None):
            m += x[i]
            n += 1
    m = m/n
    return m

def cov(x, y):
    moyenneX = moyenne(x)
    moyenneY = moyenne(y)
    c = 0
    n = 0

    for i in range(0, len(x)):
        if (x[i] != None and y[i] != None):
            c += (x[i] - moyenneX) * (y[i] - moyenneY)
            n += 1
    c = c/n
    return c

def sigma(x):
    s = 0
    n = 0
    moyenneX = moyenne(x)

    for i in range(0, len(x)):
        if (x[i] != None):
            s += (x[i] - moyenneX)**2
            n += 1
    s = (s/n)**(1/2)
    return s

def correlation(x, y):
    i = 0
    while (i < len(x)):
        if (x[i] != None or y[i] != None):
            x.pop(i)
            y.pop(i)
        i += 1

    c = cov(x, y)
    sigmaXY = sigma(x) * sigma(y)
    r = c/sigmaXY
    return r
        


fileName = "oasis_longitudinal.csv"
demented = "Demented"
nonDemented = "Nondemented"
converted = "Converted"
male = 'M'
female = 'F'
left = 'L'
right = 'R'
dataLen = 13
sameLen = True
result = []
numericData = []

csvFile = open(fileName)
reader = csv.reader(csvFile)

for row in reader:
    result.append(row)

# Convertin data to numeric
for i in range(1, len(result)):
    row = []

    print(result[i])

    if (result[i][2] == demented):
        row.append(1)
    elif (result[i][2] == nonDemented):
        row.append(0)
    elif (result[i][2] == converted):
        row.append(2)

    for j in range(3, 5):
        if (result[i][j] != ''):
            row.append(float(result[i][j]))
        else:
            row.append(None)
    
    if (result[i][5] == male):
        row.append(1)
    elif (result[i][5] == female):
        row.append(0)
    
    if (result[i][6] == left):
        row.append(0)
    elif (result[i][6] == right):
        row.append(1)
    
    for j in range(7, len(result[i])):
        if (result[i][j] != ''):
            row.append(float(result[i][j]))
        else:
            row.append(None)
    
    numericData.append(row)
    if (len(row) != dataLen):
        sameLen = False

print('------------------NUMERIC DATA--------------------------------')
for i in range(0, len(numericData)):
    print(numericData[i])

print("sameLen = ", sameLen)

print("Correlation")
#print(numpy.corrcoef(numericData[0], numericData[1]))
#print(numpy.corrcoef(numericData[5], numericData[6], numericData[7], numericData[8]))
#a = (numpy.corrcoef([3, 2, 1, 2], [6, 4, 5, 1]))
var = {
    'a': numericData[0],
    'b': numericData[1],
    'c': numericData[3],
    'd': numericData[4],
    'e': numericData[5],
    'f': numericData[6],
    'g': numericData[7],
    'h': numericData[8],
    'i': numericData[9],
    'j': numericData[10],
    'k': numericData[11],
    'l': numericData[12],
}
df = pd.DataFrame(var)
print(df.corr())
print(df.corr(method='spearman'))
#print(a[0][1])
#print(pearsonr([3, 2, 1, 2], [6, 4, 5, 1]))


print("-----sd---------")
sd0 = 0
sd1 = 0
sd2 = 0
for i in range(0, len(numericData)):
    if (numericData[i][0] == 0):
        sd0 += 1
    elif (numericData[i][0] == 1):
        sd1 += 1
    elif (numericData[i][0] == 2):
        sd2 += 1
print("sd0: ", sd0)
print("sd1: ", sd1)
print("sd2: ", sd2)

            

#print(result)
#print(result[0])
#print(result[1])
    
csvFile.close()