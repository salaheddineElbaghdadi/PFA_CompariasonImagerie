import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


file = 'input/oasis_longitudinal.csv'
outputfile = 'output/numericDataSet.csv'

#reading data from csv file
data = pd.read_csv(file);
#replacing non numerical data by numerical values
data.replace({'F':0, 'M':1, 'L':0, 'R':1, 'Nondemented':0, 'Demented':1, 'Converted':1}, inplace = True)
#droping unnecessary columns
data.drop(['Subject ID', 'MRI ID', 'Visit'], axis = 'columns', inplace = True)

cor_matrix = data.corr()
hm = sb.heatmap(cor_matrix, annot=True)
plt.show()

cols = ['Group','MR Delay', 'M/F', 'Hand', 'Age', 'EDUC','SES','MMSE','CDR', 'eTIV', 'nWBV', 'ASF']
#sns_plot = sb.pairplot(data[cols])

#droping columns after correlation
data.drop(['ASF'], axis = 'columns', inplace = True)
data.dropna(inplace = True)

print(data.head())
print(data.info())

data.to_csv(outputfile, quoting=csv.QUOTE_NONE)
