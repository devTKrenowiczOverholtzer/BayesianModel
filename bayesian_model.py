import pandas as pd
import matplotlib.pyplot as plt 
# importing just the pyplot fo matplotlib
from sklearn.naive_bayes import GaussianNB
# guassian naive bayes classifier GaussianNB
from sklearn.model_selection import train_test_split
# test training split function
import seaborn as sns 
# seaborn to look at coorelation plot 

# dataset note: measurement data xmin xmax ymin ymax for each kind of steel defect (rows)
# dataset note: faults (scratch,stains,dirtiness,bumps,other)
# dataset note: column for each fault , 0 if the particular instance didnt display that fault and a 1 if it did 

#read in csv
df = pd.read_csv("faults.csv")
# print
print(df.head())

# Visualize correlation 
# print correlation corr()
##print(df.corr())

# Create heatmap of the correlation
##plot = sns.heatmap(df.corr())
# will cut off some of it becuase its not going to be enough 
##plt.show()
# clear
##plt.clf()

# see that there are a few features that are correlated with each other and the faults 
# want to be careful of eventually is these things which are really highly cross correlated because Bayesian asssumptions theres not a ton of cross correlation between your independent features or variables 
# create a classifer with all these features 
# extend model by using a couple different subsets of the features one of the ways I would suggest choosing those features is eliminating some of the ones that are cross correlated with others

 



# going to have to create a different column that has an integer for each fault to put this into our passifier for sklearn



