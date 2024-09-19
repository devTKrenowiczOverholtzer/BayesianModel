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

# list out all the types of faults 
# k_scratch typo in dataset
# creating a list of faults 
fauts = ["Pastry","Z_Scratch","K_Scatch", "Stains","Dirtin", "Dirtiness", "Bumps", "Other_Faults"]
# Create new column : code each of these faults into its own integer in this column 
# going to have to create a different column that has an integer for each fault to put this into our passifier for sklearn
# data wrangling (many datasets)
df['fault'] = 0
# indexing our dataframe at this new column fault which we are creating and we are setting the whole column to 0 so every entry in the column will be equaled to zero 
# print columns 
print(df.columns)

# now have 35 columns 
# measurements from x min to sigmoidofareas 
# faults from pastry to other_faults 
# added column fault - print head see that this column should just have 0 there
print(df.head())

# dataset: sorted by fault type 
# default behavior of test train split function is to randomly grab a sample which is good 
# if we had just taken out the last 20% of the dataset it was sorted so the test set would have faults that werent seen in the training set which would be bad 

