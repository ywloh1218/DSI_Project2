### Yen Wei LOH

import numpy as np
import scipy.stats as stats
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############ PART 1 ##########
#### 1.1: csv to dictionary, header as keys.
reader = csv.DictReader(open('sat_scores.csv'))

sat_dict = {}
for row in reader:
    for column, value in row.items():
        sat_dict.setdefault(column,[]).append(value)
#print(sat_dict)

### 1.2: Making a pd DataFrame using the dictionary above, vs directly from the pd.read_csv function
dict_DF = pd.DataFrame(sat_dict)
sat_scores = pd.read_csv('/Users/lohyenwei/Desktop/data_science/Project2/Project2/sat_scores.csv')

#print(dict_DF)
#print(sat_scores)
type_dict = dict_DF.dtypes
type_df = sat_scores.dtypes
#print(type_dict)
print(type_df)

### making a DataFrame from the dictionary returns a dataframe with all strings, while using the pd.read_csv function returns strings for the first column (State), and integers for the other three.

### 1.3: printing first ten rows of sat_scores
print(sat_scores.head(10))
#print(sat_scores.describe())
#print(sat_scores[sat_scores['Rate']==4])

### Data describes SAT scores across America, probably diplaying average state scores for Verbal and Math. Rate is likely participation percentage.

#print(sat_scores.shape)

################# PART 2 ##################
## Creating a data dictionary that describes my data.
data_dict = {'State':['object','State'], 'Rate':['int64','Percentage of State Participation'],'Verbal':['int64','State average score for Verbal'],'Math':['int64','State average score for math'], sat_scores.shape: 'rows and columns of dataset'}
print(data_dict)

################ PART 3 #################
## Plotting distribution of Rate, Math, Verbal
#sns.distplot(sat_scores.Rate)
#sns.distplot(sat_scores.Verbal)
#sns.distplot(sat_scores.Math)
#plt.show()


## pairplot of data
#sns.pairplot(sat_scores)
#plt.show()
### from the Rate vs Verbal and Rate vs Math plots, it can be seen that there is a lower rate occurence for high Math and Verbal scores. Highest rates occur for Math and Verbal scores roughly around 500. Higher Math scores correlate with high Verbal scores, perhaps because a student who studies hard or is naturally gifted with one subject is also likely to study hard or be gifted for the other.

###################### PART 4 ####################
## Plotting a stacked histogram with pandas

# initialising new DataFrame
VerbMath = pd.read_csv('/Users/lohyenwei/Desktop/data_science/Project2/Project2/sat_scores.csv')
del VerbMath['Rate']
# check to make sure Rate was dropped
print(VerbMath.head())

#plt.figure()
#VerbMath.plot.hist(stacked=True)
#plt.show()

### 4.2: plot Verbal and Math on same boxplot
#sat_scores.boxplot(['Verbal','Math'])
#plt.show()

### In this case, plotting a stacked histogram does not give clear information about the scores of Verbal and Math, while the boxplot of Verbal and Math clearly indicate the max, min and IQR of Verbal and Math

### Further, if I plot this
#sat_scores.boxplot('Verbal','Math')
#plt.show()

### This plots the individual state averages of Verbal and Math, showing an approximate linear correlation.


### I am still unsure what Rate is, but assuming it is the rate of participation, Rate shows a distribution between 0 and 100 (which makes sense as it is a percentage), but throws off the scale of the graph, making it difficult to read off values for Verbal and Math

### 4.3
#sat_scores.boxplot()
#plt.show()

### Perhaps standardizing the variables would help. The first thing to spring to mind is to convert Verbal and Math scores to percentages.


################ PART 5 ####################
### 5.1 list of States with Verbal scores above averages
#avg_verbal = sat_scores['Verbal'].mean()
#print(avg_verbal)

#above_avg_verbal = sat_scores[(sat_scores.Verbal > avg_verbal)]
#print(above_avg_verbal['State'].tolist())
#print(len(above_avg_verbal['State'].tolist()))

## The average is 532, and there are 24 states (almost half) scoring above the average. This suggests that Verbal scores are normally distributed

### 5.2 find list of states that have verbal scores greater than median
#med_verbal = sat_scores['Verbal'].median()
#print(med_verbal)

#above_med_verbal = sat_scores[(sat_scores.Verbal > med_verbal)]
#print(len(above_med_verbal['State'].tolist()))

### 26 states above median (vs 24 states above average) suggest that the median and mean are close together. Median is larger than mean, suggesting a slight negative skew

## 5.3: column of Verbal - Math
sat_scores['vmrem']=sat_scores['Verbal']-sat_scores['Math']
print(sat_scores['vmrem'].head())

## 5.4: Create two DataFrames showing states with greatest difference

## 5.4.1: DF1 = 10 States with greatest gap where Verbal is greater
sat_scores['vmrem']=sat_scores['Verbal']-sat_scores['Math']

DF1 = sat_scores.sort_values(['vmrem'],ascending=False)
DF1 = DF1[:10]
print(DF1.head(3))

## 5.4.2: DF2 = 10 states with greates gap where Math is greater
sat_scores['mvrem']=sat_scores['Math']-sat_scores['Verbal']

DF2 = sat_scores.sort_values(['mvrem'],ascending=False)
DF2 = DF2[:10]
print(DF2.head(3))

#################### PART 6 ##################
### 6.1: create correlation matrix of variables excluding State
del sat_scores['vmrem']
del sat_scores['mvrem']

print(sat_scores.corr())

### Rate is negatively correlated with Verbal and Math scores, Math scores are highly correlated with Verbal scores

### 6.2: Use .describe() on Data Frame
print(sat_scores.describe())

### .describe() returns the below summary stats
#            Rate      Verbal        Math
#count  52.000000   52.000000   52.000000
#mean   37.153846  532.019231  531.500000
#std    27.301788   33.236225   36.014975
#min     4.000000  482.000000  439.000000
#25%     9.000000  501.000000  504.000000
#50%    33.500000  526.500000  521.000000
#75%    63.500000  562.000000  555.750000
#max    82.000000  593.000000  603.000000

### 6.3: assign and print covariance matrix
cov_matrix = sat_scores.cov()
print(cov_matrix)

### 1. The covariance matrix is a more generalised form of the correlation matrix. Correlation matrix is essentially a scaled version of the covariance matrix
### 2. covariance = correlation*(std_x)*(std_y)
### 3. correlation matrix is preferred in this case as it presents the correlation value as an easily understood number between -1 and 1. The correlation matrix also standardizes variables, however this is unimportant in this case as Math and Verbal scores use the same scale.

################ PART 7 #################
### 7.1
drugs = pd.read_csv('/Users/lohyenwei/Desktop/data_science/Project2/Project2/drug-use-by-age.csv')

### Checking for empty cells
print(drugs.isnull().sum())
### Check types
print(drugs.head())
print(drugs.dtypes)
### Check shape
print(drugs.shape)

### All columns should be int or float, but a number of them (eg meth-frequency, oxycontin-frequency, inhalant-frequency, heroin-frequency) are objects. Except age, that's probably best left as an object.

print(drugs.age.head(17))
### let's make age a float
drugs.set_value([10],'age',23)
drugs.set_value([11],'age',25)
drugs.set_value([12],'age',29)
drugs.set_value([13],'age',34)
drugs.set_value([14],'age',49)
drugs.set_value([15],'age',64)
drugs.set_value([16],'age',100)

print(drugs.age.head(17))

print(drugs[['meth-frequency','oxycontin-frequency','inhalant-frequency','alcohol-frequency']].head(17))
#### There are cells with value '-'. replace these with nan.
drugs = drugs.replace('-',np.NaN)
print(drugs[['meth-frequency','oxycontin-frequency','inhalant-frequency','alcohol-frequency']].head(17))#

### change string columns to float, and using num_drugs to keep drugs 'safe'
num_drugs = drugs.astype(float)
print(drugs.dtypes)
print(num_drugs.dtypes)


#### Now, FINALLY, we have a dataframe with proper types, now we can play.#

#### 7.2 so let's see what's going on, with a correlation matrix
print(num_drugs.corr())

##### Okayyyy, that was WAYYYYY too many variables. Let's drop all the redundant pairs, and rank top correlations for funsies

### I'm not smart enough (yet) to do that on my own, luckily there are these two awesome functions on StackOverflow from arun (link: http://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas)

def redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i],cols[j]))
    return pairs_to_drop

def get_top_abs_corr(df):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:]

print('Top Absolute Correlations')
print(get_top_abs_corr(num_drugs))

### okay sure, that got me a lot of interesting stuff, but I want to see correlation of age with errthang else

age_drugs=num_drugs.drop(labels='age',axis=1).corrwith(num_drugs['age'])
print(age_drugs.sort_values(ascending=False))
### Interesting, it seems as a general rule, drug use decreases with age, but if an individual keeps using, their frequency increases. I'm going to subset data into use and frequency datasets.


use_drugs = num_drugs[[col for col in num_drugs.columns if 'use' in col]]
freq_drugs = num_drugs[[col for col in num_drugs.columns if 'frequency' in col]]
#print(use_drugs.head())
#print(freq_drugs.head())

age_use = use_drugs.corrwith(num_drugs['age'])
age_freq = freq_drugs.corrwith(num_drugs['age'])

print(age_use.sort_values(ascending=False))
print(age_freq.sort_values(ascending=False))
#sns.distplot(age_use)
#sns.distplot(age_freq)
#plt.show()
### Yes, the plot clearly shows negative correlation with drug use, but higher correlation for drug frequency

### Let's see what drugs have highest use
print(use_drugs.describe())
#sns.distplot(use_drugs)
#plt.show()

## Getting "ValueError: color kwarg must have one color per dataset". wtf is this? Anyway, highest use drugs are alcohol and marijuana (okay, saw that one coming). What about highest frequency?

print(freq_drugs.describe())

### highest frequency drugs are stimulants and heroin. Okay, I guess that makes sense, as these are highly addictive drugs. But what are stimulants? Meth, crack and cocaine can be considered stimulants, are they included?

### Agh, just getting more and more questions. I'm going to pick a hypothesis and focus on that.

### 7.3 So my hypothesis here is that as people get older, we get more responsible as a whole and drug use drops. (Even if the outliers/rebels who still take drugs end up taking more)

#let's compare the total of n for ages below 29 and above 29 (I'm assuming 29 years is the magic "I can adult now" age.)

#print(num_drugs.age)
# sum of n below age 29
#print('sum of n below 29: ', num_drugs[['n']].iloc[[1,2,3,4,5,6,7,8,9,10,11,12]].sum(axis=0))
#sum of n above 29
#print('sum of n above 29: ', num_drugs[['n']].iloc[[13,14,15,16]].sum(axis=0))

#print(num_drugs.iloc[[9,10]])

### Going to skip ahead to 8 and 9 here, but if i get time to come back, i'd like to look at ages for heavy (freq > 45) use of alcohol and marijuana vs ages for heavy use of heroin and stimulants

####################### Part 8 #####################

print(sat_scores['Rate'])
#sns.distplot(sat_scores['Rate'])
#plt.show()

### Okay, let's assume Rate is a normally distributed variable. Outliers are then values that lie 3 std away from the mean. Code found on stack from CT Zhu
print(sat_scores.Rate.describe())

## cool, now we know what std and mean are. Let's have a look at a swarmplot
#pd.DataFrame(sat_scores['Rate']).boxplot()
#plt.scatter(np.repeat(1,sat_scores.shape[0]),sat_scores.Rate.ravel(),marker='+',alpha=0.5)
#plt.show()

### okay, so assuming outliers are values that are outside the IQR, (or more than 3 std devs away from mean)
Rate_list = sat_scores['Rate'].tolist()

outliers = []
for r in Rate_list:
    if r > 63.5:
        outliers.append(r)
    elif r < 9.0:
        outliers.append(r)

### Printing outliers
print(outliers)

### Removing outliers from Rate_list
for r in Rate_list:
    if r in outliers:
        Rate_list.remove(r)
print(len(Rate_list))

### so we modified our original Rate_list. oops. Let's make one so we can compare cleaned to dirty

dirty_rate = sat_scores['Rate'].tolist()
## Now let's compare mean, median and std dev.
print('Dirty mean: ', np.mean(dirty_rate))
print('Clean mean: ', np.mean(Rate_list))
print('Dirty median: ', np.median(dirty_rate))
print('Clean median: ', np.median(Rate_list))
print('Dirty std: ', np.std(dirty_rate))
print('Clean std: ',np.std(Rate_list))

### Clean mean and median are similar, which is expected as all we've done is remove the outliers that were 'pulling' mean and median away from true values. Clean std < dirty std, which is again expected as we have removed the outliers that were 'pulling' the IQR apart.


####################### Part 9 ##############
### 9.1 Spearman vs Pearson
# calculate spearman correlation
#print('Spearman: ', stats.spearmanr(sat_scores['Verbal'],sat_scores['Math']))

#print('Pearson: ', stats.pearsonr(sat_scores['Verbal'],sat_scores['Math']))

### From https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php: Spearman is a non-parametric version of Pearson. Spearman determines the strength and direction of the monotonic relationship between two variables, while pearson determines strength and direction of the LINEAR relationship between two variables. (Monotonic here means positive or negative. does not have to be linear.) Essentially, Spearman is good for relationships that are not strictly linear.
