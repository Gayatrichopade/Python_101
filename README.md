# Python_101 💫💫⭐
Statistics and Data visualization (Matplotlib and seaborn) using python implementation.

import random
list_1 = random.sample(range(1,10),5)

list_1

print(list(filter(lambda x:x%2==0,list_1)))


### Import Data

# %matplotlib.inline
import numpy as np
import pandas as pd

df = pd.read_csv('mushrooms.csv')

display(df.head(10))

degree_count=df['class'].value_counts()

degree_count.plot(kind='bar',)

### Mean and Median

incomes = np.random.normal(27000,15000,10000)

np.mean(incomes)

%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(incomes,50)
plt.show()

np.median(incomes)

incomes = np.append(incomes,[100000000])

np.median(incomes)

np.mean(incomes)

### Mode

ages = np.random.randint(18, high = 90, size = 500)
ages

from scipy import stats
stats.mode(ages)

incomes = np.random.normal(100,20,10000)

plt.hist(incomes,50)
plt.show()

display(np.mean(incomes))
display(np.median(incomes))
display(stats.mode(incomes))

incomes = np.append(incomes,[250000])

display(np.mean(incomes))
display(np.median(incomes))
display(stats.mode(incomes))

 STD DEV & VARIANCE 

incomes = np.random.normal(100,20,10000)

plt.hist(incomes,50)
plt.show()

incomes.std()

incomes.var()

## Examples of Data Distribution

### Uniform Distribution

import numpy as np
import matplotlib.pyplot as plt

values = np.random.uniform(-10.0,10.0,10000)
plt.hist(values,50)
plt.show()

### Normal / Gaussian 

from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.arange(-3,3,0.001)
plt.plot(x,norm.pdf(x))

Generate some random numbers with normal distribution. "mu" is desired mean, "sigma" is the standard deviation:

mu = 5.0
sigma = 2.0
values = np.random.normal(mu,sigma,10000)
plt.hist(values,50)
plt.show()

### Exponential PDF / "Power LAw"

from scipy.stats import expon
x = np.arange(0,10,0.001)
plt.plot(x,expon.pdf(x))

from scipy.stats import binom

n, p = 10,0.5
x = np.arange(0,10,0.001)
plt.plot(x,binom.pmf(x,n,p))

### Poisson Probability Mass Function

Example:- My website gets average 500 visits per day. What's the odds of getting 550?

from scipy.stats import poisson

mu = 500
x = np.arange(400,600,0.5)
plt.plot(x,poisson.pmf(x,mu))

## Percentile and Moments

### Percentile

vals = np.random.normal(0,0.5,10000)
plt.hist(vals,50)
plt.show()

np.mean(vals)

np.percentile(vals, 50)

np.percentile(vals, 25)

np.percentile(vals, 75)

np.percentile(vals, 20)

#### Percentile Activity

act_val = np.random.normal(0.50,1.0,100)

plt.hist(act_val,25)
plt.show()

np.mean(act_val)

np.percentile(act_val,25)

np.percentile(act_val,50)

np.percentile(act_val,75)

np.percentile(act_val,90)

### Moments

vals = np.random.normal(0,0.5,10000)

plt.hist(vals, 50)
plt.show()

The first moment is the mean; this data should average out to about 0:

np.mean(vals)

The second moment is variance

np.var(vals)

The third moments is skewness;

import scipy.stats as sp
sp.skew(vals)

The fourth moment is "kurtosis", which describes the shape of the tail. For normal distribution this is 0

sp.kurtosis(vals)

# MatplotLib Basics

#### Draw a line graph

x = np.arange(-3,3,0.001)
plt.plot(x,norm.pdf(x))
plt.show()

#### Multiple plots in one graph

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

#### Save it to file

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.savefig('Myplot.png',format='png')

#### Adjust the Axes

axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

#### Add a Grid

axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axes.grid() # Grid
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

#### Change Line types and colors

axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axes.grid()
plt.plot(x, norm.pdf(x),'b-') # b -> blue, - -> straight line
plt.plot(x, norm.pdf(x, 1.0, 0.5),'r-.') # r -> red, -. -> -. Pattern line
plt.show()

#### Labeling Axes and Adding a Legend

axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axes.grid()
plt.xlabel("Greebles") # Adding labels to x axis
plt.ylabel("Probability") # Adding labels to y axis
plt.plot(x, norm.pdf(x),'b-') # b -> blue, - -> straight line
plt.plot(x, norm.pdf(x, 1.0, 0.5),'r-.') # r -> red, -. -> -. Pattern line
plt.legend(['Straight','Pattern'],loc=4) # Adding Legend
plt.show()

#### XKCD style

plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30,10])

data = np.ones(100)
data[70:] -= np.arange(30)
plt.annotate('The day I realised I can do better',xy=(70,1), arrowprops = dict(arrowstyle="->"),xytext=(15,-10))
plt.plot(data)

plt.xlabel('Time')
plt.ylabel('My overall health')

#### Pie Chart

plt.rcdefaults() # removing XKCD mode

values = [40,55,4,12,14]
colors = ['r','g','b','c','m']
explode = [0,0,0.2,0,0]
labels = ['India', 'Unites states', 'China', 'Russia','Europe']

plt.pie(values, colors=colors, explode=explode, labels=labels)
plt.title("Student Locations")
plt.show()

#### Bar Chart

plt.bar(range(0,5), values, color=colors)
plt.show()

#### Scatter Plot

from numpy.random import randn

x = randn(500)
y = randn(500)
plt.scatter(x,y)
plt.show()

#### Histogram

%matplotlib inline
incomes = np.random.normal(12700,15000, 10000)
plt.hist(incomes, 50)
plt.show()

#### Box and Whisker

uniformSkewed = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100

data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
plt.boxplot(data)
plt.show()

#### Activity 

Try creating a scatter plot representing random data on age vs time spent watching TV. Label the axes

x = randn(650)
y = randn(650)

plt.xlabel("Age")
plt.ylabel("Time spent on watching TV")
plt.scatter(x, y)
plt.show()



# Seaborn

Seaborn is a visualization library that sits top of matplotlib, making it nicer to look at and adding some extra capailities too.

%matplotlib inline 
import pandas as pd

df = pd.read_csv("http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv") #real dataset of 2019 model year vehicles.

gear_counts = df['# Gears'].value_counts()
gear_counts.plot(kind = 'bar')

import seaborn as sns
sns.set() #To change default settings of matplotlib

gear_counts.plot(kind = 'bar')

df

### Distplot

sns.distplot(df['CombMPG'])

### Pairplot

Let you visualize plots of every combination of various attributes together, so that you can find interesting patterns between features.

df2 = df[['Cylinders', 'CityMPG', 'HwyMPG', 'CombMPG']]
df2.head()

sns.pairplot(df2, hue="Cylinders", height= 3)

### ScatterPlot

sns.scatterplot(x="Eng Displ", y= "CombMPG", data=df)

### JointPlot

sns.jointplot(x="Eng Displ", y= "CombMPG", data=df)

### LmPlot

Provides plot scatter with linear regression line on plot

sns.lmplot(x="Eng Displ", y= "CombMPG", data=df)

### BoxPlot

sns.set(rc={'figure.figsize':(15,5)})
ax= sns.boxplot(x='Mfr Name',y='CombMPG',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

### SwarmPlot

ax= sns.swarmplot(x='Mfr Name',y='CombMPG',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

### CountPlot

ax= sns.countplot(x='Mfr Name',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

### HeatMap

df2 = df.pivot_table(index='Cylinders', columns ='Eng Displ', values = 'CombMPG', aggfunc='mean')
sns.heatmap(df2)

### Activity

Explore the relationship between the number of gears a car has, and it's combined MPG rating. Visualize these two dimensions using a scatter plot, lmplot, swarmplot, jointplot and pairplot. 

What conclusions can you draw?

df2 = df[["# Gears",'CombMPG']]

sns.set(rc={'figure.figsize':(15,5)})
ax= sns.boxplot(x="# Gears",y='CombMPG',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax= sns.scatterplot(x="# Gears",y='CombMPG',data=df)


sns.set(rc={'figure.figsize':(15,5)})
ax= sns.lmplot(x="# Gears",y='CombMPG',data=df2)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

sns.set(rc={'figure.figsize':(15,5)})
ax= sns.pairplot(data=df2)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

sns.set(rc={'figure.figsize':(15,5)})
ax= sns.swarmplot(x="# Gears",y='CombMPG',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

sns.set(rc={'figure.figsize':(15,5)})
ax= sns.jointplot(x="# Gears",y='CombMPG',data=df2)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)



## Covariance and Correlations

### Covariance

def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]

def covariance(x,y):
    n = len(x)
    return np.dot(de_mean(x),de_mean(y)) / (n-1)

pagespeeds = np.random.normal(3.0,1.0, 1000)
purchaseamount = np.random.normal(50.0, 10.0, 1000)

sns.scatterplot(pagespeeds, purchaseamount)

covariance(pagespeeds, purchaseamount)

purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pagespeeds

sns.scatterplot(pagespeeds, purchaseAmount )

covariance(pagespeeds, purchaseAmount )

### Correlation

def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y) / stddevx / stddevy

correlation(pagespeeds, purchaseAmount)

Numpy can do all this for you with numpy.corrcoeff. It returns a matrix of the correlation coeffiecient between every combination of the passed array in.

np.corrcoef(pagespeeds, purchaseAmount)

np.cov(pagespeeds, purchaseAmount)



# Conditional Probability Activity and Excercise

Below is some code to create some fake data on how much stuff people purchase given their age range.

It generates 100,000 random "People" and randomly assigns them as being in their 20's, 30's, 40's, 50's, 60's or 70's.

It then assigns a lower probability for young people to buy stuff.

In the end, we have two python dictionaries:

"Totals" contains the total number of people in each age group. "Purchases" contains the total number of things purchased by people in each age group. The grand total of purchases is in totalPurchases, and we know the total number of people is 10,000.

from numpy import random

random.seed(0) #A random seed is used to ensure that results are reproducible. The random number generator needs a number to start with (a seed value), to be able to generate a random number. 

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = float(ageDecade) / 100.0
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1

totals

purchases

totalPurchases

Let's play with conditional probability.

First let's compute P(E|F), wehere E is "Purchase" and F is "Your in you 30's". The probability of someone in their 30's buying something is just the percentage of how many 30-year-old's bought something:

PEF = float(purchases[30]) / float(totals[30])
print("P(purchase | 30's):" + str(PEF))

P(F) is just the probability of being 30 in this dataset:

PF = float(totals[30]) / 100000.0
print("P(30's): " + str(PF))

And P(E) is the overall probability of buying something, regardless of your age:

PE = float(totalPurchases) / 100000.0
print("P(Purchase): " + str(PE))

If E and F were independent, then we would expect P(E|F) to be about the same as P(E). But they're not; P(E) is 0.45, and P(E|F) is 0.3. So, that tells us that E and F are dependent (which we know they are in this example.)

P(E,F) is different from p(E|F). P(E|F) would be the probability of both being in your 30's and buying something, out of the total population - not just the population of people in their 30's.

print("P(30's, purchase)" + str(float(purchases[30])/ 100000.0))

Let's compute the product of P(E) and P(F), P(E)(F):

print("P(30's)P(Purchase) " + str(PE * PF))

print((purchases[30] / 100000.0) / PF)

### Exercise

Modify the code such that the purchase probability does not vary with age, making E and F actually independent. 

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = 0.4
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1

Now, we will compute P(E|F) for some age group. Let's pick 30's:

PEF = float(purchases[30]) / float(totals[30])
print("P(purchase | 30's): " + str(PEF))

Now we will compute P(E)

PE = float(totalPurchases) / 100000.0
print("P(Purchases): " + str(PE))


# Baye's Theorem
         P(A).P(B/A)
P(A|B)=  ____________
            P(B)

