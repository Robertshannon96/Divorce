# Predicting Divorce With Machine Learning

# The Goal
Can marriage success be predicted based of a 54 questonaire? In this project the goal is to cater to both consumers and data scientists. Consumers in this case are new couples entering into the world of marriage who want to discover important questions that impact how successful a marriage in. In order to satisfy fellow data scientists, the other goal in this project to create the best possible model to predict marriage sucess based on previous data.

# Background

Divorce has been increasingly common in today's society. The Center for Disease Control, [CDC](https://www.cdc.gov/) conducts a [National survery](https://www.cdc.gov/nchs/data/dvs/national-marriage-divorce-rates-00-18.pdf) anually on marriage and divorce rate trends. The latest findings reported 2,132,853 marriages and 782,038 divorces for just the year 2018. 




# The Data
The data for this project was found [here](http://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set). Hosted by the UCI machine learning repository. A break down of 54 features and 170 different participants.

For a full 54 exhaustive list of features from the study click [here](http://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set) and scroll slightly down.

Within the scope of this research,
the divorce prediction was carried out by using the Divorce Predictors Scale
(DPS) developed by Yöntem and İlhan (2017, 2018) on the basis of Gottman
couples therapy (Gottman, 2014; Gottman and Gottman, 2012). The reason
for this is that Gottman couples therapy is a model that explains the causes
of divorce based on empirical research. 

Of the participants, 84 (49%) were divorced and 86 (51%) were
married couples. There were 84 males (49%) and 86 females (51%) in the
study group. The ages of the participants ranged from 20 to 63 (X̄= 36.04).
 Although the study was collected from seven different
regions of Turkey, the data can be applied to most demographics.

Important to note that of the participants, 74 (43.5%) were married for love, and 96 (56.5%) were married in an arranged marriage



# Intital EDA
First, taking a look at the raw data yields us:

![Raw_data]('imgs/raw_data.png)


To get a better understanding of my data I needed to first grasp how the questionaire was conducted. Participiants were provided a list of 54 statements amount marriage and asked to provide a 1-4 response of 4 = strongly agree, 3 = agree, 2 = disagree, and 1 = strongly disagree.

With this in mind I wanted to then look at a simple distribution of our output class, "Married or dviorced".

![simple_plot]('imgs/count_plot.png)
Luckily our data is evenely disributed so we won't need to worry about any class imablances here. 


# For New Couples

To address the new couples audience one of the most important things to look at from this project is based off this questionaire "What really are the most important questions for a successul mariage?". 

In order to answer this question a Principal Component Analysis was conducted on the existing data to determine which question from the survey will be most important on future surveys completed by new couples. Now, for the time being you can forget about the process behind how this was calculated as we will discuss that once we address the Data science audience. 

Based off my calculations from this survey I can tell you that the top 5 most important features in a marriage asked in this survey are the following:

## 1: I enjoy traveling with my significant other.
## 2: The time I spend with my significant other is special for us.
## 3: My significant other and I share the same views about being happy in life
## 4: I know my significant other very well.
## 5: My significant other and I have similiar ideas about roles in marriage

You can see a clear relationship from the top two both dealing with spending time with a significant other. And it turns out, experts agree. According to a recent article in the Journal of marriage and family, couples were twice as happy in their life when they spent more time together. Who would of thought!

# Principal Component Analysis
In the first look at my data, it appeared that the questions all had a high level of correlation to one another. To solve this the Principal Component Analysis method was applied.

**Steps of PCA**
1) Standardize columns
2) Create covariance (correlation if standardized) matrix
3) Find the eigenvectors and eigenvalues of the covariance/correlation matrix
4) The eigenvectors are the principal components

![Pca_plot]('data/pca_plot.png')


# Logistic Regression



# Single Decsion Tree



# Findings and conclusions



# Weaknesses of Study