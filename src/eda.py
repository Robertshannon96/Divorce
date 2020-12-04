import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
plt.style.use('ggplot')


def get_data():
    """
    Reads in data frame from data folder.

    Returns: dataFrame
    """
    df = pd.read_csv('../data/divorce_set.csv', sep=',')
    return df


def explore_data(df):
    """
    performs somes basic eda on the data to allow for better modeling later.
    Input: DataFrame
    Output: Characteristics of DataFrame
    """

    print(df.head())
    print("Shape of Data: ", df.shape)
    print(df.info())


def count_plot(df):
    """
    Distribution of each outcome class. Divorce or no divorce

    Input: DataFrame
    Output: Simple histogram display

    """
    ax, fig = plt.subplots(1, figsize=(10, 6))
    sns.countplot('Class', data=df)
    plt.xlabel('Yes or No', fontsize=18)
    plt.ylabel('Total', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Distribution of Divorce vs no divorce', fontsize=24)
    plt.savefig('count_plot.png')


def correlation_map(df):
    """
    This is not for presentation, rather for the programmer to better understand the correlaton between questions
    and final outcome.

    Input: DataFrame
    Output: Correlation heatmap for the sole purpose of the programmer.
    """
    plt.figure(figsize=(35, 35))
    sns.heatmap(df.corr(), annot=True, cmap="magma")
    plt.savefig('correlation_map.png')
    print(df.corr())


def standardize_dataframe(df):
    """
    In order to perform a principal component analysis on the data, you first need to standardize it. The goal here
    is to have column means at 0 and standard deviation at 1.

    Input: DataFrame
    Returns: Standardized DataFrame
    """

    features = []
    for i in range(1, 55):
        features.append(f"Atr{i}")

    x_ = df.loc[:, features].values
    y = df.loc[:, ['Class']].values
    x = StandardScaler().fit_transform(x_)

    # check that column means are 0, standard deviation of 1
    print(x.mean(axis=0))
    print(x.std(axis=0, ddof=1))

    return x


def pca(df):
    """
    Principal component analysis to better understand the correlation between the data columns.
    Used standardized Data to perform the analysis.

    Input: DataFrame
    Returns: PCA DataFrame and print the top 5 most important questions
    """
    x = standardize_dataframe(df)
    pca_ = PCA(n_components=2)
    principal_components = pca_.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, df[['Class']]], axis=1)

    features = []
    for i in range(1, 55):
        features.append(f"Atr{i}")
    top_5 = pca_.components_[0].argsort()[-5:]
    print('Top 5 most important questions: ', np.array(features)[top_5])

    return final_df


def plot_pca(df):
    """
    Plotting of the PCA analysis.
    """

    final_df = pca(df)
    no_divorce_df = final_df[final_df['Class'] == 0]
    divorce_df = final_df[final_df['Class'] == 1]
    plot1 = divorce_df[['principal component 1', 'principal component 2']]
    plot2 = no_divorce_df[['principal component 1', 'principal component 2']]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    targets = ['Divorce', 'No-Divorce']
    ax.scatter(plot1['principal component 1'], plot1['principal component 2'], color='r')
    ax.scatter(plot2['principal component 1'], plot2['principal component 2'], color='b')
    ax.legend(targets)
    ax.grid()
    plt.savefig('pca_plot.png')


def main(df):
    explore_data(df)
    count_plot(df)
    correlation_map(df)
    plot_pca(df)


if __name__ == '__main__':
    df = get_data()

    main(df)
