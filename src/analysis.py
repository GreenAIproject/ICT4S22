import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from tabulate import tabulate

matplotlib.rcParams['font.family'] = 'serif'

def setup_plot_format():
    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 42

    matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
    matplotlib.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_results():
    results = []
    with open('results.csv', 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for reader_row in reader:
            results.append(reader_row)
        return results[0], results[1:]


def remove_outliers(results):
    groups = results.groupby('experiment_id')
    groups = [
        df[df['train_energy(J)'] != 0]
        for _, df in groups
    ]
    new_results = [
        df[(np.abs(stats.zscore(df['train_energy(J)'])) < 3)]
        for df in groups
    ]
    
    
    return new_results

def test_normality(groups):
    for sample in groups:
        print(stats.shapiro(sample["train_energy(J)"]))

def print_results(algorithms_dict):
    for classifier, data in algorithms_dict.items():
        print()
        print('==========', classifier, '==========')
        print(data)

def energy_precision(data, algo=None):
    if algo is None:
        algo = data.algorithm.unique()
    fig, ax = plt.subplots(2, len(algo), figsize=(36,12))
    color = ['mediumvioletred', 'skyblue','limegreen','orange','olive','red','gray']
    
    data_rq21 = data[data.RQ == 2.1]
    data_rq22 = data[data.RQ == 2.2]
    
    for i in range(len(algo)):
        ax[0,i].set_title(algo[i])    
        dsi = data_rq21[data_rq21['algorithm'] == algo[i]]
        
        # dsi[dsi.groupby('experiment_id').median()]
        result = pd.DataFrame()
        for experiment_id in dsi['experiment_id'].unique():
            group = dsi[dsi['experiment_id']== experiment_id]
            new_row = group[group['train_energy(J)'] == group['train_energy(J)'].quantile(0.5, interpolation='nearest')]
            result = result.append(new_row)
        dsi = result
        ax[0,i].scatter(y=dsi['f1'], x=dsi['no_datapoints'], s=100, color=color[i], alpha=0.8)
        z = np.polyfit(dsi['no_datapoints'], dsi['f1'], 1)
        p = np.poly1d(z)
        ax[0,i].plot(dsi['no_datapoints'], p(dsi['no_datapoints']), color=color[i], linewidth=3)
        ax[0,i].set_ylim(bottom=0, top=1)
        ax[0,i].set_xlim(left=0)
    ax[0,0].set_xlabel('Number of Datapoints')
    ax[0,0].set_ylabel('F1-score')
    
    for i in range(len(algo)):
        ax[1,i].set_title(algo[i])    
        dsi = data_rq22[data_rq22['algorithm'] == algo[i]]
        
        # dsi[dsi.groupby('experiment_id').median()]
        result = pd.DataFrame()
        for experiment_id in dsi['experiment_id'].unique():
            group = dsi[dsi['experiment_id']== experiment_id]
            new_row = group[group['train_energy(J)'] == group['train_energy(J)'].quantile(0.5, interpolation='nearest')]
            result = result.append(new_row)

        dsi = result
        
        ax[1,i].scatter(y=dsi['f1'], x=dsi['no_features'], s=100, color=color[i], alpha=0.8)
        z = np.polyfit(dsi['no_features'], dsi['f1'], 1)
        p = np.poly1d(z)
        ax[1,i].plot(dsi['no_features'], p(dsi['no_features']), color=color[i], linewidth=3)
        ax[1,i].set_ylim(bottom=0, top=1)
        ax[1,i].set_xlim(left=0)
    ax[1,0].set_xlabel('Number of Features')
    ax[1,0].set_ylabel('F1-score')
    
        
    plt.tight_layout()
    plt.savefig("rq3.pdf")

def small_multi(ds3_1, ds3_2, algo=None):
    # draw small multiples
    if algo is None:
        algo = ds3_1.algorithm.unique()
    fig, ax = plt.subplots(2, len(algo), figsize=(36,12))
    color = ['mediumvioletred', 'skyblue','limegreen','orange','olive','red','gray']


#     # draw no_datapoints and train_energy(J)
    for i in range(len(algo)):
        dsi = ds3_1[ds3_1['algorithm'] == algo[i]]
    
        dsi = dsi.groupby('no_datapoints').median()
        ax[0,i].scatter(x=dsi.index, y=dsi['train_energy(J)'], s=100, color=color[i], alpha=0.8)
        ax[0,i].set_title(algo[i])    
        z = np.polyfit(dsi.index, dsi['train_energy(J)'], 1)
        p = np.poly1d(z)
        ax[0,i].plot(dsi.index, p(dsi.index), color=color[i], linewidth=3)
        # ax[0,i].spines['right'].set_visible(False) # remove the top and right border lines
        # ax[0,i].spines['top'].set_visible(False)
        ax[0,i].set_ylim(bottom=0)
    ax[0,0].set_xlabel('Number of Datapoints')
    ax[0,0].set_ylabel('Energy Consumption (Joules)')

    # draw no_features and train_energy(J)
    for i in range(len(algo)):
        dsi = ds3_2[ds3_2['algorithm'] == algo[i]]
        dsi = dsi.groupby('no_features').median()
        ax[1,i].scatter(x=dsi.index, y=dsi['train_energy(J)'], s=100, color=color[i], alpha=0.8)
        z = np.polyfit(dsi.index, dsi['train_energy(J)'], 1)
        p = np.poly1d(z)
        ax[1,i].plot(dsi.index, p(dsi.index), color=color[i], linewidth=3)
        # ax[1,i].spines['right'].set_visible(False) # remove the top and right border lines
        # ax[1,i].spines['top'].set_visible(False)
        ax[1,i].set_ylim(bottom=0)
    ax[1,0].set_xlabel('Number of Features')
    ax[1,0].set_ylabel('Energy Consumption (Joules)')

    plt.tight_layout()
    plt.savefig("rq2.pdf")


def spearman_table_rq2(ds3_1, ds3_2, algo=None):
    # draw small multiples
    if algo is None:
        algo = ds3_1.algorithm.unique()
#     # draw no_datapoints and train_energy(J)
    
    rows = [['Algorithm', 'Column', 'Spearman', 'p-value']]
    for i in range(len(algo)):
        dsi = ds3_1[ds3_1['algorithm'] == algo[i]]

        print(ds3_1.head())
    
        correlation, pvalue = stats.spearmanr(dsi['f1'], dsi['no_datapoints'])
        row = [algo[i], 'no_datapoints', correlation, pvalue]    
        print(row)
        rows.append(row)

    for i in range(len(algo)):
        dsi = ds3_2[ds3_2['algorithm'] == algo[i]]
        correlation, pvalue = stats.spearmanr(dsi['f1'], dsi['no_features'])
        row = [algo[i], 'no_features', correlation, pvalue]    
        print(row)
        rows.append(row)
    print(tabulate(rows, headers='firstrow'))
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))
    
def rq1(data, algo):
    color = ['mediumvioletred', 'skyblue','limegreen','orange','olive','red','gray']
    max_no_datapoints = data['no_datapoints'].max()
    max_no_features = data['no_features'].max()
    data = data[data['no_datapoints'] == max_no_datapoints]
    data = data[data['no_features'] == max_no_features]
    
    result = pd.DataFrame()
    for algorithm in algo:
        group = data[data['algorithm']== algorithm]
        new_row = group[group['train_energy(J)'] == group['train_energy(J)'].quantile(0.5, interpolation='nearest')]
        result = result.append(new_row)
    
    result['train_energy(J)']= result['train_energy(J)'].apply(lambda x: round(x, 2))

    matplotlib.rcParams.update({'font.size': 11})

    ax = result.plot(kind='bar', x='algorithm', y='train_energy(J)', legend=False, color=color)

    for p in ax.patches:
        ax.annotate("{0:.2f}".format(p.get_height()), xy=(p.get_x()+0.08 , p.get_height()+0.01  ), ha='left') 
    

    ax = plt.gca()
    ax.set_ylabel('Energy Consumption (Joules)')
    ax.set_xlabel('Algorithm')
    ax.spines['right'].set_visible(False) # remove the top and right border lines
    ax.spines['top'].set_visible(False)
    
    classifiers_dict = {
        'Decision Tree': 'Decision\nTree',
        'Random Forest': 'Random\nForest',
        'Bagging Classifier': 'Bagging\nClassifier'
    }
    locs, labels = plt.xticks()
    plt.xticks(locs,
               [classifiers_dict.get(name.get_text(), name) for name in labels],
               rotation='horizontal', ha='center')
    # plt.xticks(rotation=45, ha='right')
    
    # plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig("rq1.pdf")

def main():
    # header, results = load_results()

    # this loads classifier names statically. If it changes in energy_experiment.py, change it here too!
    # (or note to self, save default params in a JSON and load it here - James)
    classifiers = [
        'SVM',
        'Decision Tree',
        # 'Naive Bayes',
        'KNN',
        'Random Forest',
        'AdaBoost',
        'Bagging Classifier',
    ]

    data = pd.read_csv('results.csv')
    
    
    #groups = remove_outliers(data)
    # test_normality(groups)
    
    ds1 = data[['algorithm','experiment_id','RQ','iteration','no_datapoints','no_features','datatype','f1']]
    ds1_1 = ds1[ds1.RQ == 2.1]
    ds1_2 = ds1[ds1.RQ == 2.2]
    ds3_1 = ds1_1[['algorithm','no_datapoints','f1']]
    ds3_2 = ds1_2[['algorithm','no_features','f1']]
    
    rq1(data, algo=classifiers)
    
    setup_plot_format()
    small_multi(ds3_1, ds3_2, algo=classifiers)
    energy_precision(data, algo=classifiers)
    spearman_table_rq2(ds3_1, ds3_2, algo=classifiers)
    
    
    
    #####
    ax = sns.violinplot(x="experiment_id", y="f1", hue="algorithm",
                        data=data[(data['algorithm'] == "Bagging Classifier")], palette="muted")
    #plt.show()
    plt.savefig('vioplot.png')
    # print(data.T)
    algorithm_data = {}
    for classifier in classifiers:
        algorithm_data[classifier] = data[(data['algorithm'] == classifier)]

    #print_results(algorithm_data)


if __name__ == '__main__':
    main()
