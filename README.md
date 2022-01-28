# Data-Centric Green AI: An Exploratory Empirical Study
This repository is a companion page for the following research, submitted for revision at the 8th International Conference on ICT for Sustainability (ICT4S):
> Authors Blinded for Review. 2022. Data-Centric Green AI: An Exploratory Empirical Study. Submitted for revision to the 8th International Conference on ICT for Sustainability (ICT4S).

## Quick Start

1) Clone repo

```
$ git clone git@github.com:GreenAIproject/ICT4S22.git
$ cd ICT4S22
```

2) Install dependencies

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ mkdir output
```

3) get and prepare dataset

```
$ python src/get_data.py
$ python src/text_preprocessing.py
```

4) Execute the experiment script

```
$ python src/energy_experiment.py
```

**Note:** this step takes a long time because it runs all the energy measurements. If you want to skip this step, you can copy our measurements:

```
cp data/experiment_data.csv results.csv
```

5) Execute the data analysis script

```
$ python src/analysis.py
```


Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    .
    ├── LICENSE                       
    ├── README.md
    ├── data
    │   └── experiment_data.csv          Final experimental results data
    ├── plots                            
    │   ├── qqplot-energy.pdf            QQ-plot of energy measurements distribution
    │   ├── qqplot-f1.pdf                QQ-plot of F1-scores distribution
    │   ├── rq1.pdf                      RQ1 plot
    │   ├── rq2.pdf                      RQ2 plot
    │   └── rq3.pdf                      RQ3 plot
    ├── requirements.txt       
    └── src
        ├── analysis.py                  Statistical analysis script
        ├── energy_experiment.py         Empirical experiment script
        ├── get_data.py                  Retrieve dataset from UCI
        ├── modify_dataset.py            Dataset preprocessing script
        ├── text_classification.py       AI classification script
        └── text_preprocessing.py        tf-idf dataset preprocessing
        
