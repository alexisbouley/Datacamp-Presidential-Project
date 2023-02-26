# French presidential elections 2022

Authors : [Morvan Theo](https://github.com/Theo-Morvan), [Islamov Rustem](https://github.com/Rustem-Islamov), [Popov Nicolai](https://github.com/k0l1ka), [Wasik Thomas](https://github.com/WskThomas), [Bouley Alexis](https://github.com/alexisbouley)

This challenge was done as a project for the Master 2 Data Science (2022/2023), DATACAMP course

## Introduction

The goal of this project is to predict the results of 2022 French presidental elections. For this, we use different sources of data that are described later. The participants should predict the number of votes that were given for Emmanuel Macron, Marine Le Pen, and Jean-Luc Mélenchon in several cities of France.

### Data
The data used in the project is open, One can find below the links to download the data used in this project. <br>
However one should manually download these datasets. One shall launch the python file download_data which loads and properly processes the different datasets to match our problem. If the python script were to fail, download the following dataset in the folder data_source and then activate set download=False in download_data.py
- 2022 elections results : https://www.data.gouv.fr/fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/, download the txt file "resultats-par-niveau-burvot
- 2017 elections results: https://www.data.gouv.fr/fr/datasets/election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour-par-bureaux-de-vote/, same than for 2022, the file should be called "PR17_BVot_T1_FE.txt"
- 2019 employement statistics per town hall : https://www.insee.fr/fr/statistiques/6454652?sommaire=6454687#consulter
- 2019 average income per city : https://www.insee.fr/fr/statistiques/6036907, we use the FILO2019_DISP_COM.csv file


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.



### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
