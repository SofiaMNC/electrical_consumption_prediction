# **Anticipating The Electrical Consumption of Buildings**
*Sofia Chevrolat (June 2020)*

> NB: This project is the first of a series comprising [the syllabus offered by OpenClassrooms in partnership with Centrale Supélec and sanctioned by the Data Scientist diploma - Master level](https://openclassrooms.com/fr/paths/164-data-scientist).
___
This study in 2 notebooks aims to estimate the CO2 emissions and the total energy consumption of the buildings of the city of Seattle by exploiting the declarative data indicated in their commercial exploitation permits (size and use of the buildings, recent works, construction dates...) released by the city for the years 2015 and 2016.
___

This study is divided into 2 notebooks: 
- A cleaning up, feature engineering and EDA notebook
- A modeling and prediction notebook
___
## Notebook 1 : Data Clean Up, Feature Engineering and EDA

This notebook is organised as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries and functions
- 0.2 Loading and describing the dataset
- 0.3 Assembling the data
    * 0.3.1 Deleting columns not present for both years
    * 0.3.2 Feature harmonization
    * 0.3.3 Fusing the data
    
**1. Data Targeting: Restricting To Buildings Not Used For Housing**

**2. Data Clean Up**
- 2.1 Deleting non exploitable features
- 2.2 Deleting duplicate features
- 2.3 Deleting outliers
- 2.4 Deleting the least complete features
- 2.5 Deleting completely empty observations
- 2.6 Deleting NaN values
- 2.7 Cleaning up string data
- 2.8 Correcting data types
- 2.9 Deleting absurd values
- 2.10 Re-indexing

**3. Feature Engineering**
- 3.1 Encoding and feature engineering
- 3.2 Creating new features

**4. Exploratory Data Analysis**
- 4.0 Splitting the dataset
- 4.1 Study of the repartition of the use of the different energy sources
    * 4.1.1 All buildings included
    * 4.1.2 By the buildings' main type of property
- 4.2 Statistical values
    * 4.2.1 Central tendency
        * 4.2.1.1 Qualitative features
        * 4.2.1.2 Quantitative features
    * 4.2.2 Feature distribution
        * 4.2.1.1 Qualitative features
        * 4.2.1.2 Quantitative features
- 4.3 Study of the first variable of interest: energy consumption (<i>SiteEnergyUse(kBtu)</i>)
    * 4.3.1 Study of the correlations between qualitative variables and total energy consumption
    * 4.3.2 Study of the correlations between quantitative variables and total energy consumption
- 4.4 Study of the second variable of interest: CO2 emissions (<i>TotalGHGEmissions</i>)
    * 4.4.1 Study of the correlations between qualitative variables and total CO2 emissions
    * 4.4.2 Study of the correlations between quantitative variables and total CO2 emissions

**5. Exporting the data**

**6. Conclusions**

___
## Notebook 2 : Data Analysis

This notebook is organized as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries
- 0.2 Loading the data set
- 0.3 Preparing the data set for feature selection
- 0.4 Choosing an evaluation criteria for the modeling

**1. Prediction energy consumption**
- 1.1 Feature selection
- 1.2 Splitting the dataset
- 1.3 Algorithm selection
    * 1.3.1 Instantiating and configuring the algorithms to test
        * 1.3.1.1 Model 0 - Baseline: Simple linear regression
        * 1.3.1.2 Model 1 - Linear models family: ElasticNet
        * 1.3.1.3 Model 2 - Support vector machine family: Support Vector Regressor
        * 1.3.1.4 Model 3 - Nearest Neighbors family: K Neighbors Regressor
        * 1.3.1.5 Model 4 - Ensemble family: Random Forest Regressor
    * 1.3.2 Performance comparison
    * 1.3.3 Conclusion : selected algorithm
- 1.4 Feature selection
    * 1.4.1 Setting up the features
        * 1.4.1.1 Initial features + the other most correlated features
        * 1.4.1.2 Initial features + usage proportions of the energy sources
        * 1.4.1.3 Initial features + localization data
        * 1.4.1.4 Initial features + the buildings' use types
        * 1.4.1.5 Initial features + the buildings' types
        * 1.4.1.6 Initial features + the buildings' main types
    * 1.4.2 Performance comparison
    * 1.4.3 Conclusion: selected features
- 1.5 Exploring the algorithm's hyperparameters
- 1.6 Model validation

**2. Predicting CO2 emissions**
- 2.1 Feature selection
- 2.2 Splitting the dataset
- 2.3 Algorith selection
    * 2.3.1 Instantiating and configuring the algorithms to test
        * 2.3.1.1 Model 0 - Baseline: Simple linear regression
        * 2.3.1.2 Model 1 - Linear models family: ElasticNet
        * 2.3.1.3 Model 2 - Support vector machine family: Support Vector Regressor
        * 2.3.1.4 Model 3 - Nearest Neighbors family: K Neighbors Regressor
        * 2.3.1.5 Model 4 - Ensemble family: Random Forest Regressor
    * 2.3.2 Performance comparison
    * 2.3.3 Conclusion : selected algorithm
- 2.4 Feature selection
    * 2.4.1 Setting up the features
        * 2.4.1.1 Initial features + the other most correlated features
        * 2.4.1.2 Initial features + localization data
        * 2.4.1.3 Initial features + the buildings' use types
        * 2.4.1.4 Initial features + the buildings' types
        * 2.4.1.5 Initial features + the buildings' main types
    * 2.4.2 Performance comparison
    * 2.4.3 Conclusion: selected features
- 2.5 Exploring the algorithm's hyperparameters
- 2.6 Model validation

**3. Conclusion**

_________

## Requirements

This assumes that you already have an environment allowing you to run Jupyter notebooks. 

The libraries used otherwise are listed in requirements.txt

_________

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv), and place it under sources/.

2. Run the following in your terminal to install all required libraries :

```bash
pip3 install -r requirements.txt
```

4. Run the notebooks in order (Notebook 1 first, then Notebook 2).
__________

## Results

For a complete presentation and commentary of the results of this analysis, please see the PowerPoint presentation.

> NB: The presentation is in French.