############### Predicting The Energy Consumption Of Buildings #################

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from scipy import stats

##########################################
#                                        #
# Computation and description functions  #
#                                        #
##########################################

def getMissingValuesPercentPer(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column
        
        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed
        
        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''
    
    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})

    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']

    missing_percent_df['Total'] = 100

    percent_missing = data.isnull().sum() * 100 / len(data.columns)
    
    return missing_percent_df


#------------------------------------------

def descriptionJeuDeDonnees(sourceFiles):
    '''
        Outputs a presentation pandas dataframe for the dataset.
        
        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file
        
        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(sourceFiles)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in sourceFiles.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

        
    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    
    return presentation_df
    
#------------------------------------------

def eta_squared(data, x_qualit,y_quantit):
    '''
        Calculate the proportion of variance
        in the given quantitative variable for
        the given qualitative variable
        
        ----------------
        - data      : dataframe
                      Working data
        - x_quantit : The name of the qualitative variable
        - y_quantit : The name of the quantitative variable
        
        Returns
        ---------------
        Eta_squared : float
    '''
    
    sous_echantillon = data.copy().dropna(how="any")

    x = sous_echantillon[x_qualit]
    y = sous_echantillon[y_quantit]

    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
 
#------------------------------------------

def getLorenzGini(data):
    '''
        Calculate the lorenz curve and Gini coeff
        for a given variable
        
        ----------------
        - data       : data series
                       Working  data
        
        Returns
        ---------------
        A tuple containing :
        - lorenz_df  : list
                       The values for the Lorenz curve
        - gini_coeff : float
                       The associated Gini coeff
        
        Source : www.openclassrooms.com
    '''
    
    dep = data.dropna().values
    n = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

    #---------------------------------------------------
    # Gini :
    # Surface sous la courbe de Lorenz. Le 1er segment
    # (lorenz[0]) est à moitié en dessous de 0, on le
    # coupe donc en 2, on fait de même pour le dernier
    # segment lorenz[-1] qui est à 1/2 au dessus de 1.
    #---------------------------------------------------

    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    # surface entre la première bissectrice et le courbe de Lorenz
    S = 0.5 - AUC
    gini_coeff = [2*S]
         
    return (lorenz, gini_coeff)
    
#------------------------------------------

def getLorenzsGinis(data):
    '''
        Calculate the lorenz curve and Gini coeffs
        for all columns in the given dataframe
        
        ----------------
        - data       : dataframe
                       Working data
        
        Returns
        ---------------
        A tuple containing :
        - lorenz_df  : dataframne
                       The values for the Lorenz curve for each
                       column of the given dataframe
        - gini_coeff : dataframe
                       The associated Gini coeff for each column of
                       the given dataframe
    '''
    
    ginis_df = pd.DataFrame()
    lorenzs_df = pd.DataFrame()

    for ind_quant in data.columns.unique().tolist():
        lorenz, gini = getLorenzGini(data[ind_quant])
        ginis_df[ind_quant] = gini
        lorenzs_df[ind_quant] = lorenz

    n = len(lorenzs_df)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    lorenzs_df["index"]=xaxis[:-1]
    lorenzs_df.set_index("index", inplace=True)
    
    ginis_df = ginis_df.T.rename(columns={0:'Indice Gini'})
    
    return (lorenzs_df, ginis_df)

#------------------------------------------

def getPerformances(data_dict, y_col, r2_adjusted=None, std=None):
    '''
        Gives the performance in terms of RMSE, R2, adjusted R2 for
        the K Neighbors Regressor algorithm
        
        Parameters
        ----------------
        data_dict   : dict with :
                    - name of the input data as keys (string)
                    - input data as values (pandas dataframe)
                 
        y_col       : string
                      The name of the feature to predict
        
        r2_adjusted : bool
                      True if adjusted R2 should be adjusted
                
        Returns
        ---------------
        A tuple containing :
        - r2_rmse_time  : dataframne
                          The RMSE et R2 score values for each model
        - model_df      : dataframe
                          The associated Gini coeff for each column of
                          the given dataframe
    '''
    
    r2_rmse_time = pd.DataFrame(columns=["RMSE", "R2", "Time", "R2 Ajusté"])
    model_df = pd.DataFrame()

    for ajout_name, data_ajout in data_dict.items():
        
        xtrain, xtest, ytrain, ytest = train_test_split(
                                data_ajout.loc[:, data_ajout.columns != y_col],
                                data_ajout[[y_col]],
                                test_size=0.3)
                                
        if std == "standardized":
            # Standardisation des X
            std_scale = preprocessing.StandardScaler().fit(xtrain)
            X_train = std_scale.transform(xtrain)
            X_test = std_scale.transform(xtest)
        else:
            X_train = xtrain
            X_test = xtest
            
        ytrain_log = np.log(ytrain)
        ytest_log = np.log(ytest)

        parameters = [{'n_estimators': [100, 200, 500, 700, 1000]}]

        clf = GridSearchCV(RandomForestRegressor(), parameters)
    
        clf.fit(X_train, ytrain_log.values.ravel())

        model = RandomForestRegressor(n_estimators=clf.best_params_["n_estimators"],
                                   oob_score=True, random_state=4)
        
        start_time = time.time()

        model.fit(X_train, ytrain_log.values.ravel())
        ypred = model.predict(X_test)

        elapsed_time = time.time() - start_time

        model_df[ajout_name] = [model]
    
        r2_rmse_time.loc[ajout_name, "RMSE"] = np.sqrt(mean_squared_error(ytest_log, ypred))
        
        if r2_adjusted==True:
            r2_rmse_time.loc[ajout_name, "R2 Ajusté"] = 1 - (1-r2_score(ytest_log, ypred))*(len(ytest_log)-1)/(len(ytest_log)-X_test.shape[1]-1)
        else:
            r2_rmse_time.loc[ajout_name, "R2"] = r2_score(ytest_log, ypred)
        
        r2_rmse_time.loc[ajout_name, "Time"] = elapsed_time

    return r2_rmse_time, model_df

#------------------------------------------

def fitPredictPlot(algorithms, xtrain, xtrain_std, ytrain, xtest, xtest_std, ytest):
    '''
        For each given algorithms :
        - Trains it on (xtrain, ytrain)
        - Predicts the values for xtest
        - Calculate the RMSE and R2 score with ytest
        - Getst the calculation time
        
        It does the same for xtrain_std, ytrain, xtest_std, ytest.
        
        The function then plots for each algorithm and each type of input the predicted value
        as a function of the known value, and return the performance data for all models.
        
        Parameters
        ----------------
        algorithms  : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values
        
        - xtrain    : pandas dataframe
                      x training data
        - xtrain_std: pandas dataframe
                      standardized x training data
        - ytrain    : pandas dataframe
                      y training data
        - ytrain_std: pandas dataframe
                      standardized y training data
        - xtest     : pandas dataframe
                      x test data
        - xtest_std : pandas dataframe
                      standardized x test data
        - ytest     : pandas dataframe
                      y test data
                
        Returns
        ---------------
        r2_rmse_time   : pandas dataframe containing the RMSE, R2 score and calculation time for
        each algorithm
-
    '''
    
    r2_rmse_time = pd.DataFrame()
    
    # Set up for the plot
    TITLE_SIZE = 45
    SUBTITLE_SIZE = 25
    TITLE_PAD = 1.05
    TICK_SIZE = 25
    TICK_PAD = 20
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 5
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(20, 40))
    f.suptitle("Performances des modèles", fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for algoname, algo in algorithms.items():

        if "std" in algoname:
            X_train = xtrain_std
            X_test = xtest_std
        else:
            X_train = xtrain
            X_test = xtest
        
            algo.fit(X_train, ytrain.values.ravel())
            
            start_time = time.time()

            ypred = algo.predict(X_test)
            
        elapsed_time = time.time() - start_time

        r2_rmse_time.loc[algoname, "RMSE"] = np.sqrt(mean_squared_error(ytest, ypred))
        r2_rmse_time.loc[algoname, "R2"] = r2_score(ytest, ypred)
        r2_rmse_time.loc[algoname, "Time"] = elapsed_time

        # plot
        ax = axes[row, column]

        b = sns.regplot(x=ytest, y=ypred, ax=ax)

        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        ax.set_xlabel('Valeur mesurée', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        ax.set_ylabel('Valeur prédite', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)
        scores = (r'$R^2={:.2f}$' + '\n' + r'$RMSE={:.2f}$').format(r2_score(ytest, ypred),
                np.sqrt(mean_squared_error(ytest, ypred)))
        
        
        ax.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        ax.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")
        
            
        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
            
    return r2_rmse_time
 
#------------------------------------------
def getRhoPValueHeatmaps(data, interest_cols, FEATURE, THRESHOLD):
    '''
        Calculates the rho of Spearman for a given dataset
        and a given set of features compared to FEATURE.
        It returns the dataframe corresponding to a heatmap of the rho
        containing only the features whose rho is superior to THRESHOLD.
        
        Parameters
        ----------------
        data            : pandas dataframe
                          Data containing interest_cols and FEATURE as features.
                          All features must be numeric.
                                 
        interest_cols   : list
                          The names of the features to calculate the rho
                          of Spearman compared to FEATURE
        
        FEATURE         : string
                          The name of the feature to calculate the rho of Spearman
                          against.
        
        THRESHOLD       : float
                          The function will return only those features whose rho is
                          superior to THRESHOLD
        
        Returns
        ---------------
        sorted_corrs_df : pandas dataframe
                          The rhos of Spearnma, sorted by feature of highest rho
        sorted_ps_df    : pandas dataframe
                          The p-values associated to the calculated rhos
    '''
    
    # Calcul du rho de Spearman avec p-value
    corrs, ps = stats.spearmanr(data[FEATURE],
                                data[interest_cols])

    # Transformation des arrays en DataFrame
    corrs_df = pd.DataFrame(corrs)
    ps_df = pd.DataFrame(ps)

    # Renommage des colonnes
    interest_cols.insert(0, FEATURE)
    corrs_df.columns=interest_cols
    ps_df.columns=interest_cols

    # Suppression des colonnes dont le rho est < THRESHOLD
    x = corrs_df[corrs_df[FEATURE]>THRESHOLD].index

    corrs_df = corrs_df.iloc[x,x].reset_index(drop=True).sort_values([FEATURE], ascending=False)
    ps_df = ps_df.iloc[x,x].reset_index(drop=True)

    # Classement des colonnes par ordre de rho décroissant
    sort_rows_ps_df = pd.DataFrame()

    for x in corrs_df.index.tolist():
        sort_rows_ps_df = pd.concat([sort_rows_ps_df, pd.DataFrame(ps_df.iloc[x, :]).T])

    # Tri par ordre de plus grand rho
    sorted_corrs_df = pd.DataFrame()
    sorted_ps_df = pd.DataFrame()

    for x in corrs_df.sort_values([FEATURE], ascending=False).index.tolist():
        sorted_corrs_df = pd.concat([sorted_corrs_df, corrs_df.iloc[:,x]], axis=1)
        sorted_ps_df = pd.concat([sorted_ps_df, sort_rows_ps_df.iloc[:,x]], axis=1)

    sorted_corrs_df["Index"] = sorted_corrs_df.columns
    sorted_corrs_df.set_index("Index", inplace=True)
    sorted_ps_df["Index"] = sorted_ps_df.columns
    sorted_ps_df.set_index("Index", inplace=True)

    sorted_corrs_df.index.name = None
    sorted_ps_df.index.name = None

    return (sorted_corrs_df, sorted_ps_df)
 
##########################################
#                                        #
# Graphics Functions                     #
#                                        #
##########################################

def plotPercentageMissingValuesFor(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.
        
        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"
                                 
       long : int
            The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = getMissingValuesPercentPer(data).sort_values("Percent Filled").reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    b = sns.barplot(x="Total", y="index", data=data_to_plot,label="non renseignées", color="thistle", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    c = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,label="renseignées", color="darkviolet")
    c.set_xticklabels(c.get_xticks(), size = TICK_SIZE)
    c.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)

    ax.set(ylabel="Colonnes",xlabel="Pourcentage de valeurs (%)")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    plt.savefig('missingPercentagePerColumn.png')

    # Display the figure
    plt.show()

#------------------------------------------

def plotPieChart(data, long, larg, title):
    '''
        Plots a pie chart of the proportion of each modality for groupby_col
        with the dimension (long, larg), with the given title and saved figure
        title.
        
        Parameters
        ----------------
        data           : pandas dataframe
                         Working data, with a "groupby_col" column
        
        groupby_col    : string
                         The name of the quantitative column of which the modality
                         frequency should be plotted.
                                  
        long           : int
                         The length of the figure for the plot
        
        larg           : int
                         The width of the figure for the plot
        
        title          : string
                         title for the plot
        
        title_fig_save : string
                         title under which to save the figure
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 25
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    # Set figure title
    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Put everything in bold
    #plt.rcParams["font.weight"] = "bold"

    # Create pie chart for topics
    a = data.plot(kind='pie', autopct=lambda x:'{:2d}'.format(int(x)) + '%', fontsize =20)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal')
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotQualitativeDist(data, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.
        
        Parameters
        ----------------
        data : dataframe
               Working data containing exclusively qualitative data
                                 
        long : int
               The length of the figure for the plot
        
        larg : int
               The width of the figure for the plot
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 80
    TITLE_PAD = 1.05
    TICK_SIZE = 40
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 3
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VALEURS QUALITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        ax = axes[row, column]
        
        b = sns.countplot(y=ind_qual,
                          data=data_to_plot,
                          color="darkviolet",
                          ax=ax,
                          order = data_to_plot[ind_qual].value_counts().index)


        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        
        if ind_qual == "nova_group":
            ylabels = [item.get_text()[0] for item in ax.get_yticklabels()]
        else:
            ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
        b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        ly = ax.get_ylabel()
        ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))
        
        ax.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
    
#------------------------------------------

def plotBoxPlots(data, long, larg, nb_rows, nb_cols):
    '''
        Displays a boxplot for each column of data.
        
        Parameters
        ----------------
        data    : dataframe
                  Working data containing exclusively quantitative data
                                 
        long    : int
                  The length of the figure for the plot
        
        larg    : int
                  The width of the figure for the plot
               
        nb_rows : int
                  The number of rows in the subplot
        
        nb_cols : int
                  The number of cols in the subplot
                                  
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("VALEURS QUANTITATIVES - DISTRIBUTION", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
                    
#------------------------------------------

def plotLorenz(lorenz_df, long, larg):
    '''
        Plots a Lorenz curve with the given title
        
        Parameters
        ----------------
        - lorenz_df : dataframe
                      Working data containing the Lorenz values
                      one column = lorenz value for a variable
                      
        - long      : int
                      The length of the figure for the plot
        
        - larg      : int
                      The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")
    
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.title("VARIABLES QUANTITATIVES - COURBES DE LORENZ",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    sns.set_color_codes("pastel")
    
    b = sns.lineplot(data=lorenz_df, palette=sns.color_palette("hls", len(lorenz_df.columns)),
                    linewidth=5, dashes=False)
    
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

    b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    ax.set_xlabel("")

    # Add a legend and informative axis label
    leg = ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)


    # Display the figure
    plt.show()

#------------------------------------------

def plotCorrelationHeatMap(data,long, larg, title):
    '''
        Plots a heatmap of the correlation coefficients
        between the quantitative columns in data
        
        Parameters
        ----------------
        - data : dataframe
                 Working data
                 
        - corr : string
                 the correlation method ("pearson" or "spearman")
        
        - long : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 40
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle(title, fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(data, mask=np.zeros_like(data, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=data, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotRegplot(data_x, data_y, title, long, larg):
    '''
        Plots a regression plot of the columns col_y and
        col_x in data
        
        Parameters
        ----------------
        - data  : dataframe
                  Working data containing the col_x and col_y columns
        - col_x : string
                  the name of a column present in data
        - col_y : string
                  the name of a column present in data
        - title : string
                  the title of the figure
        - long  : int
                  the length of the figure
        - larg  : int
                  the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.regplot(x=data_x, y=data_y)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------

def plotEnergySourcesDistributionFor(data, criterion, long, larg):
    '''
        Plots the proportions of use per energy source in % for each value
        taken by criterion as a stacked horizontal bar chart
        
        Parameters
        ----------------
        data   : pandas dataframe with:
                    - an index named criterion containing the different criterion
                      values
                    - a column per energy source containing the cumulated % of
                      energy source use (the current energy source + the value for the
                      previous column) for that energy source for each criterion value
                 
                                 
        long   : int
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                                
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 30
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 30

    # Reset index to access the Seuil as a column
    data_to_plot = data.reset_index()

    sns.set(style="whitegrid")
    palette = sns.husl_palette(len(data.columns))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("RÉPARTITION DES SOURCES D'ÉNERGIE",
              fontweight="bold",fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Get the list of topics from the columns of data
    column_list = list(data.columns)

    # Create a barplot with a distinct color for each topic
    for idx, column in enumerate(reversed(column_list)):
        color = palette[idx]
        b = sns.barplot(x=column, y=criterion, data=data_to_plot, label=str(column), orient="h", color=color)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        _, ylabels = plt.yticks()
        b.set_yticklabels(ylabels, size=TICK_SIZE)

        
    # Add a legend and informative axis label

    ax.legend(bbox_to_anchor=(0,-0.6,1,0.2), loc="lower left", mode="expand",
              borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    ax.set(ylabel="Type du bâtiment principal",xlabel="% des sources d'énergie")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x))))

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plotBarplot(data, col_x, col_y, long, larg, title):
    '''
        Plots a horizontal bar plot
        
        Parameters
        ----------------
        data    : pandas dataframe
                  Working data containing col_x et col_y
        
        col_x   : string
                  Name of the features to use for x
                  
        col_y   : string
                  Name of the features to use for y
                 
        long    : int
                  The length of the figure for the plot
        
        larg    : int
                  The width of the figure for the plot
                                  
        title   : string
                  Title for the plot
                  
        Returns
        ---------------
        -
    '''
    
    LABEL_SIZE = 20
    LABEL_PAD = 15

    f, ax = plt.subplots(figsize=(larg, long))

    plt.title(title,
                  fontweight="bold",
                  fontsize=30, pad=10)

    b = sns.barplot(x=col_x, y=col_y,
                    data=data,
                    label="non renseignées",
                    color="darkviolet")

    b.set_xticklabels(b.get_xticks(), size = 20)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    _=b.set_yticklabels(ylabels, size=20)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
            
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

#------------------------------------------

def plot3Regplots(data, col_x1, col_x2, col_x3, col_y, long, larg, title):
    '''
        Plots 3 regplots horizontally in a single figure
        
        Parameters
        ----------------
        data   : pandas dataframe
                 Working data
                 
        col_x1 : string
                 Name of the feature to use for x in the 1st regplot
         
        col_x2 : string
                 Name of the feature to use for x in the 2nd regplot
                 
        col_x3 : string
                Name of the feature to use for x in the 3rd regplot
        
        col_y : string
                Name of the feature to use for y in the regplots
        
        long   : int
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                
        title   : string
                  The title for the plot
        
        Returns
        ---------------
        -
    '''
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(larg, long))

    plt.suptitle(title, fontweight="bold", fontsize=25)

    sns.regplot(x=data[col_x1], y=data[col_y], ax=ax1)
    sns.regplot(x=data[col_x2], y=data[col_y], ax=ax2)
    sns.regplot(x=data[col_x3], y=data[col_y], ax=ax3)

#------------------------------------------

def plot2Distplots(col_x1, label_col_x1, col_x2, label_col_x2, long, larg, title):
    '''
        Plots 2 distplots horizontally in a single figure
        
        Parameters
        ----------------
        col_x1          : pandas series
                          The data to use for x in the 1st distplot
                   
        label_col_x1    : string
                          The label for x in the 1st distplot
        
        col_x2          : pandas series
                          The data to use for x in the 2nd distplot
        
        label_col_x2    : string
                          The label for x in the 2nd distplot
        
        long            : int
                          The length of the figure for the plot
        
        larg            : int
                          The width of the figure for the plot
                
        title           : string
                          The title for the plot
        
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(long, larg))

    f.suptitle(title, fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.despine(left=True)

    b = sns.distplot(col_x1, ax=ax1)

    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    b.set_xlabel(label_col_x1,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    c = sns.distplot(col_x2, ax=ax2)

    c.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    c.set_xlabel(label_col_x2,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.setp((ax1, ax2), yticks=[])
    plt.setp((ax1, ax2), xticks=[])


    plt.tight_layout()
