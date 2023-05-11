"""
@author: Dileepa Joseph Jayamanne
Student ID: 22031359
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err
sns.set()


def Extract_DF(file_name, indicator):
    """
    This function reads the world bank data csv file and extract a 
    dataframe with the specified indicator with the records from 1990-2019.
    
    file_name: Name of the csv file:  "climate-change.csv" of world bank data
    indicator: The indicator code is required to extract the dataframe. 
    For e.g.
    CO2 emmission (kt) : EN.ATM.CO2E.KT
    Total Popuation    : SP.POP.TOT

    """
    # Extracting data from the csv file given by file_name
    DF = pd.read_csv(file_name, sep=',', skiprows=(3))

    # Dropping Country Code, Indicator Name, Unnamed columns
    DF.drop(DF.columns[[1, 2, 66]], axis=1, inplace=True)

    # Renaming columns: Country Name to 'Country' and Indicator Code to
    # 'Indicator'
    DF.rename(columns={'Country Name': 'Country',
                       'Indicator Code': 'Indicator'},
              inplace=True)

    #Extracting CO2 emissions (metric tons per capita) from DF and saving
    # in DF_01 setting the country as the index
    DF_01 = DF[DF['Indicator'] == indicator].reset_index(
        drop=True).set_index('Country')

    #Extracting the records from DF_01 filtered from 1990 to 2019
    return DF_01.loc[:, '1990':'2019']


#Drop rows if any value in the row has a nan
CO2_emission = Extract_DF('climate-change.csv',
                          'EN.ATM.CO2E.KT').dropna(how='any')
Total_population = Extract_DF(
    'climate-change.csv', 'SP.POP.TOTL').dropna(how='any')


# Obtaninng transopse of the two dataframes and cleaning.
CO2_emission_transposed = Extract_DF('climate-change.csv',
                                     'EN.ATM.CO2E.KT').T.dropna(how='any')
Total_population_transposed = Extract_DF(
    'climate-change.csv', 'SP.POP.TOTL').T.dropna(how='any')


# Merge dataframes with inner join
Merged_DF = pd.merge(Total_population, CO2_emission,
                     left_index=True, right_index=True)


# Calculating CO2 emission per head in 1990
x = Merged_DF['1990_y']/Merged_DF['1990_x']

# Calculating CO2 emission per head in 2019
y = Merged_DF['2019_y']/Merged_DF['2019_x']
df = pd.DataFrame({'x': x, 'y': y})

# Scatterplot of x and y
plt.figure()
plt.scatter(df['x'], df['y'], c='blue', marker='o')
plt.xlabel('CO$_2$ emissions per head in 1990', fontweight="bold")
plt.ylabel('CO$_2$ emissions per head in 2019', fontweight="bold")
plt.title('Comparison of CO$_2$ emissions per head in countries in 1990\
          and 2019', fontweight="bold")


def kmeans_clustering(ncluster, dframe):
    """
    This function will perform k-means clustring and return 
    cluster centers, clluster labels and silhouette score
    
    Arguments: 
        ncluster: Number of clusters
        dframe: A data frame containing features
        for example: 
            1st column of dframe can be CO2 emissions per head in 1990
            2nd column can be CO$_2$ emissions per head in 2019
    """
    # Setting a seed to get the reproducible results
    kmeans = cluster.KMeans(n_clusters=ncluster, random_state=0)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(dframe)  # fit done on x,y pairs
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    score = skmet.silhouette_score(dframe, labels)  # Silhouette score
    # print(score)
    return cen, labels, score


# Normalizing the data
df_norm, df_min, df_max = ct.scaler(df)

#-------Finding the best number of clusters based on Silhouette score----------

nclusters = np.arange(2, 31, 1)  # Number of clusters to consider
scores_list = []  # To store the silhouette scores of for k clusters
max_score = 0
for k in nclusters:
    cen, labels, score = kmeans_clustering(k, df_norm)
    scores_list.append(score)
    if score > max_score:
        max_score = score
        ncluster = k
    print(f"n = {k}, silhouette score = {score}")
    np_scores = np.asarray(scores_list)

plt.figure()
plt.plot(nclusters, np_scores, '-o')
plt.xlabel('Number of Clusters', fontweight="bold")
plt.ylabel('Silhouette Score', fontweight="bold")
plt.title('Silhouette Score vs. Number of Clusters', fontweight="bold")
plt.savefig('kmeans_01.png', dpi=300, bbox_inches="tight")

print("")
# set up the clusterer with the number of expected clusters
print(f"Best number of clusters: {ncluster}")
#------------------------------------------------------------------------------

# Now, applying k-means clustering with best number of clusters
cen, labels, score = kmeans_clustering(ncluster, df_norm)

# For plotting
x = df['x']  # CO2 emissions per head in 1990
y = df['y']

# plot using the labels to select colour
col = ["tab:blue", "tab:orange", "tab:green", "tab:red",
       "tab:purple", "tab:brown",
       "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

plt.figure()
plt.scatter(x, y, c='blue', marker='o')
plt.xlabel('CO$_2$ emissions per head in 1990', fontweight="bold")
plt.ylabel('CO$_2$ emissions per head in 2019', fontweight="bold")
plt.title('Comparison of CO$_2$ emissions per head in countries in 1990\
          and 2019', fontweight="bold")

#move the cluster centres to the original scale
cen = ct.backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]

for l in range(ncluster):  # loop over the different labels
    plt.plot(x[labels == l], y[labels == l], "o", markersize=5, color=col[l])

# show cluster centres in black diamonds in the original scatterplot
plt.plot(xcen, ycen, 'kd')
plt.savefig('kmeans_02.png', dpi=300, bbox_inches="tight")

# df dataframe contains x, y and cluster membership.
df["cluster membership"] = labels


#------------------------------------------------------------------------------
# Subsetting the dataframe (df) to identify data with the countries
# with dissimilar CO2 emissions seen in 1990 and 2019 and saving that in DF

countries = pd.DataFrame(
 {'Countries': df[df['cluster membership'] == 1].index}).set_index('Countries')

# Extracting Forest land and agricultural land percentages from
# climate-change.csv
Forest_land = Extract_DF('climate-change.csv',
                         'AG.LND.FRST.ZS').dropna(how='any')
Agricultural_land = Extract_DF(
    'climate-change.csv', 'AG.LND.AGRI.ZS').dropna(how='any')

FL_1 = pd.merge(countries, Forest_land, left_index=True, right_index=True)
AL_1 = pd.merge(countries, Agricultural_land,
                left_index=True, right_index=True)

# Merging the dataframes with inner join
Merged = pd.merge(AL_1, FL_1, left_index=True,
                  right_index=True)  # AL_1:x FL_1:y
AL = Merged['2019_x']
FL = Merged['2019_y']
plt.figure()
plt.plot(AL, FL, "o", color='blue')
plt.xlabel('Agricultural land%', fontweight="bold")
plt.ylabel('Forest land%', fontweight="bold")
plt.title('Countries with CO2 per head', fontweight="bold")

DF = pd.DataFrame({'x': AL, 'y': FL})
DF.keys()

# K-means clustering is applied again on forest land and agricultural land
# percentages of the set of countries with dissimilar CO2 emissions seen
# in 1990 and 2019

# Normalizing the data
df_norm, df_min, df_max = ct.scaler(DF)

# Finding the best number of clusters based on Silhouette score
nclusters = np.arange(2, 10, 1)  # Number of clusters to consider [2,3,...9]
scores_list = []  # To store the silhouette scores of for k clusters

max_score = 0
for k in nclusters:
    cen, labels, score = kmeans_clustering(k, df_norm)
    scores_list.append(score)
    if score > max_score:
        max_score = score
        ncluster = k
    print(f"n = {k}, silhouette score = {score}")
    np_scores = np.asarray(scores_list)

plt.figure()
plt.plot(nclusters, np_scores, '-o')
plt.xlabel('Number of Clusters', fontweight="bold")
plt.ylabel('Silhouette Score', fontweight="bold")
plt.title('Silhouette Score vs. Number of Clusters', fontweight="bold")
plt.savefig('kmeans_03.png', dpi=300, bbox_inches="tight")

print("")
# set up the clusterer with the number of expected clusters
print(f"Best number of clusters: {ncluster}")

# Applying k-means algorithm using the best number of clusters
cen, labels, score = kmeans_clustering(ncluster, df_norm)

# Plotting
x = DF['x']
y = DF['y']

# plot using the labels to select colour
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
       "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

fig = plt.figure()
plt.scatter(x, y, c='blue', marker='o')
plt.xlabel('Agricultural land% in 2019', fontweight="bold")
plt.ylabel('Forest land% in 2019', fontweight="bold")
plt.title(
    'Countries with dissimilar CO$_{2}$ emissions per head in 2019 compared\
        to 1990', fontweight="bold")

#move the cluster centres to the original scale
cen = ct.backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]

for l in range(ncluster):  # loop over the different labels
    plt.plot(x[labels == l], y[labels == l], "o", markersize=5, color=col[l])

# show cluster centres in black diamonds in the original scatterplot
plt.plot(xcen, ycen, 'kd')
plt.savefig('kmeans_04.png', dpi=300, bbox_inches="tight")

DF["cluster membership"] = labels

# DF dataframe contains x, y, and cluster membership.
# This x refers to  Agricultural land% and y refers to Forest land%

#------------- Subsetting countries from the above the clusters ---------------

#countries represented by blue
Blue_countries = DF[DF['cluster membership'] == 0]
#countries represented by orange
Orange_countries = DF[DF['cluster membership'] == 1]
#countries represented by green
Green_countries = DF[DF['cluster membership'] == 2]
#countries represented by red
Red_countries = DF[DF['cluster membership'] == 3]
#countries represented by purple
Purple_countries = DF[DF['cluster membership'] == 4]

#----------------------- END of CLustering ------------------------------------

#------------------------Curve fitting-----------------------------------------
Country_List = ['France', 'Canada', 'Belgium',
                'Israel', 'United Kingdom', 'Korea, Rep.']

# Countries with similar CO2 emissions per head in 1990 and 2019
# e.g. France: is a country

# Countries with dissimilar CO2 emissions per head in 1990 and 2019
# Examples are as folows:
# Korea, Rep : Belongs to cluster with cluster membership 4 (Purple)
# Canada : Belongs to cluster with cluster membership 3 (Red)
# Israel: Belongs to cluster with cluster membership 1 (Orange)
# United Kingdom: Belongs to cluster with cluster membership 1 (Green)
# Belgium: Belongs to cluster with cluster membership 0 (Belgium)

# The above countries are chosen by inspecting dataframes such as
# Blue_countries, Orange_countries, Green_countries, Red_countries and
# Purple_countries containing countries in previous clustering result

# Now, let's forecast time series such as CO2 emission,
# Agricultural land percentage and Forest land percentage for next 10 years

for country in Country_List:

    CO2 = CO2_emission.loc[country, :].astype(float)
    AL = Agricultural_land.loc[country, :].astype(float)
    FL = Forest_land.loc[country, :].astype(float)

    #CO2 emmission times series
    years = np.arange(1990, 2020)
    plt.figure()
    plt.plot(years, CO2, c='blue', marker='o')
    plt.xlabel('Years')
    plt.ylabel('CO$_2$ emissions (kt)')
    plt.title(f'CO$_2$ emissions (kt) of {country} from 1990–2019')

    # Agricultural land% time series
    years = np.arange(1990, 2020)
    plt.figure()
    plt.plot(years, AL, c='blue', marker='o')
    plt.xlabel('Years')
    plt.ylabel('Agricultural land%')
    plt.title(f'Agricultural land% of {country} from 1990–2019')

    # Forest land% time series
    years = np.arange(1990, 2020)
    plt.figure()
    plt.plot(years, FL, c='blue', marker='o')
    plt.xlabel('Years')
    plt.ylabel('Forest land%')
    plt.title(f'Forest land% of {country} from 1990–2019')


    def poly(x, a, b, c):
        """ Calulates polynominal used to fit the time series"""
        x = x - 1990
        f = a + b*x + c*x**2
        return f


    #------------- CO2 emission time series with forecasting ------------------

    feature = 'CO$_2$ emission (kt)'
    # This is the observation period
    years = np.arange(1990, 2020)
    # Obtaining the parameters of the model by fitting a quadratic function
    # in above period
    param, pcovar = opt.curve_fit(poly, years, CO2)  # func, x, y
    sigma = np.sqrt(np.diag(pcovar))   # Calculating sigma
    print(f'feature = {feature}')
    print(f'sigma = {sigma}')
    print('parameters of the quadratic function: ')
    print({*param})

    # This period contains the future years as well.
    over_all_years = np.arange(1990, 2031)
    # Fitting the function for the period 1990:2025
    forecast = poly(over_all_years, *param)

    plt.figure()
    plt.plot(years, CO2, c='blue', linestyle='-', label=f'{feature}')
    plt.plot(over_all_years, forecast, label='Forecast', c='orange')
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel(f'{feature}', fontweight="bold")
    plt.title(f'{feature} in {country} from 1990–2030', fontweight="bold")
    plt.legend()

    # Calculating upper and lower limits of the confidence interval of
    # the errors
    low, up = err.err_ranges(over_all_years, poly, param, sigma)
    # err_ranges(x, func, param, sigma)

    # Obtaining the envelop of errors in the forecast
    plt.fill_between(over_all_years, low, up, color="yellow", alpha=0.7)
    plt.savefig(f'{country}_01.png', dpi=300, bbox_inches="tight")

    #----------Agricultural Land% timeseries with forecasting------------------

    feature = 'Agricultural land%'
    years = np.arange(1990, 2020)    # This is the observation period
    # Obtaining the parameters of the model by fitting a quadratic function
    # in above period

    param, pcovar = opt.curve_fit(poly, years, AL)  # func, x, y
    sigma = np.sqrt(np.diag(pcovar))   # Calculating sigma
    print(f'feature = {feature}')
    print(f'sigma = {sigma}')
    print('parameters of the quadratic function: ')
    print({*param})

    # This period contains the future years as well.
    over_all_years = np.arange(1990, 2031)
    # Fitting the function for the period 1990:2025
    forecast = poly(over_all_years, *param)

    plt.figure()
    plt.plot(years, AL, c='blue', linestyle='-', label=f'{feature}')
    plt.plot(over_all_years, forecast, label='Forecast', c='orange')
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel(f'{feature}', fontweight="bold")
    plt.title(f'{feature} in {country} from 1990–2030', fontweight="bold")
    plt.legend()

    # Calculating upper and lower limits of the confidence interval of
    # the errors
    low, up = err.err_ranges(over_all_years, poly, param, sigma)
    #err_ranges(x, func, param, sigma)

    # Obtaining the envelop of errors in the forecast
    plt.fill_between(over_all_years, low, up, color="yellow", alpha=0.7)
    plt.savefig(f'{country}_02.png', dpi=300, bbox_inches="tight")

    #----------------Forest Land% time series with forecasting-----------------

    feature = 'Forest land%'
    # This is the observation period
    years = np.arange(1990, 2020)
    # Obtaining the parameters of the model by fitting a quadratic function
    # in above period

    param, pcovar = opt.curve_fit(poly, years, FL)  # func, x, y
    print(f'feature = {feature}')
    print(f'sigma = {sigma}')
    print('parameters of the quadratic function: ')
    print({*param})

    over_all_years = np.arange(1990, 2031)
    # Fitting the function for the period 1990:2025
    forecast = poly(over_all_years, *param)
    # This period contains the future years as well.
    FL_future = forecast

    plt.figure()
    plt.plot(years, FL, c='blue', linestyle='-', label=f'{feature}')
    plt.plot(over_all_years, forecast, label='Forecast', c='orange')
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel(f'{feature}', fontweight="bold")
    plt.title(f'{feature} in {country} from 1990–2030', fontweight="bold")
    plt.legend()

    # Calculating upper and lower limits of the confidence interval
    # of the errors
    low, up = err.err_ranges(over_all_years, poly, param, sigma)
    # err_ranges(x, func, param, sigma)

    # Obtaining the envelop of errors in the forecast
    plt.fill_between(over_all_years, low, up, color="yellow", alpha=0.7)
    plt.savefig(f'{country}_03.png', dpi=300, bbox_inches="tight")
    plt.show()
    #-----------------------END of Code----------------------------------------
