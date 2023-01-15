
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_worldbank(filename: str):
  """
    Reads a file containing world bank data and returns the original dataframe, the dataframe with countries as columns, and the dataframe with year as columns.
    
    Parameters:
    - filename (str): The name of the file to be read, including the file path.
    
    Returns:
    - dataframe (pandas dataframe): The original dataframe containing the data from the file.
    - df_transposed_country (pandas dataframe): Dataframe with countries as columns.
    - df_transposed_year (pandas dataframe): Dataframe with year as columns.
  """
  # Read the file into a pandas dataframe
  dataframe = pd.read_csv(filename)
    
  # Transpose the dataframe
  df_transposed = dataframe.transpose()
    
  # Populate the header of the transposed dataframe with the header information 
   
  # silice the dataframe to get the year as columns
  df_transposed.columns = df_transposed.iloc[1]

  # As year is now columns so we don't need it as rows
  df_transposed_year = df_transposed[0:].drop('year')
    
  # silice the dataframe to get the country as columns
  df_transposed.columns = df_transposed.iloc[0]
    
  # As country is now columns so we don't need it as rows
  df_transposed_country = df_transposed[0:].drop('country')
    
  return dataframe, df_transposed_country, df_transposed_year

# load data from World Bank website or a similar source
df, df_country, df_year = read_worldbank('wb_climatechange.csv')

def remove_null_values(feature):
  """
  This function removes null values from a given feature.


  Parameters:
    feature (pandas series): The feature to remove null values from.

  Returns:
    numpy array: The feature with null values removed.
  """
  # drop null values from the feature
  return np.array(feature.dropna())

def balance_data(df):
  """
  This function takes a dataframe as input and removes missing values from each column individually.
  It then returns a balanced dataset with the same number of rows for each column.

  Input:

  df (pandas dataframe): a dataframe containing the data to be balanced
  Output:

  balanced_df (pandas dataframe): a dataframe with the same number of rows for each column, after removing missing values from each column individually
  """
  # Making dataframe of all the feature in the avaiable in 
  # dataframe passing it to remove null values function 
  # for dropping the null values 
  access_to_electricity = remove_null_values(df[['access_to_electricity']])

  argicultural_land = remove_null_values(df[['agricultural_land']])

  co2_emission = remove_null_values(df[['co2_emission']])

  arable_land = remove_null_values(df[['arable_land']])

  electric_power_comsumption = remove_null_values(df[['electric_power_comsumption']])

  forest_area = remove_null_values(df[['forest_area']])

  population_growth = remove_null_values(df[['population_growth']])

  urban_population = remove_null_values(df[['urban_population']])

  GDP = remove_null_values(df[['GDP']])

  min_length = min(len(access_to_electricity), len(argicultural_land), len(co2_emission),len(arable_land), len(electric_power_comsumption), len(forest_area),
                   len(population_growth), len(urban_population), len(GDP))
  # after removing the null values we will create datafram 

  clean_data = pd.DataFrame({ 
                                'country': [df['country'].iloc[x] for x in range(min_length)],
                                'year': [df['year'].iloc[x] for x in range(min_length)],
                                'access_to_electricity': [access_to_electricity[x][0] for x in range(min_length)],
                                'argicultural_land': [argicultural_land[x][0] for x in range(min_length)],
                                 'co2_emission': [co2_emission[x][0] for x in range(min_length)],
                                 'arable_land': [arable_land[x][0] for x in range(min_length)],
                                 'electric_power_comsumption': [electric_power_comsumption[x][0] for x in range(min_length)],
                                 'forest_area': [forest_area[x][0] for x in range(min_length)],
                                 'population_growth': [population_growth[x][0] for x in range(min_length)],
                                 'urban_population': [urban_population[x][0] for x in range(min_length)],
                                 'GDP': [GDP[x][0] for x in range(min_length)]
                                 })
  return clean_data

# Clean and preprocess the data
df = balance_data(df)

df

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['access_to_electricity', 'argicultural_land', 'co2_emission',
                                      'arable_land', 'electric_power_comsumption', 'forest_area',
                                      'population_growth', 'urban_population', 'GDP']])

# Use KMeans to find clusters in the data
kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_data)

# Add the cluster assignments as a new column to the dataframe
df['cluster'] = kmeans.labels_

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['access_to_electricity'], cluster_data['GDP'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Population Growth')
plt.ylabel('GDP')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['co2_emission'], cluster_data['population_growth'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Co2 Emission')
plt.ylabel('Population Growth')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['argicultural_land'], cluster_data['forest_area'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Argicultural Land')
plt.ylabel('Forest Area')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

df.country.unique()

ku = df[df['country'] == 'Kuwait']
# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = ku[ku['cluster'] == i]
    plt.scatter(cluster_data['argicultural_land'], cluster_data['forest_area'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Argicultural Land')
plt.ylabel('Forest Area')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

china = df[df['country'] == 'China']
# create a plot showing the clusters and cluster centers using pyplot
for i in range(5):
    cluster_data = china[china['cluster'] == i]
    plt.scatter(china['argicultural_land'], china['forest_area'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Argicultural Land')
plt.ylabel('Forest Area')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

def linear(x, a, b):
  """ Simple linear function calculating a + b*x. """
  f = a + b*x
  return f

# select a country from cluster 0
country_data = df[(df['cluster'] == 1)]

# extract the year and co2_emission columns as separate arrays
years = df['year']
GDP = country_data['GDP']

# Use err_ranges function to estimate lower and upper limits of the confidence range
x = country_data['year']
y = country_data['co2_emission']

popt, pcov = curve_fit(linear, x, y)
x_pred = np.linspace(2021, 2040, 20)
# calculate the standard deviation for each parameter
sigma = np.sqrt(np.diag(pcov))
y_pred, y_pred_err = err_ranges(x_pred, linear, popt,sigma)

# Use pyplot to create a plot showing the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, linear(x, *popt), '-', label='fit')
plt.fill_between(x_pred, y_pred, y_pred_err, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()
