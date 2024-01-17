#!/usr/bin/env python
# coding: utf-8

# # Change in Rural Population over Time (1960-2022)

# In[1]:


# https://data.worldbank.org/indicator/SP.RUR.TOTL


# In[2]:


import pandas as pd


# In[3]:


def read_data(filename):
    """
    Function to read World Bank data.
    Returns:
    Two dataframes
    - countries_df: Original Data.
    - years_df: A transposed DataFrame of Original Data.
    
    """
    countries_df = pd.read_csv(filename, skiprows=4)
    years_df = countries_df.T
    years_df.columns = years_df.iloc[0]

    return years_df, countries_df


# In[4]:


years_df, countries_df = read_data('API_SP.RUR.TOTL_DS2_en_csv_v2_6302603.csv')


# In[6]:


countries_df.head(3)


# In[7]:


selected_years = list(map(str, range(1960, 2023)))


# In[9]:


df = countries_df[['Country Name', 'Indicator Name'] + selected_years]
df = df.dropna()
df.shape


# In[10]:


df.head(3)


# In[16]:


subset = df[["Country Name", "1960"]].copy()
subset.head(3)


# In[17]:


subset = subset.assign(Change = lambda x: 100.0 * (df["2022"] - df["1960"]) / df["1960"])
subset = subset.dropna()
subset.shape


# In[18]:


subset.describe()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
plt.scatter(subset["1960"], subset["Change"], 10, label="Rural population")
plt.xlabel("Rural population in 1960")
plt.ylabel("Percentage Change from 1960 to 2022")
plt.title("Rural population in 1960 vs. Percentage Change from 1960 to 2022")
plt.legend()
plt.show()


# In[25]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
subset2 = subset[["1960", "Change"]]
scaler.fit(subset2)
subset_norm = scaler.transform(subset2)


# In[29]:


from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import warnings
warnings.filterwarnings("ignore")

for i in range(2, 10):
    score = one_silhouette(subset_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[30]:


kmeans = KMeans(n_clusters=4, n_init=20)
kmeans.fit(subset_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[33]:


plt.figure(figsize=(7, 5))
plt.scatter(subset["1960"], subset["Change"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 30, "k", marker="d")
plt.xlabel("Rural population in 1960")
plt.ylabel("Percentage Change from 1960 to 2022")
plt.title("Clustes Representation")
plt.show()


# In[34]:


def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.
    """
    
    # initiate sigma the same shape as parameter
    var = np.zeros_like(x)   # initialise variance vector
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respect to the jth parameter
            deriv2 = deriv(x, func, parameter, j)
            # multiplied with the i-jth covariance
            # variance vector 
            var = var + deriv1 * deriv2 * covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta. Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.
    """

    # create vector with zeros and insert delta value for the relevant parameter
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """
    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar / matrix
    
    return corr


# In[43]:


us_rural = years_df.loc['1960':'2022', ['United States']].reset_index().rename(columns={'index': 'Year', 'United States': 'Rural population'})
us_rural = us_rural.apply(pd.to_numeric, errors='coerce')


# In[44]:


us_rural.describe()


# In[45]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=us_rural, x='Year', y='Rural population')
plt.xlabel('Year')
plt.ylabel('Rural population')
plt.title('Rural population in United States between 1960-2022')
plt.show()


# In[54]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[55]:


import scipy.optimize as opt
import numpy as np

param, covar = opt.curve_fit(poly, us_rural["Year"], us_rural["Rural population"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(1960, 2050)
forecast = poly(year, *param)
sigma = error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
us_rural["fit"] = poly(us_rural["Year"], *param)


# In[56]:


plt.figure(figsize=(8, 5))
plt.plot(us_rural["Year"], us_rural["Rural population"], label="Rural population")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("Rural population Trend Prediction - United States")
plt.xlabel("Year")
plt.ylabel("Rural population")
plt.legend()
plt.show()


# In[ ]:




