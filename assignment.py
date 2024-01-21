# coding: utf-8
# # Change in Rural Population over Time (1960-2022)


import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import errors
import warnings
warnings.filterwarnings("ignore")


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


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


years_df, countries_df = read_data(
    'API_SP.RUR.TOTL_DS2_en_csv_v2_6302603.csv')
countries_df.head(3)
selected_years = list(map(str, range(1960, 2023)))
df = countries_df[['Country Name', 'Indicator Name'] + selected_years]
df = df.dropna()
df.shape
df.head(3)


subset = df[["Country Name", "1960"]].copy()
subset.head(3)
subset = subset.assign(
    Change = lambda x: 100.0 * (df["2022"] - df["1960"]) / df["1960"])
subset = subset.dropna()
subset.shape
subset.describe()

plt.figure(figsize=(7, 5))
plt.scatter(subset["1960"], subset["Change"], 10, label="Rural population")
plt.xlabel("Rural population in 1960")
plt.ylabel("Percentage Change from 1960 to 2022")
plt.title("Rural population in 1960 vs. Percentage Change from 1960 to 2022")
plt.legend()
plt.show()

scaler = StandardScaler()
subset2 = subset[["1960", "Change"]]
scaler.fit(subset2)
subset_norm = scaler.transform(subset2)
for i in range(2, 10):
    score = one_silhouette(subset_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")

kmeans = KMeans(n_clusters=4, n_init=20)
kmeans.fit(subset_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]

plt.figure(figsize=(7, 5))
plt.scatter(subset["1960"], subset["Change"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 30, "k", marker="d")
plt.xlabel("Rural population in 1960")
plt.ylabel("Percentage Change from 1960 to 2022")
plt.title("Clustes Representation")
plt.show()

us_rural = \
    years_df.loc['1960':'2022', ['United States']].reset_index().rename(
        columns={'index': 'Year', 'United States': 'Rural population'})
us_rural = us_rural.apply(pd.to_numeric, errors='coerce')
us_rural.describe()

plt.figure(figsize=(8, 5))
sns.lineplot(data=us_rural, x='Year', y='Rural population')
plt.xlabel('Year')
plt.ylabel('Rural population')
plt.title('Rural population in United States between 1960-2022')
plt.show()

param, covar = opt.curve_fit(
    poly, us_rural["Year"], us_rural["Rural population"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(1960, 2050)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
us_rural["fit"] = poly(us_rural["Year"], *param)

plt.figure(figsize=(8, 5))
plt.plot(
    us_rural["Year"], us_rural["Rural population"], label="Rural population")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("Rural population Trend Prediction - United States")
plt.xlabel("Year")
plt.ylabel("Rural population")
plt.legend()
plt.show()