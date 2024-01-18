
"""
Created on Mon Jan  15 20:59:27 2024

@author: Saman Qayyum
"""

import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err


def read_world_bank_csv(filename):
    """
    Read a World Bank CSV file and filter the dataset based on specified years.

    Parameters:
        filename (str): Path to the World Bank CSV file.

    Returns:
        Tuple of two dataframes:
        1. Dataframe with years as index and countries as columns.
        2. Dataframe with countries as index and years as columns.
    """

    # set year range filter the datasets
    start_from_year = 1960
    end_to_year = 2022

    # read all csv files using pandas, skipping the first 3 rows
    df = pd.read_csv(filename, skiprows=3, iterator=False)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(start_from_year, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    df_country_index = df.loc[:, all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as integer
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index


def one_silhoutte(xy, n):
    """
    Calculate silhouette score for n clusters.

    Parameters:
        xy (DataFrame): Dataframe containing x and y columns.
        n (int): Number of clusters.

    Returns:
        float: Silhouette score.
    """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data to the k-means clusterer
    kmeans.fit(xy)     # fit done on x,y pairs

    # Retrieve the cluster labels assigned by the k-means algorithm
    labels = kmeans.labels_

    # Calculate the silhouette score using scikit-learn's silhouette_score
    # The silhouette score measures how well-separated the clusters are,
    # with a higher score indicating better-defined clusters.
    score = skmet.silhouette_score(xy, labels)

    # Return the calculated silhouette score
    return score


def fit_polynomial(x, a, b, c, d, e):
    """
    Calculate polynomial.

    Parameters:
        x (float): Input value.
        a, b, c, d, e (float): Coefficients of the polynomial.

    Returns:
        float: Result of the polynomial function.
    """

    # Shift the input value by 1960
    x = x - 1960

    # Calculate the polynomial function
    # f(x) = a + b*x + c*x^2 + d*x^3 + e*x^4
    f = a + b*x + c*x**2 + d*x**3 + e*x**4

    # Return the result of the polynomial function
    return f


def cluster_and_visualize(df_cluster, n_clusters=3):
    """
    Perform clustering on the given dataframe and visualize the results.

    Parameters:
        df_cluster (DataFrame): Dataframe containing 'INF' and 'GDP' columns.
        n_clusters (int): Number of clusters for K-means.

    Returns:
        None
    """
    # Drop rows with NaN values
    df_cluster = df_cluster.dropna()

    # Normalize data for clustering using a custom 'scaler' function
    df_norm, df_min, df_max = ct.scaler(df_cluster)

    # calculate silhouette score for 2 to 10 clusters
    for ic in range(2, 11):
        score = one_silhoutte(df_cluster, ic)
        # allow for minus signs
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

    # Set up the K-means clusterer with the specified number of clusters
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)

    # Fit the normalized data to the K-means clusterer
    kmeans.fit(df_norm)  # fit done on x,y pairs

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract and backscale the estimated cluster centers
    cen = kmeans.cluster_centers_
    cen = ct.backscale(cen, df_min, df_max)

    # specify centers
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # extract x and y values of data points
    x = df_cluster["INF"]
    y = df_cluster["GDP"]

    # Plot the figure, select figure size
    plt.figure(figsize=(12, 8), dpi=300)

    # plot all three clusters
    scatter = plt.scatter(x, y, c=labels, cmap='brg', marker="o", s=150,
                          alpha=0.7, edgecolors='black', label="Clusters")

    # Plot K-means Centers
    plt.scatter(xkmeans, ykmeans, s=200, marker='o', c='black',
                label="Centroid")

    # Add legend to the plot, adjusting its position, size
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend = plt.legend(handles + [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='black', markersize=10)],
                        ["Cluster 1", "Cluster 2", "Cluster 3", "Centroid"],
                        loc='upper right', fontsize=15)

    # give title to graph and add labels, adjusting font size and colors
    plt.title('Inflation Rate vs GDP per capita of Pakistan', fontsize=20,
              fontweight='bold', c='darkred', x=0.5, y=1.01)
    plt.xlabel("Inflation rate (annual %)", fontsize=14)
    plt.ylabel("GDP per capita (current US$)", fontsize=14)

    # show grids in graph
    plt.grid(True)

    # Save the plot as png
    plt.savefig("Clustering.png")

    # Show the plot
    plt.show()


def fit_and_visualize(df_cluster):
    """
    Fit polynomial functions to inflation rate and GDP per capita
    and visualize the results.

    Parameters:
        df_cluster (DataFrame): Dataframe containing 'Year', 'INF',
        and 'GDP' columns.

    Returns:
        Tuple of parameters and covariance matrices for
        inflation rate and GDP fitting.
    """

    # Plotting for Inflation Rate
    df_cluster["Year"] = df_cluster.index

    # fit polynomial to inflation rate
    param_inf, covar_inf = opt.curve_fit(fit_polynomial,
                                         df_cluster["Year"], df_cluster["INF"])

    # Calculate fitted values using the obtained parameters
    df_cluster["fit_inf"] = fit_polynomial(df_cluster["Year"], *param_inf)

    # Plot the actual inflation rate and the fitted curve
    plt.figure(figsize=(12, 8), dpi=300)
    df_cluster.plot("Year", ["INF", "fit_inf"])

    # give title to graph and add labels, adjusting font size and colors
    plt.title('Inflation rate (annual %) fitting for Pakistan', fontsize=12,
              fontweight='bold', c='darkred', x=0.5, y=1.01)
    plt.xlabel("Years", fontsize=10)
    plt.ylabel("Inflation rate (annual %)", fontsize=10)

    # show grids in graph
    plt.grid(True)

    # Save the plot as png
    plt.savefig("INF_fit.png")

    # show the graph
    plt.show()

    # Plotting for GDP per capita
    df_cluster["Year"] = df_cluster.index

    # fit polynomial to GDP
    param_gdp, covar_gdp = opt.curve_fit(fit_polynomial, df_cluster["Year"],
                                         df_cluster["GDP"])

    # Calculate fitted values using the obtained parameters
    df_cluster["fit_gdp"] = fit_polynomial(df_cluster["Year"], *param_gdp)

    # Plot the actual GDP per capita and the fitted curve
    plt.figure(figsize=(12, 8), dpi=300)
    df_cluster.plot("Year", ["GDP", "fit_gdp"])

    # give title to graph and add labels, adjusting font size and colors
    plt.title('GDP per capita (current US$) fitting for Pakistan', fontsize=12,
              fontweight='bold', c='darkred', x=0.5, y=1.01)
    plt.xlabel("Years", fontsize=10)
    plt.ylabel("GDP per capita (current US$)", fontsize=10)

    # show grids in graph
    plt.grid(True)

    # Save the plot as png
    plt.savefig("GDP_fit.png")

    # Show the plot
    plt.show()

    # Return parameters and covariance matrices for inflation rate and GDP fitting
    return param_inf, covar_inf, param_gdp, covar_gdp


def forecast_and_visualize(df_cluster, param_inf, covar_inf, param_gdp, covar_gdp):
    """
    Generate forecasts for inflation rate and GDP per capita and
    visualize the results.

    Parameters:
        df_cluster (DataFrame): Dataframe containing 'Year', 'INF', and
        'GDP' columns.
        param_inf (array): Coefficients for the inflation rate polynomial.
        covar_inf (array): Covariance matrix for the inflation rate polynomial.
        param_gdp (array): Coefficients for the GDP polynomial.
        covar_gdp (array): Covariance matrix for the GDP polynomial.

    Returns:
        None
    """
    # Forecast Inflation Rate
    year_inf = np.arange(1960, 2030)
    forecast_inf = fit_polynomial(year_inf, *param_inf)

    # Error propagation for Inflation Rate forecast
    sigma_inf = err.error_prop(year_inf, fit_polynomial, param_inf, covar_inf)
    low_inf = forecast_inf - sigma_inf
    up_inf = forecast_inf + sigma_inf

    # Plot the figure, select figure size
    plt.figure(figsize=(12, 8), dpi=300)

    # Plot Inflation Rate original
    plt.plot(df_cluster["Year"], df_cluster["INF"],
             label="Inflation Rate (Original)", linewidth=2)

    # Plot Inflation Rate forcasted
    plt.plot(year_inf, forecast_inf, label="Inflation Rate Forecast",
             linestyle="--", linewidth=2)

    # Plot Inflation Rate forcasted confidence margin
    plt.fill_between(year_inf, low_inf, up_inf, color="yellow", alpha=0.7,
                     label="Confidence Margin")

    # give title to graph, adjusting font size and colors
    plt.title('Inflation Rate (annual %) Forecast for Pakistan', fontsize=17,
              x=0.5, y=1.01, fontweight='bold',
              color='darkred')

    # add labels, adjusting font size and colors
    plt.xlabel("Years", fontsize=14)
    plt.ylabel("Inflation Rate (annual %)", fontsize=14)

    # add legends to plot and adjust size
    plt.legend(fontsize=14)

    # show grids in graph
    plt.grid(True)

    # Save the plot as png
    plt.savefig("INF_forecast.png")

    # Show the plot
    plt.show()

    # Forecast GDP per capita
    year_gdp = np.arange(1960, 2030)
    forecast_gdp = fit_polynomial(year_gdp, *param_gdp)

    # Error propagation for GDP per capita forecast
    sigma_gdp = err.error_prop(year_gdp, fit_polynomial, param_gdp, covar_gdp)
    low_gdp = forecast_gdp - sigma_gdp
    up_gdp = forecast_gdp + sigma_gdp

    # Plot the figure, select figure size
    plt.figure(figsize=(12, 8), dpi=300)

    # Plot GDP per capita original
    plt.plot(df_cluster["Year"], df_cluster["GDP"],
             label="GDP per capita (Original)", linewidth=2)

    # Plot GDP per capita forecasted
    plt.plot(year_gdp, forecast_gdp, label="GDP per capita Forecast",
             linestyle="--", linewidth=2)

    # Plot GDP per capita forecasted confidence margin
    plt.fill_between(year_gdp, low_gdp, up_gdp, color="yellow", alpha=0.7,
                     label="Confidence Margin")

    # give title to graph, adjusting font size and colors
    plt.title('GDP per capita (current US$) Forecast for Pakistan', x=0.5,
              y=1.01, fontsize=17, fontweight='bold',
              color='darkred')

    # add labels, adjusting font size and colors
    plt.xlabel("Years", fontsize=14)
    plt.ylabel("GDP per capita (current US$)", fontsize=14)

    # add legends to plot and adjust size
    plt.legend(fontsize=14)

    # show grids in graph
    plt.grid(True)

    # Save the plot as png
    plt.savefig("GDP_forecast.png")

    # Show the plot
    plt.show()


# Main Function
# Read World Bank CSV files for inflation rate and GDP per capita
inf_data_yw, inf_data_cw = read_world_bank_csv("Inflation_rate.csv")
gdp_data_yw, gdp_data_cw = read_world_bank_csv("GDP_per_capita.csv")

# create a dataframe for clustering
df_cluster = pd.DataFrame()
df_cluster["INF"] = inf_data_yw["Pakistan"]
df_cluster["GDP"] = gdp_data_yw["Pakistan"]

# perform clustering and visualize results
cluster_and_visualize(df_cluster)


# fit polynomial functions and visualize results
param_inf, covar_inf, param_gdp, covar_gdp = fit_and_visualize(df_cluster)

# generate forecasts and visualize results
forecast_and_visualize(df_cluster, param_inf, covar_inf, param_gdp, covar_gdp)
