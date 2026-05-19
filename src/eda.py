import matplotlib.pyplot as plt
import pandas as pd

def plot_long_tail_distribution(df):
    """
    Plots the long-tail distribution of item popularity.
    
    Steps to implement:
    1. Group df by 'StockCode' and sum the 'Quantity'.
    2. Sort the resulting Series in descending order.
    3. Reset index to get a DataFrame with 'StockCode' and 'Quantity'.
    4. Plot using matplotlib (e.g., plt.plot(sorted_quantities['Quantity'].values)).
    5. Set labels: "Items (ranked by popularity)" for X, "Total Quantity Sold" for Y.
    6. Set title: "Long-Tail Distribution of Item Popularity".
    7. Save or show the plot.
    """
    # TODO: Implement long-tail plotting
    raise NotImplementedError
