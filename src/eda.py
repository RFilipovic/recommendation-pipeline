import matplotlib.pyplot as plt
import pandas as pd

def plot_long_tail_distribution(df):

    # 1. Group df by 'StockCode' and sum the 'Quantity'.
    item_popularity = df.groupby('StockCode')['Quantity'].sum()
    
    # 2. Sort the resulting Series in descending order.
    sorted_popularity = item_popularity.sort_values(ascending=False)
    
    # 3. Reset index to get a DataFrame with 'StockCode' and 'Quantity'.
    sorted_quantities = sorted_popularity.reset_index()
    
    # 4. Plot using matplotlib
    plt.plot(sorted_quantities['Quantity'].values)
    
    # 5. Set labels
    plt.xlabel("Items (ranked by popularity)")
    plt.ylabel("Total Quantity Sold")
    
    # 6. Set title
    plt.title("Long-Tail Distribution of Item Popularity")
    
    # 7. Show the plot
    plt.show()
