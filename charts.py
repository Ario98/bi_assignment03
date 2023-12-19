import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

data = pd.read_csv('data/airbnbDataset.csv')

columns = ['Price', 'Person Capacity', 'Cleanliness Rating', 'Guest Satisfaction', 
           'City Center (km)', 'Metro Distance (km)', 'Attraction Index', 'Restraunt Index']

for col in columns:
    plt.figure(figsize=(10, 6))

    sns.histplot(data[col], bins=30, kde=True, color='#92A1CF', edgecolor='black', alpha=0.7, line_kws={'lw': 2, 'color': 'blue'})
    plt.title(f'{col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    output_file_path = f'images/{col.replace(" ", "")}.png'
    plt.savefig(output_file_path)
    plt.close()
