import os
import requests
import zipfile
import pandas as pd
from imdb import Cinemagoer

# Define the URL and destination paths
url = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
destination_directory = "your_directory_path"  # Replace with your directory path
zip_path = os.path.join(destination_directory, "ml-25m.zip")
extract_path = os.path.join(destination_directory, "ml-25m")  # Directory to extract files

# Create the directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)
os.makedirs(extract_path, exist_ok=True)

# Download the file
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"File downloaded and saved to {zip_path}")
else:
    print("Failed to download the file.")
    exit()

# Extract the contents of the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    print(f"Files extracted to {extract_path}")

# Check for the extracted files
movies_csv_path = os.path.join(extract_path, "ml-25m/movies.csv")
ratings_csv_path = os.path.join(extract_path, "ml-25m/ratings.csv")
links_csv_path = os.path.join(extract_path, "ml-25m/links.csv")

if os.path.exists(movies_csv_path) and os.path.exists(ratings_csv_path) and os.path.exists(links_csv_path):
    print("movies.csv, ratings.csv, and links.csv have been extracted successfully.")
else:
    print("Required files are not found in the extracted directory.")
    exit()

# Load the data into DataFrames
movies_df = pd.read_csv(movies_csv_path)
ratings_df = pd.read_csv(ratings_csv_path)
links_df = pd.read_csv(links_csv_path)

# Merge the DataFrames
temp_df = pd.merge(ratings_df, movies_df, on='movieId')
combined_df = pd.merge(temp_df, links_df, on='movieId')
df_sorted = combined_df.sort_values(by='userId')

# Select data for the first 500 unique users
df_filtered = df_sorted[df_sorted['userId'] <= 500]

# Reset index for clean DataFrame
df_filtered = df_filtered.reset_index(drop=True)

# Initialize IMDb Cinemagoer
ia = Cinemagoer()
plot_cache = {}

def get_full_plot(imdb_id):
    # Check if the plot is already in the cache
    if imdb_id in plot_cache:
        return plot_cache[imdb_id]
    
    try:
        # Fetch movie details using IMDb ID
        movie = ia.get_movie(imdb_id)
        # Get the plot
        plot = movie['plot'] if 'plot' in movie else None
        # Store the plot in the cache
        plot_cache[imdb_id] = plot
        return plot
    except Exception as e:
        print(f"Error fetching plot for IMDb ID {imdb_id}: {e}")
        return None
print("start")
# Apply the function to get the plot for each movie, using the cache
df_filtered['plot'] = df_filtered['imdbId'].apply(lambda x: get_full_plot(str(x)))

# Save the filtered DataFrame to a CSV file
output_csv_path = os.path.join(destination_directory, "filtered_movies.csv")
df_filtered.to_csv(output_csv_path, index=False)
print(f"Filtered DataFrame saved to {output_csv_path}")

print("Plot fetching and CSV saving complete.")
