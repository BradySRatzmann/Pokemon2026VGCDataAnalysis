import os
# Function to get the URLs from a text file
def read_data(filename: str):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]
