import os
import json
import csv

filtered_json = "/scratch/jin7/datasets/XCommnunityNote/dataset_filtered.json"

output_csv = "/scratch/jin7/datasets/XCommnunityNote/dataset_filtered.csv"

# Read the JSON file
with open(filtered_json, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open the CSV file for writing
with open(output_csv, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(["id", "text", "date", "author_name", "retweet_count", "image_urls", "tweet_url", "community_note", "language"])
    
    # Write the data rows
    for item in data:
        writer.writerow([
            str(item.get("id")),
            item.get("text"),
            item.get("date"),
            item.get("author_name"),
            item.get("retweet_count"),
            ",".join(item.get("image_urls", [])),
            item.get("tweet_url"),
            item.get("community_note", {}).get("summary"),
            item.get("language")
        ])