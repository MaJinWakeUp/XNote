import os
import json
from langchain_community.document_loaders import SeleniumURLLoader
from tqdm import tqdm

def get_evidence(json_file):
    # extract evidence according to the reverse search results
    if not os.path.exists(json_file):
        return ""
    with open(json_file, "r") as f:
        data = json.load(f)
    pages_list = data.get("pages_with_matching_images", [])
    urls = [page.get("url", None) for page in pages_list]
    urls = [url for url in urls if url is not None]
    loader = SeleniumURLLoader(urls=urls)
    docs = loader.load()
    evidence = ""
    for url, doc in zip(urls, docs):
        title = doc.metadata.get("title", "")
        description = doc.metadata.get("description", "")
        text = doc.page_content
        # format the page content
        text = text.replace("\n", " ")
        evidence += f"URL: {url}\nTitle: {title}\nDescription: {description}\nText: {text}\n"
    return evidence

def main():
    # Define the directory containing the evidence files
    search_dir = "/scratch/jin7/datasets/XCommnunityNote/gcloud_search"
    # Define the output directory for the extracted evidence
    output_dir = "/scratch/jin7/datasets/XCommnunityNote/evidence"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each file in the search directory
    for filename in tqdm(os.listdir(search_dir), desc="Processing files"):
        if filename.endswith(".json"):
            json_file = os.path.join(search_dir, filename)
            output_file = os.path.join(output_dir, f"{filename.split('.')[0]}.txt")
            if os.path.exists(output_file):
                continue

            evidence = get_evidence(json_file)
            with open(output_file, "w") as f:
                f.write(evidence)

if __name__ == "__main__":
    main()