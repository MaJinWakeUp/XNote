import os
from google.cloud import vision
import json
from data_parser import AMMeBa
from tqdm import tqdm

def detect_web(path):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    """
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print(f"\nBest guess label: {label.label}")

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            print(f"\n\tPage url   : {page.url}")

            if page.full_matching_images:
                print(
                    "\t{} Full Matches found: ".format(len(page.full_matching_images))
                )

                for image in page.full_matching_images:
                    print(f"\t\tImage url  : {image.url}")

            if page.partial_matching_images:
                print(
                    "\t{} Partial Matches found: ".format(
                        len(page.partial_matching_images)
                    )
                )

                for image in page.partial_matching_images:
                    print(f"\t\tImage url  : {image.url}")

    if annotations.web_entities:
        print("\n{} Web entities found: ".format(len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print(f"\n\tScore      : {entity.score}")
            print(f"\tDescription: {entity.description}")

    if annotations.visually_similar_images:
        print(
            "\n{} visually similar images found:\n".format(
                len(annotations.visually_similar_images)
            )
        )

        for image in annotations.visually_similar_images:
            print(f"\tImage url    : {image.url}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    """

    results = {}

    if annotations.best_guess_labels:
        results["best_guess_labels"] = [label.label for label in annotations.best_guess_labels]

    if annotations.pages_with_matching_images:
        results["pages_with_matching_images"] = []
        for page in annotations.pages_with_matching_images:
            page_info = {
                "url": page.url,
                "full_matching_images": [image.url for image in page.full_matching_images],
                "partial_matching_images": [image.url for image in page.partial_matching_images],
            }
            results["pages_with_matching_images"].append(page_info)

    if annotations.web_entities:
        results["web_entities"] = [
            {"score": entity.score, "description": entity.description}
            for entity in annotations.web_entities
        ]

    if annotations.visually_similar_images:
        results["visually_similar_images"] = [image.url for image in annotations.visually_similar_images]

    if response.error.message:
        print(f"Error message: {response.error.message}")
        # raise Exception(
        #     "{}\nFor more info on error messages, check: "
        #     "https://cloud.google.com/apis/design/errors".format(response.error.message)
        # )

    return results


if __name__ == "__main__":
    save_dir = "/scratch/jin7/datasets/XCommnunityNote/gcloud_search"
    os.makedirs(save_dir, exist_ok=True)
    images_dir = "/scratch/jin7/datasets/XCommnunityNote/images"
    json_file = "/scratch/jin7/datasets/XCommnunityNote/dataset_filtered.json"
    with open(json_file, "r") as f:
        dataset = json.load(f)

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        data_id = data["id"]
        image_urls = data["image_urls"]
        image_paths = []
        for image_url in image_urls:
            # Check if the image URL is a local path
            if os.path.isfile(image_url):
                image_paths.append(image_url)
            else:
                # If not, download the image from the URL
                image_name = os.path.basename(image_url)
                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    # print(f"Image not found: {image_path}")
                    continue
        # If no images were found, skip this item
        if not image_paths:
            print(f"No images found for item: {data_id}")
            continue
        res = detect_web(image_paths[0])
        output_file = os.path.join(save_dir, f"{data_id}.json")
        with open(output_file, "w") as f:
            json.dump(res, f, indent=4)