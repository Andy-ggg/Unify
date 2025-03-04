import argparse
import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

def load_coco_annotations(coco_json_path):
    """Load COCO format annotation file."""
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    return coco

def save_coco_annotations(coco_dict, output_json_path):
    """Save COCO format annotation file."""
    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)

def split_coco_annotations(
    coco_train_json,
    nclients,
    output_dir,
    val_frac=0.0,
    seed=0
):
    """
    Split a COCO format training dataset into multiple client-specific datasets.

    :param coco_train_json: Path to the original COCO training JSON file.
    :param nclients: Number of clients to distribute the dataset among.
    :param output_dir: Output directory for saving split JSON files.
    :param val_frac: Fraction of the dataset to allocate as a server validation set (default: 0).
    :param seed: Random seed for reproducibility.
    """
    random.seed(seed)

    coco = load_coco_annotations(coco_train_json)
    images = coco['images']
    annotations = coco['annotations']

    # Create a mapping from image_id to corresponding annotations
    img_id_to_anns = defaultdict(list)
    for ann in annotations:
        img_id_to_anns[ann['image_id']].append(ann)

    # Shuffle image order randomly
    random.shuffle(images)

    num_images = len(images)
    num_val = int(num_images * val_frac)
    num_train = num_images - num_val

    # Allocate validation images to the server if needed
    if val_frac > 0:
        val_images = images[:num_val]
        train_images = images[num_val:]
    else:
        val_images = []
        train_images = images

    # Calculate the number of images per client
    images_per_client = num_train // nclients
    remainder = num_train % nclients

    client_splits = {}
    start = 0
    for i in range(nclients):
        end = start + images_per_client + (1 if i < remainder else 0)
        client_splits[f'client{i+1}'] = train_images[start:end]
        start = end

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each client's dataset as a separate JSON file
    for client_name, client_images in client_splits.items():
        client_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": client_images,
            "annotations": []
        }
        # Collect annotations for the client's images
        client_img_ids = set(img['id'] for img in client_images)
        for img in client_images:
            client_coco['annotations'].extend(img_id_to_anns[img['id']])

        # Assign category list (assuming the same for all clients)
        client_coco['categories'] = coco['categories']

        # Save JSON file for the client
        output_json = os.path.join(output_dir, f'train_{client_name}.json')
        save_coco_annotations(client_coco, output_json)
        print(f"[INFO] Saved {client_name} annotations to {output_json}")

    # If validation set is needed, save server validation annotations
    if val_frac > 0:
        server_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": val_images,
            "annotations": []
        }
        server_img_ids = set(img['id'] for img in val_images)
        for img in val_images:
            server_coco['annotations'].extend(img_id_to_anns[img['id']])

        server_coco['categories'] = coco['categories']

        output_server_json = os.path.join(output_dir, 'val_server.json')
        save_coco_annotations(server_coco, output_server_json)
        print(f"[INFO] Saved server validation annotations to {output_server_json}")

def main():
    parser = argparse.ArgumentParser(description='Split COCO dataset into multiple clients.')
    parser.add_argument('--coco_train_json', type=str, required=True,
                        help='Path to the original COCO train.json file.')
    parser.add_argument('--nclients', type=int, required=True,
                        help='Number of clients to split the dataset into.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the split JSON files.')
    parser.add_argument('--val_frac', type=float, default=0.0,
                        help='Fraction of data to use for server validation (default: 0.0).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for shuffling (default: 0).')
    args = parser.parse_args()

    split_coco_annotations(
        coco_train_json=args.coco_train_json,
        nclients=args.nclients,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed
    )

if __name__ == '__main__':
    main()

