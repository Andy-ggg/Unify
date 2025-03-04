import json
import random
from collections import defaultdict
import argparse
import os

def split_coco_by_category(ann_file, client_num, skew_ratio=0.8, out_dir='./clients'):
    """Split the COCO dataset into non-IID partitions based on category distribution.
    Example: Assign the majority of a category's data to a single random client."""
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # To ensure reproducibility, set a random seed
    random.seed(42)

    # Group images by category {cat_id -> [list of img_ids]}
    cat2imgs = defaultdict(list)
    for ann in data['annotations']:
        cat2imgs[ann['category_id']].append(ann['image_id'])

    # Initialize client information: images store image IDs, anns store annotations
    clients = [{'images': set(), 'anns': []} for _ in range(client_num)]

    for cat, img_ids in cat2imgs.items():
        img_ids = list(set(img_ids))  # Remove duplicates
        random.shuffle(img_ids)
        # Assign the majority (skew_ratio) to a primary client
        main_client = random.randint(0, client_num - 1)
        split_point = int(len(img_ids) * skew_ratio)
        main_imgs = img_ids[:split_point]
        leftover_imgs = img_ids[split_point:]

        clients[main_client]['images'].update(main_imgs)
        # Distribute the remaining images randomly among other clients
        for img_id in leftover_imgs:
            c = random.choice([i for i in range(client_num) if i != main_client])
            clients[c]['images'].add(img_id)

    # Generate COCO JSON for each client
    os.makedirs(out_dir, exist_ok=True)
    for cid in range(client_num):
        client_ann = {
            'images': [],
            'annotations': [],
            'categories': data['categories']  # Retain category information
        }
        # Collect images for the client
        img_id_map = {}
        new_img_id = 1

        # Process original image information
        for img in data['images']:
            if img['id'] in clients[cid]['images']:
                # Remap image_id to avoid conflicts with other clients
                img_id_map[img['id']] = new_img_id
                new_img = dict(img)
                new_img['id'] = new_img_id
                client_ann['images'].append(new_img)
                new_img_id += 1

        # Collect annotations and remap image_id
        for ann in data['annotations']:
            if ann['image_id'] in clients[cid]['images']:
                new_ann = dict(ann)
                new_ann['image_id'] = img_id_map[ann['image_id']]
                client_ann['annotations'].append(new_ann)

        out_path = os.path.join(out_dir, f'train_client{cid+1}_noniid.json')
        with open(out_path, 'w') as f:
            json.dump(client_ann, f)
        print(f"Saved client {cid+1} ann_file => {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True, help='Path to COCO training JSON file')
    parser.add_argument('--client_num', type=int, default=4, help='Number of clients')
    parser.add_argument('--skew_ratio', type=float, default=0.8, help='Ratio of data assigned to the primary client per category')
    parser.add_argument('--out_dir', type=str, default='./clients', help='Output directory')
    args = parser.parse_args()

    split_coco_by_category(
        ann_file=args.ann_file,
        client_num=args.client_num,
        skew_ratio=args.skew_ratio,
        out_dir=args.out_dir
    )

if __name__ == '__main__':
    main()

