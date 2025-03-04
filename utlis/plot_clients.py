import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_client_distribution(clients_dir, client_num, save_dir):
    """
    Read each client's COCO JSON file, count the number of images per category,
    and visualize the distribution using a heatmap and bar chart, then save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    distributions = {}

    for cid in range(1, client_num + 1):
        file_path = os.path.join(clients_dir, f'train_client{cid}_noniid.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Map category IDs to names
        cat_id2name = {cat['id']: cat['name'] for cat in data['categories']}
        cat_counts = {cat['name']: 0 for cat in data['categories']}
        
        # Track category occurrences per image
        image2cats = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            image2cats.setdefault(img_id, set()).add(cat_id)
        
        # Count the number of images each category appears in
        for img in data['images']:
            img_id = img['id']
            if img_id in image2cats:
                for cat_id in image2cats[img_id]:
                    cat_name = cat_id2name.get(cat_id, f"cat_{cat_id}")
                    cat_counts[cat_name] += 1

        distributions[f'client_{cid}'] = cat_counts

    # Convert to DataFrame
    df = pd.DataFrame(distributions).fillna(0)
    
    # Generate heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Number of Images per Category for each Client")
    plt.ylabel("Category")
    plt.xlabel("Client")
    plt.savefig(os.path.join(save_dir, "category_distribution_heatmap.png"))
    plt.close()
    
    # Generate bar chart
    df.T.plot(kind='bar', figsize=(12, 6))
    plt.title("Number of Images per Category for each Client")
    plt.xlabel("Client")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "category_distribution_bar.png"))
    plt.close()
