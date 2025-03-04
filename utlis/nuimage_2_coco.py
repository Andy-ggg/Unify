from nuimages import NuImages
import json
import os

def convert_nuimages_to_coco(nuim_version, data_root, output_path):
    # Initialize the nuImages dataset
    nuim = NuImages(dataroot=data_root, version=nuim_version, verbose=True, lazy=False)
    
    # Define the COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create a category mapping and add categories to COCO format
    category_map = {cat["token"]: idx for idx, cat in enumerate(nuim.category)}
    for cat in nuim.category:
        coco_format["categories"].append({
            "id": category_map[cat["token"]],
            "name": cat["name"]
        })
    
    # Iterate over all samples
    ann_id = 0
    for sample in nuim.sample:
        # Get image information
        sample_data = nuim.get("sample_data", sample["key_camera_token"])
        img_id = len(coco_format["images"])
        coco_format["images"].append({
            "id": img_id,
            "file_name": sample_data["filename"],
            "width": sample_data["width"],
            "height": sample_data["height"]
        })
        
        # Get object annotations
        object_tokens, _ = nuim.list_anns(sample["token"], verbose=False)
        for obj_token in object_tokens:
            obj = nuim.get("object_ann", obj_token)
            category_id = category_map[obj["category_token"]]
            bbox = obj["bbox"]  # [left, top, right, bottom]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [bbox[0], bbox[1], width, height],  # [x, y, width, height]
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1
    
    # Save the converted data to a file
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=4)

