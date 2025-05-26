import os
import shutil
import yaml

# Define paths
combined_base = 'combined_dataset'
source_datasets = [
    {
        'path': 'sitting_and_standing',
        'label_map': {'sitiing': 0, 'standing': 1},  # Match your trained model's classes
        'name': 'Sitting/Standing Dataset'
    },
    {
        'path': 'phone_detection',
        'label_map': {'phone': 2},  # Add phone as class 2
        'name': 'Phone Detection Dataset'
    }
]

def verify_dataset_structure(dataset_path, dataset_name):
    """Verify dataset structure and classes"""
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found in {dataset_path}")
        return False
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"\nVerifying {dataset_name}:")
    print(f"Classes: {data['names']}")
    print(f"Number of classes: {data['nc']}")
    
    # Verify directories exist
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, 'images')
        lbl_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(img_dir):
            print(f"Error: Images directory not found: {img_dir}")
            return False
        if not os.path.exists(lbl_dir):
            print(f"Error: Labels directory not found: {lbl_dir}")
            return False
        print(f"Found {split} split with images and labels")
    
    return True

def count_dataset_statistics(dataset_path):
    """Count images and labels in each split"""
    stats = {'train': 0, 'valid': 0, 'test': 0}
    for split in stats.keys():
        img_dir = os.path.join(dataset_path, split, 'images')
        if os.path.exists(img_dir):
            stats[split] = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    return stats

# Verify all datasets before combining
print("Verifying datasets before combining...")
for ds in source_datasets:
    if not verify_dataset_structure(ds['path'], ds['name']):
        print(f"Error: Dataset verification failed for {ds['name']}")
        exit(1)
    
    # Print dataset statistics
    stats = count_dataset_statistics(ds['path'])
    print(f"\nDataset statistics for {ds['name']}:")
    for split, count in stats.items():
        print(f"{split}: {count} images")

# Setup combined dataset directories
print("\nSetting up combined dataset structure...")
for split in ['train', 'valid', 'test']:
    os.makedirs(f'{combined_base}/images/{split}', exist_ok=True)
    os.makedirs(f'{combined_base}/labels/{split}', exist_ok=True)

# Process all datasets
print("\nCombining datasets...")
total_stats = {'train': 0, 'valid': 0, 'test': 0}

for ds in source_datasets:
    base = ds['path']
    label_map = ds['label_map']
    print(f"\nProcessing {ds['name']}...")
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base, split, 'images')
        lbl_dir = os.path.join(base, split, 'labels')
        
        image_count = 0
        label_count = 0
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.png')):
                continue
            
            # Copy image
            new_img_path = os.path.join(combined_base, 'images', split, img_file)
            shutil.copy(os.path.join(img_dir, img_file), new_img_path)
            image_count += 1
            
            # Copy and remap label
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            old_label_path = os.path.join(lbl_dir, label_file)
            new_label_path = os.path.join(combined_base, 'labels', split, label_file)

            if os.path.exists(old_label_path):
                with open(old_label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                    for line in f_in:
                        cls, *coords = line.strip().split()
                        cls_name = list(label_map.keys())[int(cls)]
                        new_cls = label_map[cls_name]
                        f_out.write(f"{new_cls} {' '.join(coords)}\n")
                label_count += 1
        
        print(f"Processed {split} split: {image_count} images, {label_count} labels")
        total_stats[split] += image_count

# Create data.yaml for the combined dataset
print("\nCreating combined dataset data.yaml...")
with open(os.path.join(combined_base, 'data.yaml'), 'w') as f:
    f.write(f"""path: {combined_base}
train: images/train
val: images/valid
test: images/test

nc: 3
names: ['sitiing', 'standing', 'phone']  # Match your trained model's classes and add phone
""")

print("\nDataset combination complete!")
print(f"Combined dataset saved to: {combined_base}")
print("\nFinal dataset statistics:")
for split, count in total_stats.items():
    print(f"{split}: {count} images")
print("\nClasses in combined dataset:")
print("0: sitiing (from sitting/standing dataset)")
print("1: standing (from sitting/standing dataset)")
print("2: phone (from phone detection dataset)")

print("\nYou can now train your model using:")
print(f"yolo train model=yolov11s.pt data={combined_base}/data.yaml epochs=100 imgsz=640")
