import json
import pandas as pd
import os
import re
from pathlib import Path

def parse_jsonl(file_path, image_root):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # Extract parts
                user_parts = entry['contents'][0]['parts']
                model_parts = entry['contents'][1]['parts']
                
                # Extract image filename
                gs_uri = user_parts[0]['fileData']['fileUri']
                filename = os.path.basename(gs_uri)
                local_image_path = os.path.join(image_root, filename)
                
                # Check if image exists
                if not os.path.exists(local_image_path):
                    continue
                    
                # Extract text
                full_text = user_parts[1]['text']
                # Regex to extract title
                match = re.search(r'Title:"(.*?)"\n', full_text, re.DOTALL)
                if match:
                    title = match.group(1)
                else:
                    # Fallback if regex fails, just take first line
                    title = full_text.split('\n')[0].replace('Title:', '').strip('"')
                
                # Extract label
                label_text = model_parts[0]['text'].strip()
                if label_text == "Yes":
                    label = 1 # Fake
                elif label_text == "No":
                    label = 0 # Real
                else:
                    continue # Skip unclear labels
                
                data.append({
                    'image_path': local_image_path,
                    'caption': title,
                    'label': label
                })
                
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue
                
    return pd.DataFrame(data)

def balance_dataset(df):
    # Balance classes by downsampling majority
    real = df[df['label'] == 0]
    fake = df[df['label'] == 1]
    
    n = min(len(real), len(fake))
    
    real_downsampled = real.sample(n=n, random_state=42)
    fake_downsampled = fake.sample(n=n, random_state=42)
    
    return pd.concat([real_downsampled, fake_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    base_dir = r"c:\Users\indra\Fake News Detection system\data"
    fakeddit_dir = os.path.join(base_dir, "fakeddit_subset")
    
    # Process Training
    train_jsonl = os.path.join(fakeddit_dir, "training_data_fakeddit.jsonl")
    train_image_root = os.path.join(fakeddit_dir, "image_folder")
    
    print("Processing Training Data...")
    df_train = parse_jsonl(train_jsonl, train_image_root)
    print(f"Original Train Data: {len(df_train)}")
    print(df_train['label'].value_counts())
    
    df_train_balanced = balance_dataset(df_train)
    print(f"Balanced Train Data: {len(df_train_balanced)}")
    
    # Process Validation
    # Note: Validation images might be in 'validation_image' or 'image_folder'?
    # Listing showed "validation_image" directory.
    val_jsonl = os.path.join(fakeddit_dir, "validation_data_fakeddit.jsonl")
    val_image_root = os.path.join(fakeddit_dir, "validation_image")
    
    print("Processing Validation Data...")
    # NOTE: Sometimes validation images are in the same folder or different. 
    # Based on directory listing, it is 'validation_image'.
    df_val = parse_jsonl(val_jsonl, val_image_root)
    print(f"Original Val Data: {len(df_val)}")
    
    # Save to CSV
    df_train_balanced.to_csv(os.path.join(base_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
    
    print("Saved train.csv and val.csv to data/")

if __name__ == "__main__":
    main()
