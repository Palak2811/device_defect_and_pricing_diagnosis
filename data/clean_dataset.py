import pandas as pd
from PIL import Image, ImageFile
import os
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def verify_and_clean_csv(csv_path, output_path=None):
    
    if output_path is None:
        output_path = csv_path.replace('.csv', '_cleaned.csv')
    
    df = pd.read_csv(csv_path)
    
    valid_rows = []
    issues = {
        'missing': 0,
        'corrupted': 0,
        'truncated': 0,
        'other': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        img_path = row['image_path']
        
        if not os.path.exists(img_path):
            issues['missing'] += 1
            continue
        
        try:
            with Image.open(img_path) as img:
                img.verify()
            
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img.load()  
            
            valid_rows.append(row)
            
        except OSError as e:
            if 'truncated' in str(e).lower():
                issues['truncated'] += 1
            else:
                issues['corrupted'] += 1
        except Exception as e:
            issues['other'] += 1
    
    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaned_df


if __name__ == "__main__":
    
    phone_cleaned = verify_and_clean_csv(
        r'E:\fortransferee\mlproject7\aws_cust\training_data_phone213.csv',
        r'E:\fortransferee\mlproject7\aws_cust\data\phone\phone_training_data2134.csv'
    )
    
    laptop_cleaned = verify_and_clean_csv(
        r'E:\fortransferee\mlproject7\aws_cust\training_data_laptop213.csv',
        r'E:\fortransferee\mlproject7\aws_cust\data\laptop\laptop_training_data2134.csv'
    )
    
    