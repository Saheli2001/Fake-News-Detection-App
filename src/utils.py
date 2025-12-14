import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(file_path, test_size=0.2, random_state=42, output_dir=None):

    df = pd.read_csv(file_path)

    train, val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])

    os.makedirs(output_dir, exists_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)

    print(f"Training set size: {len(train)}, Validation set size: {len(val)}")
    print(f"Train and validation sets saved to {output_dir}")