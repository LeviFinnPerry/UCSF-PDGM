import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


SEED = 1606009

def load_mini_ucsf():
    print("loading dataset...")
    ds = load_dataset("chehablab/UCSF_PDGM", split="train[:18100]", keep_in_memory=True)
    df = pd.DataFrame(ds, columns=["volume_id", "slice_id", "t1", "t1c", "t2", "tumor_mask", "is_tumorous", "sex", "age"])
    df = df.dropna(subset=["age"])
    df["sex"] = (df["sex"] == "M").astype(np.float32)
    return df

def load_ucsf():
    print("loading dataset...")
    ds = load_dataset("chehablab/UCSF_PDGM", split="train")
    df = pd.DataFrame(ds, columns=["volume_id", "slice_id", "t1", "t1c", "t2", "tumor_mask", "is_tumorous", "sex", "age"])
    df = df.dropna(subset=["age"])
    df["sex"] = (df["sex"] == "M").astype(np.float32)
    return df
    
def id_split(ids, test_size=0.3, seed=SEED):
    ids = np.asarray(ids, dtype=object)
    return train_test_split(ids, test_size=test_size, random_state=seed)

def df_split(df, ids):
    return df[df["volume_id"].isin(ids)].reset_index(drop=True)
    
def split_data(df):
    patient_ids = np.asarray(df["volume_id"].unique(), dtype=object)
    train_ids, temp_ids = id_split(patient_ids, seed=SEED)
    val_ids, test_ids = id_split(temp_ids, 0.5, SEED)
    train_df = df_split(df, train_ids)
    val_df = df_split(df, val_ids)
    test_df = df_split(df, test_ids)
    return train_df, val_df, test_df

def split_y(y_df):
    return y_df["is_tumorous"].astype(int), y_df["tumor_mask"]

def split_y_df(train_df, val_df, test_df):
    y_train_meta, y_train_pixel = split_y(train_df)
    y_val_meta, y_val_pixel = split_y(val_df)
    y_test_meta, y_test_pixel = split_y(test_df)
    return y_train_meta, y_train_pixel, y_val_meta, y_val_pixel, y_test_meta, y_test_pixel

def split_x_df(train_df, val_df, test_df):
    x_train_meta, x_val_meta, x_test_meta = split_x_meta_df(train_df, val_df, test_df)
    x_train_pixel, x_val_pixel, x_test_pixel = split_x_pixel_df(train_df, val_df, test_df)
    return x_train_meta, x_train_pixel, x_val_meta, x_val_pixel, x_test_meta, x_test_pixel
        
def split_x_meta_df(train_df, val_df, test_df):
    meta_cols = ["age", "sex", "slice_id"]
    return train_df[meta_cols], val_df[meta_cols], test_df[meta_cols]

def split_x_pixel_df(train_df, val_df, test_df):
    image_cols = ["t1", "t1c", "t2"]
    return train_df[image_cols], val_df[image_cols], test_df[image_cols]
    
def print_split(train_df, val_df, test_df, y_train, y_val, y_test):
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")
    print(f"Train set class balance - not_tumorous: {(y_train == 0).mean():.3f}, "
        f"tumorous: {(y_train == 1).mean():.3f}")
    print(f"Val set class balance - not_tumorous: {(y_val == 0).mean():.3f}, "
        f"tumorous: {(y_val == 1).mean():.3f}")
    print(f"Test set class balance - not_tumorous: {(y_test == 0).mean():.3f}, "
        f"tumorous: {(y_test == 1).mean():.3f}")

def combine_all_meta_pixel(y_train_meta, y_train_pixel, y_val_meta, y_val_pixel, y_test_meta, y_test_pixel,
                           x_train_meta, x_train_pixel, x_val_meta, x_val_pixel, x_test_meta, x_test_pixel):
    y_train, y_val, y_test = combine_meta_pixel(y_train_meta, y_train_pixel), combine_meta_pixel(y_val_meta, y_val_pixel), combine_meta_pixel(y_test_meta, y_test_pixel)
    x_train, x_val, x_test = combine_meta_pixel(x_train_meta, x_train_pixel), combine_meta_pixel(x_val_meta, x_val_pixel), combine_meta_pixel(x_test_meta, x_test_pixel)
    return x_train, y_train, x_val, y_val, x_test, y_test

def combine_meta_pixel(meta, pixel):
    return np.hstack([meta, pixel])

def load_and_clean_data(mini=False, model=False):
    if mini:
        df = load_mini_ucsf()
    else:
        df = load_ucsf()
    train_df, val_df, test_df = split_data(df)
    y_train_meta, y_train_pixel, y_val_meta, y_val_pixel, y_test_meta, y_test_pixel = split_y_df(train_df, val_df, test_df)
    x_train_meta, x_train_pixel, x_val_meta, x_val_pixel, x_test_meta, x_test_pixel = split_x_df(train_df, val_df, test_df)
    print("Data loaded.")
    print_split(train_df, val_df, test_df, y_train_meta, y_val_meta, y_test_meta)
    
    if model:
        return combine_all_meta_pixel(y_train_meta, y_train_pixel, y_val_meta, y_val_pixel, y_test_meta, y_test_pixel,
                           x_train_meta, x_train_pixel, x_val_meta, x_val_pixel, x_test_meta, x_test_pixel)
    else:
        return x_train_meta, y_train_meta, x_train_pixel, x_test_meta, y_test_meta, x_test_pixel
    