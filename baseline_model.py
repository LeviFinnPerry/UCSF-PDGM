import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from methods.load_data import load_and_clean_data
from PIL import Image

SEED = 1606009
PIXEL_IMG_SIZE = 32

def logistic_regression(x_train, y_train, x_test, max_iter=1000):
    logreg = LogisticRegression(max_iter=max_iter, random_state=SEED)
    logreg.fit(x_train, y_train)
    pred = logreg.predict(x_test)
    proba = logreg.predict_proba(x_test)[:, 1]
    return pred, proba

def flatten_pixels(t1, t1c, t2):
    channels = []
    for arr in (t1, t1c, t2):
        arr = np.array(arr, dtype=np.float32)
        img = Image.fromarray(arr).resize((PIXEL_IMG_SIZE, PIXEL_IMG_SIZE), Image.Resampling.BILINEAR)
        ch = np.array(img, dtype=np.float32)
        lo, hi = ch.min(), ch.max()
        ch = (ch - lo) / (hi - lo + 1e-8) # Normalise
        channels.append(ch.flatten())
    return np.concatenate(channels) # (SIZE, SIZE, 3)

def baseline_meta(x_train_meta, y_train, x_test_meta, y_test):
    meta_scaler = StandardScaler()
    x_train_meta_scaled = meta_scaler.fit_transform(x_train_meta)
    x_test_meta_scaled = meta_scaler.fit_transform(x_test_meta)
    
    meta_pred, meta_proba = logistic_regression(x_train_meta_scaled, y_train, x_test_meta_scaled)
    print("Baseline Meta-Data Logistic Regression \n")
    print(classification_report(y_test, meta_pred, target_names=["not_tumorous", "tumorous"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, meta_proba):.4f}")
    
def baseline_pixel(x_train_pixel, y_train, x_test_pixel, y_test):
    print(f"Flattening images to {PIXEL_IMG_SIZE}*{PIXEL_IMG_SIZE}*3 (T1, T1c, T2) for pixel baseline...")
    x_train_pixel_flat = np.stack([flatten_pixels(r["t1"], r["t1c"], r["t2"]) for _, r in x_train_pixel.iterrows()])
    x_test_pixel_flat = np.stack([flatten_pixels(r["t1"], r["t1c"], r["t2"]) for _, r in x_test_pixel.iterrows()])

    pixels_pred, pixels_proba = logistic_regression(x_train_pixel_flat, y_train, x_test_pixel_flat)

    print(f"Baseline Pixel Logistic Regression on {PIXEL_IMG_SIZE}*{PIXEL_IMG_SIZE}*3 (T1, T1c, T2) \n")
    print(classification_report(y_test, pixels_pred, target_names=["not_tumorous", "tumorous"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, pixels_proba):.4f}")
    return x_train_pixel_flat, x_test_pixel_flat
    
def baseline_combined(x_train_meta, x_train_pixel_flat, y_train, x_test_meta, x_test_pixel_flat, y_test):
    x_train_meta = np.asarray(x_train_meta, dtype=np.float32)
    x_test_meta = np.asarray(x_test_meta, dtype=np.float32)
    x_train = np.hstack([x_train_meta, x_train_pixel_flat])
    x_test = np.hstack([x_test_meta, x_test_pixel_flat])
    
    combined_pred, combined_proba = logistic_regression(x_train, y_train, x_test, max_iter=2000)
    print("Logistic regression on pixels + metadata combined \n")
    print(classification_report(y_test, combined_pred, target_names=["not_tumorous", "tumorous"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, combined_proba):.4f}")
    
def run_logistic_regression(mini=False):
    x_train_meta, y_train_meta, x_train_pixel, x_test_meta, y_test_meta, x_test_pixel = load_and_clean_data(mini)
    baseline_meta(x_train_meta, y_train_meta, x_test_meta, y_test_meta)
    x_train_pixel_flat, x_test_pixel_flat = baseline_pixel(x_train_pixel, y_train_meta, x_test_pixel, y_test_meta)
    baseline_combined(x_train_meta, x_train_pixel_flat, y_train_meta, x_test_meta, x_test_pixel_flat, y_test_meta)
    
