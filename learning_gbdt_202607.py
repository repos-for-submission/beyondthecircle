# Ablation Study: learning by gbdt
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    X = df[['client_address_latitude', 'client_address_longitude']].values

    le = LabelEncoder()
    y_dir = le.fit_transform(df['direction_label'].values)
    n_classes = len(le.classes_)

    y_dist_log = np.log1p(df['dist'].values)

    return X, y_dir, y_dist_log, le, n_classes


def build_distance_features(X, y_dir_onehot):
    return np.hstack([X, y_dir_onehot])


def one_hot(y_dir, n_classes):
    eye = np.eye(n_classes)
    return eye[y_dir]

class GBDTDirectionDistanceModel:
    def __init__(self, n_classes=8, random_state=42,
                 n_estimators=300, max_depth=5, learning_rate=0.05):
        self.n_classes = n_classes
        self.dir_clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=n_classes,
            random_state=random_state,
            eval_metric='mlogloss',
        )
        self.dist_reg = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=random_state,
        )

    def fit(self, X_train, y_dir_train, y_dist_train):
        self.dir_clf.fit(X_train, y_dir_train)

        dir_onehot_train = one_hot(y_dir_train, self.n_classes)
        dist_features_train = build_distance_features(X_train, dir_onehot_train)
        self.dist_reg.fit(dist_features_train, y_dist_train)
        return self

    def predict_direction_proba(self, X):
        return self.dir_clf.predict_proba(X)

    def predict_distance_per_direction(self, X):
        n = X.shape[0]
        dist_mean = np.zeros((n, self.n_classes))
        for j in range(self.n_classes):
            dir_onehot = np.zeros((n, self.n_classes))
            dir_onehot[:, j] = 1.0
            dist_features = build_distance_features(X, dir_onehot)
            dist_mean[:, j] = self.dist_reg.predict(dist_features)
        return dist_mean


def gbdt_predict(model, X, label_encoder):
    dir_prob = model.predict_direction_proba(X)               
    dist_mean_log = model.predict_distance_per_direction(X)   

    results = {
        'dir_prob_mean':  dir_prob,
        'dir_prob_std':   np.zeros_like(dir_prob),
        'dir_pred_class': dir_prob.argmax(axis=-1),
        'dist_mean':      dist_mean_log,
        'dist_total_std': np.zeros_like(dist_mean_log),
    }
    return results


def format_outputs(results, label_encoder, i=0):
    classes = label_encoder.classes_
    probs = results['dir_prob_mean'][i]
    dist_mean_log = results['dist_mean'][i]

    output = {}
    for j, cls in enumerate(classes):
        dm = float(np.expm1(dist_mean_log[j]))
        output[cls] = {
            'prob':        float(probs[j]),
            'dist_km':     dm,
            'dist_std_km': 0.0,
            'dist_95ci':   (dm, dm),
        }
    return output


def train_and_evaluate(df, test_size=0.2, random_state=42):
    X, y_dir, y_dist_log, label_encoder, n_classes = prepare_data(df)

    print(f"Classes: {label_encoder.classes_}")
    print(f"N Classes: {n_classes}")
    print(f"X shape: {X.shape}")

    try:
        X_train, X_test, y_dir_train, y_dir_test, y_dist_train, y_dist_test = \
            train_test_split(X, y_dir, y_dist_log, test_size=test_size,random_state=random_state, stratify=y_dir)
    except ValueError:
        X_train, X_test, y_dir_train, y_dir_test, y_dist_train, y_dist_test = \
            train_test_split(X, y_dir, y_dist_log, test_size=test_size,random_state=random_state)

    model = GBDTDirectionDistanceModel(n_classes=n_classes, random_state=random_state)
    model.fit(X_train, y_dir_train, y_dist_train)

    results = gbdt_predict(model, X_test, label_encoder)

    acc = accuracy_score(y_dir_test, results['dir_pred_class'])
    print(f"\n=== Evaluation ===")
    print(f"Direction Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_dir_test, results['dir_pred_class'],
                                 target_names=label_encoder.classes_))

    dist_pred_log_true_dir = results['dist_mean'][np.arange(len(y_dir_test)), y_dir_test]
    dist_pred_km = np.expm1(dist_pred_log_true_dir)
    dist_true_km = np.expm1(y_dist_test)
    mae = np.mean(np.abs(dist_pred_km - dist_true_km))
    print(f"Distance MAE (true-direction column): {mae:.4f} km")

    return model, results, label_encoder, X_test, y_dir_test, y_dist_test

def save_model(model, label_encoder, save_dir='./saved_model_gbdt'):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model.dir_clf,  f'{save_dir}/dir_clf.joblib')
    joblib.dump(model.dist_reg, f'{save_dir}/dist_reg.joblib')
    joblib.dump(label_encoder,  f'{save_dir}/label_encoder.joblib')

    config = {'n_classes': model.n_classes}
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to '{save_dir}/'")


def load_model(save_dir='./saved_model_gbdt'):
    with open(f'{save_dir}/config.json', 'r') as f:
        config = json.load(f)

    model = GBDTDirectionDistanceModel(n_classes=config['n_classes'])
    model.dir_clf  = joblib.load(f'{save_dir}/dir_clf.joblib')
    model.dist_reg = joblib.load(f'{save_dir}/dist_reg.joblib')
    label_encoder  = joblib.load(f'{save_dir}/label_encoder.joblib')

    print(f"Model loaded from '{save_dir}/'")
    return model, label_encoder


def predict(lat, lon, save_dir='./saved_model_gbdt'):
    model, label_encoder = load_model(save_dir)

    lats = [lat] if isinstance(lat, (int, float)) else lat
    lons = [lon] if isinstance(lon, (int, float)) else lon
    X = np.array(list(zip(lats, lons)))

    results = gbdt_predict(model, X, label_encoder)

    classes = label_encoder.classes_
    outputs = []
    for i in range(len(lats)):
        probs = results['dir_prob_mean'][i]
        dist_mean_log = results['dist_mean'][i]

        directions = {}
        for j, cls in enumerate(classes):
            dm = float(np.expm1(dist_mean_log[j]))
            directions[cls] = {
                'prob':        float(probs[j]),
                'prob_std':    0.0,
                'dist_km':     dm,
                'dist_std_km': 0.0,
                'dist_95ci':   (dm, dm),
            }

        top_idx = probs.argmax()
        outputs.append({
            'latitude':       lats[i],
            'longitude':      lons[i],
            'directions':     directions,
            'top_direction':  classes[top_idx],
            'top_confidence': float(probs[top_idx]),
        })
    return outputs

if __name__ == "__main__":
    df = pd.read_csv('./direction_dist_table.csv')

    model, results, label_encoder, X_test, y_dir_test, y_dist_test = train_and_evaluate(df)

    save_model(model, label_encoder, save_dir='./saved_model_gbdt')

    preds = predict(lat=35.6762, lon=139.6503, save_dir='./saved_model_gbdt')
    for p in preds:
        print(f"\n({p['latitude']}, {p['longitude']})")
        print(f"  Top direction: {p['top_direction']} "
              f"(confidence: {p['top_confidence']:.2%})")
        print(f"\n  {'Dir':4s} | {'Prob':6s} | {'Dist(km)':10s}")
        print(f"  {'-'*35}")
        for cls, vals in sorted(p['directions'].items(),
                                 key=lambda x: -x[1]['prob']):
            print(f"  {cls:4s} | {vals['prob']:.3f}  | {vals['dist_km']:8.2f}")
