# Code for learning by bnn
# 2026.07

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

def prepare_data(df):
    # features
    X = df[['client_address_latitude', 'client_address_longitude']].values
    
    # direction
    le = LabelEncoder()
    y_dir = le.fit_transform(df['direction_label'].values)
    n_classes = len(le.classes_)
    
    # distance
    y_dist = df['dist'].values.reshape(-1, 1)
    
    # scaler
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Scale dist 
    y_dist_log = np.log1p(y_dist) 
    scaler_y = StandardScaler()
    y_dist_scaled = scaler_y.fit_transform(y_dist_log)
    
    return X_scaled, y_dir, y_dist_scaled, le, scaler_X, scaler_y, n_classes

class MCDropoutBNN(nn.Module):
    # you can change parameters when initializing
    def __init__(self, input_dim=2, hidden_dim=256, n_classes=8, dropout_rate=0.2):
        super(MCDropoutBNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        
        #  backbone 
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), # the #of output should be half
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        
        # Head 1: direction
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, n_classes)
        )
        
        # Head 2: distance
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, n_classes * 2)   # mean + log_var for each direction
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # direction
        dir_logits = self.dir_head(features)              
        
        # distance
        dist_output = self.dist_head(features)             
        dist_mean    = dist_output[:, :self.n_classes]     
        dist_log_var = dist_output[:, self.n_classes:]     
        
        return dir_logits, dist_mean, dist_log_var
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

def heteroscedastic_loss(y_pred_mean, y_pred_log_var, y_true):
    # calculate GNLL loss
    precision = torch.exp(-y_pred_log_var)
    loss = 0.5 * precision * (y_true - y_pred_mean)**2 + 0.5 * y_pred_log_var
    return loss.mean()

def multitask_loss(dir_logits, dist_mean, dist_log_var,y_dir, y_dist,n_classes=8,lambda_dir=1.0, lambda_dist=1.0):
    # cls loss
    cls_loss = F.cross_entropy(dir_logits, y_dir)
    sigma_angle = 45.0   
    angle_step  = 360.0 / n_classes   # n_classes = 8 at default

    # angular difference between true direction y_dir[i] and every class j
    true_idx = y_dir.unsqueeze(1).float()                  
    all_idx  = torch.arange(n_classes, device=y_dir.device).float()  

    # calculate shortest arc distance in bins and  converted to degrees
    diff_bins = (all_idx - true_idx).abs()              
    diff_bins = torch.min(diff_bins, n_classes - diff_bins)  
    diff_deg  = diff_bins * angle_step                  

    weights = torch.exp(-0.5 * (diff_deg / sigma_angle) ** 2)  

    y_dist_expanded = y_dist.expand(-1, n_classes)         

    precision = torch.exp(-dist_log_var)                    
    nll = 0.5 * precision * (y_dist_expanded - dist_mean) ** 2 + 0.5 * dist_log_var                        

    # Weight each direction's loss
    weighted_nll = (weights * nll).sum(dim=1) / weights.sum(dim=1)  
    reg_loss     = weighted_nll.mean()

    total_loss = lambda_dir * cls_loss + lambda_dist * reg_loss
    return total_loss, cls_loss, reg_loss

def mc_dropout_predict(model, X_tensor, n_samples=200, device='cpu'):
    model.eval()
    model.enable_dropout()
    
    dir_probs_samples  = []
    dist_mean_samples  = []   
    dist_var_samples   = []  
    
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        
        for _ in range(n_samples):
            dir_logits, dist_mean, dist_log_var = model(X_tensor)
            
            dir_probs = F.softmax(dir_logits, dim=-1)          
            dir_probs_samples.append(dir_probs.cpu().numpy())
            dist_mean_samples.append(dist_mean.cpu().numpy())  
            dist_var_samples.append(
                torch.exp(dist_log_var).cpu().numpy()          
            )
    
    dir_probs_samples = np.array(dir_probs_samples)   
    dist_mean_samples = np.array(dist_mean_samples)   
    dist_var_samples  = np.array(dist_var_samples)   
    
    # make a result dictionary
    results = {
        # Direction
        'dir_prob_mean':     dir_probs_samples.mean(axis=0),   
        'dir_prob_std':      dir_probs_samples.std(axis=0),   
        'dir_pred_class':    dir_probs_samples.mean(axis=0).argmax(axis=-1),  
        
        # Per-direction distance 
        'dist_mean':         dist_mean_samples.mean(axis=0),   
        'dist_aleatoric_var':dist_var_samples.mean(axis=0),    
        'dist_epistemic_var':dist_mean_samples.var(axis=0),    
        'dist_total_std':    np.sqrt(
            dist_var_samples.mean(axis=0) +dist_mean_samples.var(axis=0)
                             ),                                 
        # Full samples
        'dir_probs_all_samples': dir_probs_samples,
        'dist_mean_all_samples': dist_mean_samples,
    }
    
    return results


def format_outputs(results, label_encoder, scaler_y, i=0):
    """    
    Output dictionary sample:
    {
      'S':  {'prob': 0.45, 'dist_km': 12.3, 'dist_std_km': 2.1},
      'SE': {'prob': 0.30, 'dist_km':  8.7, 'dist_std_km': 1.5},
      ...
    }
    """
    classes   = label_encoder.classes_
    probs     = results['dir_prob_mean'][i]         
    dist_mean = results['dist_mean'][i]             
    dist_std  = results['dist_total_std'][i]         
    
    output = {}
    for j, cls in enumerate(classes):
        dm = float(np.expm1(
            scaler_y.inverse_transform([[dist_mean[j]]])[0][0]
        ))
        ds = float(np.expm1(
            scaler_y.inverse_transform([[dist_std[j]]])[0][0]
        ))
        output[cls] = {
            'prob':        float(probs[j]),
            'dist_km':     dm,
            'dist_std_km': ds,
            'dist_95ci':   (max(0.0, dm - 1.96 * ds), dm + 1.96 * ds),
        }
    
    return output


def train_model(model, train_loader, test_loader,
                n_epochs=200, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    history = {'train_loss': [], 'test_loss': [], 'dir_acc': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        # train
        model.train()
        train_losses = []
        for X_batch, y_dir_batch, y_dist_batch in train_loader:
            X_batch      = X_batch.to(device)
            y_dir_batch  = y_dir_batch.to(device)
            y_dist_batch = y_dist_batch.to(device)

            optimizer.zero_grad()
            dir_logits, dist_mean, dist_log_var = model(X_batch)
            loss, cls_loss, reg_loss = multitask_loss(
                dir_logits, dist_mean, dist_log_var,
                y_dir_batch, y_dist_batch
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_dir_batch, y_dist_batch in test_loader:
                X_batch      = X_batch.to(device)
                y_dir_batch  = y_dir_batch.to(device)
                y_dist_batch = y_dist_batch.to(device)

                dir_logits, dist_mean, dist_log_var = model(X_batch)
                loss, _, _ = multitask_loss(
                    dir_logits, dist_mean, dist_log_var,
                    y_dir_batch, y_dist_batch
                )
                val_losses.append(loss.item())
                correct += (dir_logits.argmax(dim=1) == y_dir_batch).sum().item()
                total   += len(y_dir_batch)

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        dir_acc    = correct / total

        history['train_loss'].append(train_loss)
        history['test_loss'].append(val_loss)
        history['dir_acc'].append(dir_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Dir Acc: {dir_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"\n Training complete. Best val loss: {best_val_loss:.4f}")
    return model, history

def main(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, y_dir, y_dist, label_encoder, scaler_X, scaler_y, n_classes = prepare_data(df)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"N Classes: {n_classes}")
    print(f"X shape: {X.shape}")
    
    X_train, X_test, y_dir_train, y_dir_test, y_dist_train, y_dist_test = \
        train_test_split(X, y_dir, y_dist, test_size=0.2, random_state=42)
    
    def to_tensors(X, y_dir, y_dist):
        return (torch.FloatTensor(X),
                torch.LongTensor(y_dir),
                torch.FloatTensor(y_dist))
    
    train_data = TensorDataset(*to_tensors(X_train, y_dir_train, y_dist_train))
    test_data  = TensorDataset(*to_tensors(X_test,  y_dir_test,  y_dist_test))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
    
    model = MCDropoutBNN(
        input_dim=2,
        hidden_dim=256,
        n_classes=n_classes,
        dropout_rate=0.2
    ).to(device)
        
    model, history = train_model(
        model, train_loader, test_loader,
        n_epochs=500, lr=1e-3, device=device
    )
    
    # inference
    X_test_tensor = torch.FloatTensor(X_test)
    results = mc_dropout_predict(model, X_test_tensor, n_samples=200, device=device)
    
    # evaluate
    acc = accuracy_score(y_dir_test, results['dir_pred_class'])
    print(f"\n=== Evaluation ===")
    print(f"Direction Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_dir_test, results['dir_pred_class'],
                                 target_names=label_encoder.classes_))
    
    
    
    return model, results, label_encoder, scaler_X, scaler_y

import os
import json


def save_model(model, label_encoder, scaler_X, scaler_y, save_dir='./saved_model'):
    #Save model weights and preprocessors
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f'{save_dir}/model_weights.pth')
    
    # will be able to be configured through arguments
    config = {
        'input_dim':    2,
        'hidden_dim':   256,
        'n_classes':    len(label_encoder.classes_),
        'dropout_rate': 0.2,
        'classes':      label_encoder.classes_.tolist(),
    }
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    import joblib
    joblib.dump(scaler_X,       f'{save_dir}/scaler_X.pkl')
    joblib.dump(scaler_y,       f'{save_dir}/scaler_y.pkl')
    joblib.dump(label_encoder,  f'{save_dir}/label_encoder.pkl')
    
    print(f"Model saved to '{save_dir}/'")


def load_model(save_dir='./saved_model', device='cpu'):
    import joblib
    
    with open(f'{save_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    model = MCDropoutBNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_classes=config['n_classes'],
        dropout_rate=config['dropout_rate'],
    ).to(device)
    
    model.load_state_dict(
        torch.load(f'{save_dir}/model_weights.pth', map_location=device)
    )
    model.eval()
    
    scaler_X      = joblib.load(f'{save_dir}/scaler_X.pkl')
    scaler_y      = joblib.load(f'{save_dir}/scaler_y.pkl')
    label_encoder = joblib.load(f'{save_dir}/label_encoder.pkl')
    
    print(f"Model loaded from '{save_dir}/'")
    print(f"  Classes: {label_encoder.classes_}")
    
    return model, label_encoder, scaler_X, scaler_y


def predict(lat, lon,save_dir='./saved_model',n_samples=200,device='cpu'):

    # load
    model, label_encoder, scaler_X, scaler_y = load_model(save_dir, device)
    
    lats = [lat] if isinstance(lat, (int, float)) else lat
    lons = [lon] if isinstance(lon, (int, float)) else lon
    
    points = np.array(list(zip(lats, lons)))          
    points_scaled = scaler_X.transform(points)
    X_tensor = torch.FloatTensor(points_scaled)
    
    results = mc_dropout_predict(model, X_tensor, n_samples=n_samples, device=device)
    
    classes = label_encoder.classes_
    outputs = []
    
    for i in range(len(lats)):
        probs     = results['dir_prob_mean'][i]    # [n_classes]
        dir_std   = results['dir_prob_std'][i]     # [n_classes]
        dist_mean = results['dist_mean'][i]        # [n_classes] ← per-direction
        dist_std  = results['dist_total_std'][i]   # [n_classes] ← per-direction
        
        directions = {}
        for j, cls in enumerate(classes):
            dm = float(np.expm1(
                scaler_y.inverse_transform([[dist_mean[j]]])[0][0]
            ))
            ds = float(np.expm1(
                scaler_y.inverse_transform([[dist_std[j]]])[0][0]
            ))
            directions[cls] = {
                'prob':        float(probs[j]),
                'prob_std':    float(dir_std[j]),
                'dist_km':     dm,
                'dist_std_km': ds,
                'dist_95ci':   (max(0.0, dm - 1.96 * ds), dm + 1.96 * ds),
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
    # read a lerning file prepared in advance
    df = pd.read_csv('./direction_dist_table.csv')
    model, results, label_encoder, scaler_X, scaler_y = main(df)
    
    save_model(model, label_encoder, scaler_X, scaler_y, save_dir='./saved_model')
    
    # single prediction
    preds = predict(lat=35.6762, lon=139.6503)
    
    for p in preds:
        print(f"\n({p['latitude']}, {p['longitude']})")
        print(f"  Top direction: {p['top_direction']} "
              f"(confidence: {p['top_confidence']:.2%})")
        print(f"\n  {'Dir':4s} | {'Prob':6s} | {'Dist(km)':10s} | {'Std':8s} | 95% CI")
        print(f"  {'-'*55}")
        for cls, vals in sorted(p['directions'].items(), 
                                key=lambda x: -x[1]['prob']):
            print(f"  {cls:4s} | {vals['prob']:.3f}  | "
                  f"{vals['dist_km']:8.2f}   | "
                  f"{vals['dist_std_km']:6.2f}   | "
                  f"({vals['dist_95ci'][0]:.1f}, {vals['dist_95ci'][1]:.1f})")
    
    # multiple predictions
    preds_multi = predict(
        lat=[35.6762, 34.6937, 43.0642],
        lon=[139.6503, 135.5023, 141.3469]
    )
    for p in preds_multi:
        print(f"\n({p['latitude']}, {p['longitude']}) → "
              f"Top: {p['top_direction']} ({p['top_confidence']:.2%})")
        for cls, vals in sorted(p['directions'].items(),
                                key=lambda x: -x[1]['prob']):
            print(f"  {cls:4s}: prob={vals['prob']:.3f} | "
                  f"dist={vals['dist_km']:.2f} ± {vals['dist_std_km']:.2f} km")
