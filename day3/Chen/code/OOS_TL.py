# Example of Transfer Learning for MLPs

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import random
import pandas as pd
import os

# You need to update the directory below
os.chdir("/Users/huichen/Desktop")
# print(os.getcwd())  

seed3 = 42

# Python random seeds
random.seed(seed3)

# NumPy random seeds
np.random.seed(seed3)

# PyTorch random seeds
torch.manual_seed(seed3)

torch.cuda.manual_seed_all(seed3)

# To get replicable results on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dff = pd.DataFrame()


# ============= 1. Data Generation =============

# Target domain data (from true DGP)

def generate_data_target(num=3000, e_std=1.0, seed=None):
    
    """
    True DGP: y = x1 + x2 + x1*x2 + e
    x1, x2 ~ N(0, 2), e ~ N(0, e_std^2)
    Return: X, y    
    """
    
    if seed is not None:
        np.random.seed(seed)
    x1 = np.random.normal(0, np.sqrt(1), size=num)
    x2 = np.random.normal(0, np.sqrt(1), size=num)
    e  = np.random.normal(0, e_std,      size=num)  
    
    y = x1 + x2 + x1 * x2 + e
    X = np.stack([x1, x2], axis=1)  # (num, 2)
    return X, y
    

# Source domain data (from economic model)

def generate_data_source(num=5000, e_std=1.0, m=0.5, seed=None):
    """
    Source model: y = x1 + x2 + (1 - m)*x1*x2 + e
    x1, x2 ~ N(0, 2), e ~ N(0, 0) 
    Return: X, y
    """
    if seed is not None:
        np.random.seed(seed)
    x1 = np.random.normal(0, np.sqrt(1), size=num)
    x2 = np.random.normal(0, np.sqrt(1), size=num)
    e  = np.random.normal(0, e_std,      size=num)  
    
    ## could also remove noise
    # e = np.random.normal(0, 0, size=num)

    y = x1 + x2 + (1 - m) * (x1 * x2) + e
    X = np.stack([x1, x2], axis=1)  # (num, 2)
    return X, y


# ==================== 2. Define MLP ====================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# =============== 3. Training and Testing ===============

# Training

def train_model(model, 
                X_train, y_train, 
                X_val,   y_val, 
                max_epochs=30, #100
                batch_size=20,
                patience=5,    #10
                lr=1e-3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1,1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    train_size = X_train.shape[0]
    num_batches = (train_size // batch_size) + (1 if train_size % batch_size != 0 else 0)
    
    for epoch in range(max_epochs):
        model.train()
        
        indices = np.random.permutation(train_size)
        
        # mini-batch training
        for i in range(num_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation set result
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
        
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    return model, best_epoch


# Testing

def evaluate_model(model, X_test, y_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1,1).to(device)
    
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t)
        mse = criterion(y_pred, y_test_t).item()
    return mse


# =============== 4. Experiments ===============

def run_experiment_for_train_size(X_trainval, y_trainval,
                                  X_test,     y_test,
                                  train_size=100, 
                                  batch_size=10,
                                  n_repeats=10,     #20
                                  output_dir="results_var2",
                                  extra_tag=""):
    """
    1) repeat n_repeats times;
       - randomly select train_size obs as training data; 10% as validation set;
    2) average over n_repeats tests
    """

    # output directory
    os.makedirs(output_dir, exist_ok=True)

    # m value
    M_values = np.linspace(0, 1, 5)    #np.linspace(0, 1, 11)

    # saving MSE
    dl_mse_all_runs = []
    tl_mse_all_runs = {m: [] for m in M_values}

    n_avail = X_trainval.shape[0]
    if train_size > n_avail:
        raise ValueError(f"train_size={train_size} exceeded data available={n_avail}。")

    # ---------- repeat n_repeats times ----------
    
    for repeat_i in range(n_repeats):
        
        indices = np.random.permutation(n_avail)
        train_indices = indices[:train_size]
        val_size = max(1, int(0.1 * train_size))
        val_indices = indices[train_size : train_size + val_size]

        X_train = X_trainval[train_indices]
        y_train = y_trainval[train_indices]
        X_val   = X_trainval[val_indices]
        y_val   = y_trainval[val_indices]
        
        dl_model = SimpleMLP(input_dim=2, hidden_dim=64, output_dim=1)
        dl_model, _ = train_model(dl_model, 
                                  X_train, y_train, 
                                  X_val,   y_val, 
                                  max_epochs=200,
                                  batch_size=batch_size,
                                  patience=5,
                                  lr=1e-3)
        mse_dl = evaluate_model(dl_model, X_test, y_test)
        dl_mse_all_runs.append(mse_dl)

        for m in M_values:
            # source domain training
            Xs, ys = generate_data_source(num=5000, m=m, seed=None)
            tl_model = SimpleMLP(input_dim=2, hidden_dim=64, output_dim=1)
            tl_model, _ = train_model(tl_model,
                                      Xs, ys,
                                      X_val, y_val,
                                      max_epochs=100,
                                      batch_size=batch_size,
                                      patience=10,
                                      lr=1e-3)
            # target domain fine-tuning
            tl_model, _ = train_model(tl_model,
                                      X_train, y_train,
                                      X_val,   y_val,
                                      max_epochs=200,
                                      batch_size=batch_size,
                                      patience=5,
                                      lr=1e-5)
            mse_tl = evaluate_model(tl_model, X_test, y_test)
            tl_mse_all_runs[m].append(mse_tl)

    dl_mse_mean = np.mean(dl_mse_all_runs)
    dl_mse_std  = np.std(dl_mse_all_runs)

    tl_mean_list = []
    tl_std_list  = []
    
    for m in M_values:
        arr = np.array(tl_mse_all_runs[m])
        tl_mean_list.append(np.mean(arr))
        tl_std_list.append(np.std(arr))

    # ---------- plots ----------
    plt.figure(figsize=(8,6))
    plt.plot(M_values, tl_mean_list, marker='o', label='TL mean')
    plt.axhline(y=dl_mse_mean, color='r', linestyle='--', label=f'DL mean={dl_mse_mean:.4f}')
    plt.xlabel('m')
    plt.ylabel('Test MSE (avg over repeats)')
    plt.title(f'Training size={train_size}, batch_size={batch_size}, e_std={extra_tag}')
    plt.legend()

    # Save plots
    fig_path = os.path.join(output_dir, f"train_size_{train_size}_eStd_{extra_tag}_avg_plot.png")
    plt.savefig(fig_path, dpi=100)
    #plt.show()

    print(f"[DL] train_size={train_size}, e_std={extra_tag}, n_repeats={n_repeats}, mean MSE={dl_mse_mean:.4f} (std={dl_mse_std:.4f})")

    # ---------- save results ----------
    # 1) DL 
    dl_detail_df = pd.DataFrame({"run_index": range(n_repeats), "dl_mse": dl_mse_all_runs})
    dl_detail_path = os.path.join(output_dir, f"train_size_{train_size}_eStd_{extra_tag}_DL_detail.csv")
    dl_detail_df.to_csv(dl_detail_path, index=False)

    # 2) TL 
    tl_detail_df = pd.DataFrame()
    tl_detail_df["run_index"] = range(n_repeats)
    for m in M_values:
        tl_detail_df[f"m={m:.2f}"] = tl_mse_all_runs[m]
    tl_detail_path = os.path.join(output_dir, f"train_size_{train_size}_eStd_{extra_tag}_TL_detail.csv")
    tl_detail_df.to_csv(tl_detail_path, index=False)

    # 3) averages
    avg_df = pd.DataFrame(columns=["Method","m_value","mean_mse","std_mse"])
    # DL
    avg_df.loc[len(avg_df)] = ["DL", None, dl_mse_mean, dl_mse_std]
    # TL
    for i, m in enumerate(M_values):
        avg_df.loc[len(avg_df)] = ["TL", m, tl_mean_list[i], tl_std_list[i]]
    avg_path = os.path.join(output_dir, f"train_size_{train_size}_eStd_{extra_tag}_AVG.csv")
    avg_df.to_csv(avg_path, index=False)

    dff[f"ts={train_size}_eStd={extra_tag}"] = tl_mean_list

    print("Saved results:\n", dl_detail_path, "\n", tl_detail_path, "\n", avg_path, "\n", fig_path)


# ========== 5. e_std = 1, 5; train_size = 100, 1000 ==========

def main():
    """
    1. Try e_std=1 and e_std=5
    2. train_size=40, 100 -- run_experiment_for_train_size
       - batch_size=10
    3. Save results to results_var2
    """
    for e_std in [1, 5]:
    
        X_all, y_all = generate_data_target(num=50000, e_std=e_std, seed=42)
        
        inside_std_indices = np.where(
            (np.sqrt(X_all[:, 0]**2 + X_all[:, 1]**2) <= 1) 
        )[0]  
        
        outside_std_indices = np.where(
            (np.sqrt(X_all[:, 0]**2 + X_all[:, 1]**2) > 1)  
        )[0]  

        X_trainval = X_all[inside_std_indices[:2000]]  
        y_trainval = y_all[inside_std_indices[:2000]]
        X_test = X_all[outside_std_indices]  
        y_test = y_all[outside_std_indices]

        for ts in [100, 1000]:

            print(f"\n=== Now testing e_std={e_std}, ts={ts}, X_trainval={X_trainval.shape}, X_test={X_test.shape} ===")

            run_experiment_for_train_size(
                X_trainval, y_trainval,
                X_test,     y_test,
                train_size=ts,
                batch_size=10,  
                n_repeats=10,   
                output_dir="oos", 
                extra_tag=str(e_std)
            )

    dff.to_csv("results.csv", index=False)
    print("\n Experiment completed and recorded in 'results.csv'。")


if __name__ == "__main__":
    main()