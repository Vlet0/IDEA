import torch
import numpy as np
from sklearn.decomposition import PCA
from .backbone import Net
from .latent_attack import make_U, get_deformation_matrix
from evaluate import evaluate_model
from utils.logger import setup_logger

@torch.no_grad()
def fit_pca_U(net, dataloader, K=2, device='cuda'):
    """
    Fits PCA on clean latent vectors z to extract intrinsic top-K principal directions.
    Used for Experiment A (Direction / Shaping Controls) in Report 9/9 Slide 29.
    """
    net.eval()
    all_z = []
    for x, _ in dataloader:
        x = x.to(device)
        _, z = net(x)
        all_z.append(z.cpu())
        
    Z = torch.cat(all_z, dim=0).numpy()  # (N, m)
    pca = PCA(n_components=K)
    pca.fit(Z)
    
    # Components shape: (K, m) -> transpose to (m, K)
    U_pca = torch.tensor(pca.components_.T, dtype=torch.float32).to(device)
    # Ensure unit norm per column
    U_pca = U_pca / torch.linalg.vector_norm(U_pca, dim=0, keepdim=True)
    return U_pca

def train_pure_clean_model(config, train_loader, PS, device='cuda'):
    """
    Trains a completely clean pose estimation network (no poisoning, no backdoor loss).
    Acts as the uncompromised reference victim model.
    """
    latent_dim = config.get('model', {}).get('latent_dim', 256)
    net = Net(m=latent_dim).to(device)
    
    lr = config.get('training', {}).get('learning_rate', 2e-3)
    wd = config.get('training', {}).get('weight_decay', 1e-4)
    epochs = config.get('training', {}).get('epochs', 25)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader)
    )
    
    net.train()
    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            p, _ = net(x)
            loss = torch.nn.functional.l1_loss(p, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    return net
