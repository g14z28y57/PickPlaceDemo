import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisionActionModel
from dataset import PickPlaceDataset
from tqdm import tqdm


def train():
    device = "cuda"
    batch_size = 64
    num_epochs = 200
    learning_rate = 1e-3
    save_every = 1
    checkpoint_pth = "checkpoint.pth"
    stat_path = "stat.npz"

    model = VisionActionModel().to(device)
    model.train()
    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)

    dataset = PickPlaceDataset(json_path="train_data.json", stat_path=stat_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_cache = 0
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            ncols=100)

        for i, (img1, img2, state, action_gt, catch_state_gt, task_state_gt) in progress_bar:
            img1 = img1.to(device)
            img2 = img2.to(device)
            state = state.to(device)
            action_gt = action_gt.to(device)
            catch_state_gt = catch_state_gt.to(device)
            task_state_gt = task_state_gt.to(device)

            optimizer.zero_grad()
            action_pd, catch_state_pd, task_state_pd = model(img1, img2, state)
            loss_action = nn.L1Loss()(action_pd, action_gt)
            loss_catch = nn.BCEWithLogitsLoss()(catch_state_pd, catch_state_gt)
            loss_task = nn.BCEWithLogitsLoss()(task_state_pd, task_state_gt)
            loss = loss_action + loss_catch + loss_task
            loss.backward()
            optimizer.step()
            loss_cache += loss.item()

            # 实时更新进度条上的 loss
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                'avg_loss': f"{loss_cache / (i + 1):.4f}"
            })

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_pth)


if __name__ == '__main__':
    train()