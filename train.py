import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VisionActionModel
from dataset import PickPlaceDataset


def train():
    device = "cuda"
    batch_size = 64
    num_epochs = 200
    learning_rate = 1e-3
    save_every = 1
    print_every = 100
    checkpoint_pth = "checkpoint.pth"

    model = VisionActionModel().to(device)
    model.train()
    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)

    dataset = PickPlaceDataset(npz_path="data.npz")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print_count = 0
    for epoch in range(num_epochs):
        loss_list = []
        loss_cache = 0
        for img1, img2, state, action_gt in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            state = state.to(device)
            action_gt = action_gt.to(device).float()

            optimizer.zero_grad()

            action_pd = model(img1, img2, state)
            loss = criterion(action_pd, action_gt)

            loss.backward()
            optimizer.step()

            loss_cache += loss.item()
            loss_list.append(loss.item())

            if (print_count > 0) and (print_count % print_every == 0):
                loss_cache = loss_cache / print_every
                print(f"iter: {print_count}, avg_loss: {loss_cache}")
                loss_cache = 0

            print_count += 1

        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_pth)
            print("model weights saved at", checkpoint_pth)


if __name__ == '__main__':
    train()