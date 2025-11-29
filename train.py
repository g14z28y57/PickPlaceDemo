from model import VisionActionModel
from dataset import PickPlaceDataset
from torch.utils.data import DataLoader


def train():
    device = "cuda"
    model = VisionActionModel().to(device)
    batch_size = 64
    dataset = PickPlaceDataset(npz_path="data.npz")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for img1, img2, state, action_gt in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        state = state.to(device)
        action_gt = action_gt.to(device)
        action_pd = model(img1, img2, state)



if __name__ == '__main__':
    train()