from torch.utils.data import Dataset

class DTUDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return {
            'imgs': None, # V x H x W x 3
            'R': None,    # V x 3 x 3
            't': None, 
            'pc': None, 
            'dep': None
        }