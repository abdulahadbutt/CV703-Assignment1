import torch 
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os 

# from architecture import base_model, preprocessor
from architecture import ConvNextV2
from dataset import get_combined_dataset
import wandb 
from tqdm import tqdm 
import numpy as np 
import yaml 
import argparse

def set_determinism():
    import random
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu'  
):
    # print(model)

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(data_loader, unit='batch') as data:
        batch_loss_list = []
        for (inputs, labels) in data:
            data.set_description(f"Epoch {epoch_index}")

            labels = labels.to(device)
            inputs = model.preprocessor(images=inputs, return_tensors="pt")['pixel_values'].to(device)

            optimizer.zero_grad()
            
            # ? Feeding to CNN
            with torch.autocast("cuda"):
                outputs = model(inputs)['logits']
                # ? Getting Loss
                batch_loss = criterion(outputs, labels)
            
            batch_loss_list.append(batch_loss.item())

            # Backpropagation with scaling
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)  # Update optimizer
            # ? Gradient Descent
            scaler.update()  # Update the scale for next step

            data.set_postfix(
                batch_loss=batch_loss.item()
            )

    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list)
    }



def test_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu' 
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad(), tqdm(data_loader, unit='batch') as data:
        batch_loss_list = []
        for (inputs, labels) in data:
            data.set_description(f"Testing after epoch{epoch_index}")

            labels = labels.to(device)
            inputs = model.preprocessor(images=inputs, return_tensors="pt")['pixel_values'].to(device)

            # ? Feeding to CNN
            outputs = model(inputs)
            outputs = outputs['logits']
            
            # ? Getting Loss
            batch_loss = criterion(outputs, labels)
            batch_loss_list.append(batch_loss.item())

            data.set_postfix(
                batch_loss=batch_loss.item()
            )

            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)


    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list),
        'accuracy': (num_correct / num_samples).item()
    }




def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:torch.device,
    test_dataloader: torch.utils.data.DataLoader,
    scheduler,
    save_path: str
):

    
    best_acc = 0
    train_statistics_list = []
    for epoch in range(epochs):
        # * Training Code
        train_epoch_statistics = train_one_epoch(
            model, optimizer, train_dataloader, epoch, criterion, device
        )
        train_statistics_list.append(train_epoch_statistics)
        # live.log_metric('train/loss', train_epoch_statistics['epoch_loss'], plot=True)
        
        # * Testing Code
        test_epoch_statistics = test_one_epoch(
            model, test_dataloader, epoch, criterion, device
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_epoch_statistics["epoch_loss"])

        latest_test_acc = test_epoch_statistics['accuracy']
        if latest_test_acc > best_acc:
            print(f'UPDATING BEST ACC [{best_acc}] -> [{latest_test_acc}]')
            best_acc = latest_test_acc
            save_checkpoint(model, epoch, optimizer, best_acc, os.path.join(save_path, "best.pth"))
        
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_statistics['epoch_loss'],
            "test/loss": test_epoch_statistics['epoch_loss'],
            "test/accuracy": test_epoch_statistics['accuracy']
        })
        if(scheduler):
            wandb.log({"lr": optimizer.param_groups[0]['lr']})

        print(latest_test_acc, type(latest_test_acc))
        # live.log_metric('test/loss', test_epoch_statistics['epoch_loss'], plot=True)
        # live.log_metric('test/accuracy', test_epoch_statistics['accuracy'], plot=True)
        # live.next_step()




    return train_statistics_list



def save_checkpoint(
        model: torch.nn.Module, 
        epoch: int, 
        optimizer: torch.optim, 
        f1_score: int, 
        path: str):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "f1_score": f1_score,
        },
        path,
    )


def main(args):
    # # 
    # DATASET = 'flowers102'
    # LEARNING_RATE = 0.2
    # ARCHITECTURE = "ConvNextV2"
    # EPOCHS = 30

    params = yaml.safe_load(open('params.yaml'))
    # IMG_SIZE = params['IMG_SIZE']
    # IMG_SIZE = int(IMG_SIZE)
    ROOT_DIR = params['ROOT_DIR']
    BATCH_SIZE = params['BATCH_SIZE']
    LEARNING_RATE = params['LEARNING_RATE']
    EPOCHS = params['EPOCHS']
    OPTIMIZER = params['OPTIMIZER']
    DATA_AUG = params['DATA_AUG']
    ARCHITECTURE = params['ARCHITECTURE']
    DATASET = params['DATASET']

    set_determinism()

    if not DATA_AUG:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop to 224x224 with scale variation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random rotation and translation
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            transforms.PILToTensor()  # Convert image to tensor
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 (standard for pretrained models)
            transforms.PILToTensor()  # Convert image to tensor
        ])

    else:
        print('DATA AUG NOT IMPLEMENTED')
        exit(0)


    if DATASET == 'flowers102':
        NUM_CLASSES = 102
        flowers102_root = './data/flowers-102'
        train_dataset = torchvision.datasets.Flowers102(root=flowers102_root, split='test', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.Flowers102(root=flowers102_root, split='train', download=True, transform=test_transform)

    elif DATASET == 'imagewoof':
        NUM_CLASSES = 10
        extract_path = './data/imagewoof2-160' 
        train_dir = os.path.join(extract_path, 'imagewoof2-160/train')
        valid_dir = os.path.join(extract_path, 'imagewoof2-160/val')

        train_dataset = ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = ImageFolder(root=valid_dir, transform=test_transform)


    elif DATASET == 'combined':
        NUM_CLASSES = 212
        train_dataset, test_dataset = get_combined_dataset(train_transform, test_transform)
    else:
        print('Error: Wrong Dataset')
        exit(0)

    #NOTE: using this to overfit
    # from torch.utils.data import Subset
    # frac = 0.1
    # train_dataset = Subset(train_dataset, np.random.choice(np.arange(len(train_dataset)), int(len(train_dataset) * frac)))
    # test_dataset = Subset(test_dataset, np.random.choice(np.arange(len(test_dataset)), int(len(test_dataset) * frac)))

    num_workers = os.cpu_count()//2
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ConvNextV2(num_classes=NUM_CLASSES, weights_path=ARCHITECTURE)
    model.to(device)

    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        print('Optimizer not set')
        exit(0)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5) 
    # Cosine Scheduler that resets every 10 epochs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
        
    wandb.init(
        # set the wandb project where this run will be logged
        project="convnextv2",
        name=f"{args.exp}_{DATASET}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": ARCHITECTURE,
        "dataset": DATASET,
        "epochs": EPOCHS,
        }
    )
    save_path = os.path.join("models", DATASET)

    loss_statistics = train(
        model, optimizer, train_dataloader, EPOCHS, criterion, device, test_dataloader, scheduler, save_path
    )
    save_checkpoint(model, EPOCHS, optimizer, '00', os.path.join(save_path, "last.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training parser")
    parser.add_argument("--exp", "-e", type=str, help="WandB experiment name")
    args = parser.parse_args()
    main(args)