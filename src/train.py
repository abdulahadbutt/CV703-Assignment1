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
    with tqdm(data_loader, unit='batch') as data:
        batch_loss_list = []
        for batch in data:
            data.set_description(f"Epoch {epoch_index}")

            # ? Feeding to CNN
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            # print(inputs.shape, labels.shape)
            outputs = model(inputs)
            outputs = outputs['logits']
            # print(outputs)
            # exit(0)
            
            # ? Getting Loss
            batch_loss = criterion(outputs, labels)
            batch_loss_list.append(batch_loss.item())
            batch_loss.backward()

            # ? Gradient Descent
            optimizer.step()


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
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu' 
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            batch_loss_list = []
            for batch in data:
                data.set_description(f"Testing after epoch{epoch_index}")

                # ? Feeding to CNN
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                
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
    scheduler: torch.optim.lr_scheduler.LRScheduler
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
        if scheduler:
            scheduler.step()
        
        # * Testing Code
        test_epoch_statistics = test_one_epoch(
            model, optimizer, test_dataloader, epoch, criterion, device
        )
        latest_test_acc = test_epoch_statistics['accuracy']
        if latest_test_acc > best_acc:
            print(f'UPDATING BEST ACC [{best_acc}] -> [{latest_test_acc}]')
            best_acc = latest_test_acc
            save_checkpoint(model, epoch, optimizer, best_acc, 'models/best_model.pth')
        
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_epoch_statistics['epoch_loss'],
            "test/loss": test_epoch_statistics['epoch_loss'],
            "test/accuracy": test_epoch_statistics['accuracy']
        })
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
    
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "f1_score": f1_score,
        },
        path,
    )



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



if not DATA_AUG:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 (standard for pretrained models)
        transforms.ToTensor()  # Convert image to tensor
    ])
else:
    print('DATA AUG NOT IMPLEMENTED')
    exit(0)


if DATASET == 'flowers102':
    NUM_CLASSES = 102
    flowers102_root = './data/flowers-102'
    train_dataset = torchvision.datasets.Flowers102(root=flowers102_root, split='test', download=True, transform=transform)
    test_dataset = torchvision.datasets.Flowers102(root=flowers102_root, split='train', download=True, transform=transform)

elif DATASET == 'imagewoof':
    NUM_CLASSES = 10
    extract_path = './data/imagewoof2-160' 
    train_dir = os.path.join(extract_path, 'imagewoof2-160/train')
    valid_dir = os.path.join(extract_path, 'imagewoof2-160/val')

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=valid_dir, transform=transform)


elif DATASET == 'combined':
    NUM_CLASSES = 212
    train_dataset, test_dataset = get_combined_dataset()
else:
    print('Error: Wrong Dataset')
    exit(0)




train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ConvNextV2(num_classes=NUM_CLASSES, weights_path=ARCHITECTURE)
model.to(device)

if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    print('Optimizer not set')
    exit(0)

criterion = torch.nn.CrossEntropyLoss()
    

scheduler = None 
wandb.init(
    # set the wandb project where this run will be logged
    project="CV703-Assignment1",

    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": ARCHITECTURE,
    "dataset": DATASET,
    "epochs": EPOCHS,
    }
)


loss_statistics = train(
    model, optimizer, train_dataloader, EPOCHS, criterion, device, test_dataloader, scheduler
)
save_checkpoint(model, EPOCHS, optimizer, '00', 'models/last.pth')