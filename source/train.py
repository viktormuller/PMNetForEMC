import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation.eval import ROI_RMSE_Loss
from dataloader.loader_USC import PMnet_usc
from torch.utils.data import DataLoader
from network.PMNet import PMNet

def getTrainingData(data_dir, samples_csv, batch_size=16, **kwargs):
    usc_training_data=PMnet_usc(dir_dataset=data_dir, csv_file=samples_csv)
    print(f"Number of samples in the training dataset: {len(usc_training_data)}")
    usc_training_dataloader = DataLoader(usc_training_data, batch_size=batch_size, shuffle=True, **kwargs)
    return usc_training_dataloader

def getTestData(data_dir, samples_csv, batch_size=1000, **kwargs):
    usc_test_data=PMnet_usc(csv_file=samples_csv, dir_dataset=data_dir)
    print(f"Number of samples in the test dataset: {len(usc_test_data)}")
    usc_test_dataloader = DataLoader(usc_test_data, batch_size=batch_size, shuffle=True, **kwargs)
    return usc_test_dataloader

from evaluation.eval import roi_rmse_loss

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            roi = data[:, 0, :, :].unsqueeze(1)
            target_with_roi = torch.cat((target, roi), dim=1)
            output = model(data)
            test_loss= roi_rmse_loss(output, target_with_roi)
    print(
        "Test set: Average loss: {:.4f}\n".format(
            test_loss
        )
    )

import zipfile

def train(args):
    use_cuda = args.num_gpus > 0
    print("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    with zipfile.ZipFile(args.data_dir + "/" + args.data_file, 'r') as zip_ref:
        zip_ref.extractall(args.data_dir)
    USC_suffix = '/data/USC'
    data_dir = args.data_dir + USC_suffix
    usc_training_dataloader = getTrainingData(
        data_dir=data_dir, 
        samples_csv=data_dir + '/train.csv', 
        batch_size=args.batch_size, 
        **kwargs)
    usc_test_dataloader = getTrainingData(
        data_dir=data_dir, 
        samples_csv=data_dir + '/test.csv', 
        batch_size=args.test_batch_size, 
        **kwargs)
    
    bnlayers = [args.bottleneck_layers] * 4
    
    network = PMNet(bnlayers, [1,1,1], None, 16).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10, gamma=0.5, verbose=True)
    loss_fn = ROI_RMSE_Loss()
    
    for epoch in range(1,2):        
        network.train()
        for batch_idx, (data, target) in enumerate(usc_training_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)    
            roi = data[:, 0, :, :].unsqueeze(1)
            target_with_roi = torch.cat((target, roi), dim=1)
            loss = loss_fn(output,target_with_roi)
            loss.backward()
            optimizer.step()
            idx = batch_idx +1
            if (idx) % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        (idx) * len(data),
                        len(usc_training_dataloader.sampler),
                        100.0 * idx / len(usc_training_dataloader),
                        loss.item(),
                    )
                )
        test(network, usc_test_dataloader, device)
        scheduler.step()
    save_model(network, args.model_dir)
    print("model saved successfully to: {}".format(args.model_dir))

def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    
    parser.add_argument(
        "--bottleneck-layers",
        type=int,
        default=1,
        metavar="N",
        help="number of bottleneck layers per ResLayer (default: 1)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="sets the initial learning reate (default: 0.001)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="data.zip",
        help="the file name for the zip file containing the training and test data",
    )

    # Container environment
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR","model"))
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=os.getenv("SM_CHANNEL_TRAINING", '.'))
    parser.add_argument("--num-gpus", type=int, default=os.getenv("SM_NUM_GPUS",0))

    train(parser.parse_args())