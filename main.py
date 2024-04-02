import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import CTW2019DataLoader, KUleuvenDataLoader, DeepMIMODataLoader
from utils import MDELoss

def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
    if (args.datatype == "CTW2019"):
    # Create the data loader
        train_loader, test_loader = CTW2019DataLoader(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory,
            scenario=args.scenario)()
    elif (args.datatype == "KUleuven"):
        train_loader, test_loader = KUleuvenDataLoader(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory,
            scenario=args.scenario)()
    elif (args.datatype == "DeepMIMO"):
        train_loader, test_loader = DeepMIMODataLoader(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory,
            scenario=args.scenario)()


    # Define model
    model = init_model(args)
    model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])

    # Define loss function
    criterion = MDELoss().to(device)
    # criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return
    
    # Define optimizer and scheduler
    lr_init = 0.01 if args.scheduler == 'const' else 0.02
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    loss_list = []
    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=10 * len(train_loader),
                                            eta_min=1e-5)
        
    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume)
    
    # Start training
    trainer.loop(args.epochs, train_loader, test_loader)

    # Final testing
    loss, mde, rmse = Tester(model, device, criterion)(test_loader)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test MDE: {mde:.3e}"
          f"\n         test RMSE: {rmse:.3e}\n")
    

if __name__ == "__main__":
    main()