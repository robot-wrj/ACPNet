import time
import os
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

from utils import logger
from utils.statics import AverageMeter, evaluator, evaluator_p

__all__ = ['Trainer', 'Tester']


field = ('rmse', 'mde', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_mde = Result()
        self.best_rmse = Result()

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None

    def loop(self, epochs, train_loader, test_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)

            # loss_list.append(self.train_loss.detach().cpu().numpy())
            # if ep % self.val_freq == 0:
            #     self.val_loss = self.val(val_loader)

            if ep % self.test_freq == 0:
                self.test_loss, mde, rmse = self.test(test_loader)
            else:
                mde, rmse = None, None

            # conduct saving, visualization and log printing
            self._loop_postprocessing(mde, rmse)

            # plt.plot(np.arange(len(loss_list)),np.array(loss_list))
            # plt.savefig('/home/wanrongjie/DeepMIMO_acpnet_noise.png')


    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    # def val(self, val_loader):
    #     r""" exam the model with validation set.

    #     Args:
    #         val_loader: (DataLoader): the validation data loader
    #     """

    #     self.model.eval()
    #     with torch.no_grad():
    #         return self._iteration(val_loader)

    def test(self, test_loader):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (h_data, pos_data) in enumerate(data_loader):
            # print(type(h_data), type(pos_data))
            # print(h_data.size(), pos_data.size())
            h_data = h_data.to(self.device)
            pos_data_pred = self.model(h_data)
            # print(pos_data, pos_data_pred)
            loss = self.criterion(pos_data_pred, pos_data)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MDE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}')



        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        r""" protected function which resume from checkpoint at the beginning of training.
        """

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_mde = checkpoint['best_mde']
        self.best_rmse = checkpoint['best_rmse']
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, mde, rmse):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_mde': self.best_mde,
            'best_rmse': self.best_rmse
        }

        # save model with best mde and rmse
        if mde is not None:
            if self.best_mde.mde is None or self.best_mde.mde > mde:
                self.best_mde = Result(mde=mde, rmse=rmse, epoch=self.cur_epoch)
                state['best_mde'] = self.best_mde
                self._save(state, name=f"best_mde.pth")
            if self.best_rmse.rmse is None or self.best_rmse.rmse > rmse:
                self.best_rmse = Result(mde=mde, rmse=rmse, epoch=self.cur_epoch)
                state['best_rmse'] = self.best_rmse
                self._save(state, name=f"best_rmse.pth")

        self._save(state, name='last.pth')

        # print current best results
        if self.best_mde.mde is not None:
            print(f'\n=! Best mde: {self.best_mde.mde:.3e} ('
                  f'Corresponding rmse={self.best_mde.rmse:.3e}; '
                  f'epoch={self.best_mde.epoch})'
                  f'\n   Best RMSE: {self.best_rmse.rmse:.3e} ('
                  f'Corresponding MDE={self.best_rmse.mde:.3e};  '
                  f'epoch={self.best_rmse.epoch})\n')


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, mde, rmse = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'    MDE: {mde:.3e}    RMSE: {rmse:.3e}\n')
        return loss, mde, rmse

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        # iter_rho = AverageMeter('Iter rho')
        # iter_nmse = AverageMeter('Iter nmse')
        iter_mde = AverageMeter('Iter mde')
        iter_rmse = AverageMeter('Iter rmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (h_data, pos_data) in enumerate(data_loader):
            h_data = h_data.to(self.device)
            pos_data_pred = self.model(h_data)
            loss = self.criterion(pos_data_pred, pos_data)
            # mde = loss
            mde, rmse = evaluator_p(pos_data_pred, pos_data)
            # rmse = torch.abs(rmse).mean()

            # Log and visdom update
            iter_loss.update(loss)
            iter_mde.update(mde)
            iter_rmse.update(rmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | MDE: {iter_mde.avg:.3e} |'
                            f'RMSE: {iter_rmse.avg:.3e} | time: {iter_time.avg:.3f}')

        # logger.info(f'=> Test rho:{iter_rho.avg:.3e}  NMSE: {iter_nmse.avg:.3e}\n')
        logger.info(f'=> Test MDE:{iter_mde.avg:.3e}  RMSE: {iter_rmse.avg:.3e}\n')
        
        return iter_loss.avg, iter_mde.avg, iter_rmse.avg
