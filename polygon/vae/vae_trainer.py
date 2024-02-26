import time
import os
import logging
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
#from utils.moses_utils import OneHotVocab
from polygon.utils.moses_utils import CircularBuffer
from polygon.utils.moses_utils import set_torch_seed_to_all_gens
from polygon.utils.utils import save_model
from polygon.vae.vae_misc import CosineAnnealingLRWithRestart, KLAnnealer
from abc import ABC, abstractmethod


class VAETrainer(ABC):
    def __init__(self, model, **kwargs):
        self.model = model

        # defaults
        self.log_file = None
        self.model_save = "model_{}.pt".format(int(time.time()))
        self.save_frequency = 1
        self.n_workers = 0
        self.n_batch = 512
        self.n_epoch = 10


        self.n_last = 1000

        self.clip_grad = 50
        self.lr_start = 0.0003
        self.kl_w_start = 0
        self.kl_start = 1
        self.kl_w_end = 1
        self.lr_n_period = 10
        self.lr_n_restarts = 6
        self.lr_n_mult = 1
        self.lr_end = 0.0003


        # over ride deaults with kwargs
        self.__dict__.update(kwargs)


        self.device = self.model.device


    def _n_epoch(self):
        """ Default number of epochs to use
        """
        if self.n_epoch:
            return self.n_epoch
        return sum(
            self.lr_n_period * (self.lr_n_mult ** i)
            for i in range(self.lr_n_restarts)
        )

    def _save_model(self, output_path=None):
        """ Save a model
            Args: 
                output_path (str): file location to save model to
            Returns:
                None

        """
        # if no output path is provided use the instance variable
        if output_path is None:
            output_path = self.model_save
        # convert to cpu for storage
        self.model = self.model.to('cpu')
        # write the model to disk
        torch.save(self.model.state_dict(), output_path)
        # convert back to proper device
        self.model = self.model.to(self.device)


    def get_dataloader(self, data, batch_size=None, collate_fn=None, shuffle=True):
        if batch_size is None:
            batch_size = self.n_batch

        if collate_fn is None:
            collate_fn = self.model.get_collate_fn()

        if self.n_workers > 0:
            worker_init_fn = set_torch_seed_to_all_gens
        else:
            worker_init_fn = None

        loader = DataLoader(data,
                          batch_size=self.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers,
                          collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn)

        return loader

    def get_optim_params(self):
        return (p for p in self.model.vae.parameters() if p.requires_grad)

    def _train_epoch(self,  epoch, data_loader, kl_weight, optimizer=None, label="Training"):

        epoch_start_time = time.process_time()

        if optimizer is None:
            self.model.eval()
        else:
            self.model.train()


        # get the number of batches
        n_batches = len(data_loader)

        kl_loss_values = CircularBuffer(n_batches)
        recon_loss_values =  CircularBuffer(n_batches)
        loss_values =  CircularBuffer(n_batches)



        for i, input_batch in enumerate(data_loader):
            # record batch start time
            batch_start_time = time.process_time()
            #input_batch.to(self.device)
            #input_batch = tuple(data.to(self.model.device) for data in input_batch)
            # Forward
            kl_loss, recon_loss = self.model(input_batch)

            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(self.get_optim_params(),
                                self.clip_grad)
                optimizer.step()

            # Log

            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())

            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0.)

            # Update logger

            elapsed_time = time.process_time() - batch_start_time
            postfix = [f'{label}',
                       f'Batch {str(i).zfill(len(str(n_batches)))}/{n_batches}',
                       f'Time={elapsed_time:.2f}',
                       f'loss={loss.sum():.5f}',
                       f'(kl={kl_loss.sum():.5f}',
                       f'recon={recon_loss.sum():.5f})',
                       f'klw={kl_weight:.5f}',
                       f'lr={lr:.5f}']



            logging.debug(' '.join([str(i) for i in postfix]))

        elapsed_time = time.process_time() - epoch_start_time
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'{label}',
                   #f'Batch {str(i).zfill(len(str(n_batches)))}/{n_batches}',
                   f'Time={elapsed_time:.2f}',
                   f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f}',
                   f'lr={lr:.5f}']
        logging.info(' '.join([str(i) for i in postfix]))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix

    def _log_epoch(self, postfix, header=False):
        """ Write one epoch training results to log file
        """
        columns = ['epoch','kl_weight','lr','kl_loss','recon_loss','loss','mode']


        if self.log_file is None:
            return 
        with open(self.log_file, "a+") as handle:
            if header:
                line = ",".join(columns)
            else:
                line = ",".join([str(postfix[i]) for i in columns])
            handle.write("{}\n".format(line))

    def _train(self, train_loader, val_loader=None, n_epoch=None, save_frequency=None):
        """ 
        """
        device = self.device
        if n_epoch is None:
            n_epoch = self._n_epoch()

        if save_frequency is None:
            save_frequency = self.save_frequency


        optimizer = optim.Adam(self.get_optim_params(),
                               lr=self.lr_start)
        kl_annealer = KLAnnealer(n_epoch,
                                kl_start=self.kl_start,
                                kl_w_end=self.kl_w_end,
                                kl_w_start=self.kl_w_start)
        lr_annealer = CosineAnnealingLRWithRestart( optimizer,
                                                    lr_n_period = self.lr_n_period,
                                                    lr_n_restarts = self.lr_n_restarts,
                                                    lr_n_mult = self.lr_n_mult,
                                                    lr_end = self.lr_end
                                                   )
        self.model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)


            # tqdm_data = tqdm(train_loader,
            #                      desc='Training   (epoch #{})'.format(epoch))

            # postfix = self._train_epoch(epoch,
            #                             tqdm_data, kl_weight, optimizer)
            desc='Training   (epoch #{})'.format(str(epoch).zfill(len(str(n_epoch))))
            postfix = self._train_epoch(epoch,
                            train_loader, kl_weight, optimizer, label=desc)
            if self.log_file is not None:
                self._log_epoch(postfix)

            if val_loader is not None:
                # tqdm_data = tqdm(val_loader,
                #                  desc='Validation (epoch #{})'.format(epoch))
                desc = 'Validation (epoch #{})'.format(str(epoch).zfill(len(str(n_epoch))))
                postfix = self._train_epoch(epoch, val_loader, kl_weight, label=desc)
                if self.log_file is not None:
                    self._log_epoch(postfix)

            # save model
            if (self.model_save is not None) and \
                    (epoch % save_frequency == 0):
                output_path = '{path}_{epoch:03d}.pt'.format(
                                path=os.path.splitext(self.model_save)[0],
                                epoch=epoch)
                #self._save_model(output_path = output_path)
                save_model(self.model, output_path = output_path)


            # Epoch end
            lr_annealer.step()

    def fit(self, train_data, val_data=None, n_epoch=None, batch_size=None, save_frequency=None):
        #log_file = Logger() if self.log_file is not None else None
        
        if self.log_file is not None:
            # Write training parameters to log file
            params = ("Training Parameters:",
                      "\n".join(["{}: {}".format(i,j) for i,j in self.__dict__.items()]),
                      "----------------------")
            params = "\n".join(params) + "\n"
            with open(self.log_file, "w") as handle:
                handle.write(params)

        train_loader = self.get_dataloader(train_data,
            batch_size=batch_size,
            shuffle=True)

        val_loader = None if val_data is None else self.get_dataloader(
            val_data, batch_size=batch_size,
            shuffle=False
        )

        self._train(train_loader, val_loader, n_epoch=n_epoch, save_frequency=save_frequency)

        #self._save_model()
        if self.model_save is not None:
            save_model(self.model, output_path = self.model_save)
