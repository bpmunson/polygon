import os
import logging
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from functools import total_ordering
from typing import List, Set
from polygon.utils.scoring_function import ScoringFunction
from polygon.utils.utils import canonicalize_list
from polygon.utils.utils import save_model
from polygon.utils.utils import get_raw_scores
from polygon.vae.vae_model import VAE
from polygon.vae.vae_trainer import VAETrainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score
    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)
    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)

class SmilesVaeMoleculeGenerator:
    """
    character-based VAE language model 

    """
    def __init__(self,  model: VAE, max_len: int, device: str, out_dir: str, lr=0.0003, n_jobs=1, model_save=True) -> None:
        """
        Args:
            model: Pre-trained VAE model
            max_len: maximum SMILES length
            device: 'cpu' | 'cuda'
            out_dir: path to write results
        """
        self.device = device
        self.model = model
        self.max_len = max_len
        self.output_dir = out_dir
        self.n_jobs = n_jobs
        self.model_save = model_save
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        
        self.pool = joblib.Parallel(n_jobs=self.n_jobs)

        self.trainer = VAETrainer(      model=self.model,
                                        model_save = None,
                                        device=self.device,
                                        log_dir=self.output_dir,
                                        kl_w_start=1,
                                        kl_w_end=1,
                                        lr_start=self.lr,
                                        lr_end=self.lr)     


    def score_pool(self, smiles, scoring_function):
        joblist = (joblib.delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        return scores


    def sample(self, n_batch) -> List[str]:
        """
        Args:
            n_batch: number of molecules to sample

        Returns:
            a list of molecules
        """
        return self.model.sample(n_batch, max_len=self.max_len)

    def optimise(self, objective: ScoringFunction, start_population, keep_top, n_epochs, mols_to_sample,
                 optimize_n_epochs, optimize_batch_size, pretrain_n_epochs, save_frequency=10,
                 ind_scorers=None, save_payloads=False) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param objective: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param kwargs need to contain:
                keep_top: number of molecules to keep at each iterative finetune step
                mols_to_sample: number of molecules to sample at each iterative finetune step
                optimize_n_epochs: number of episodes to finetune
                optimize_batch_size: batch size for fine-tuning
                pretrain_n_epochs: number of epochs to pretrain on start population
        :return: Candidate molecules
        """
        int_results = self.pretrain_on_initial_population(objective, start_population,
                                                           pretrain_epochs=pretrain_n_epochs)


        scores = objective.score_list(start_population)

        int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(start_population, scores)]


        results: List[OptResult] = []
        seen: Set[str] = set()

        
        keep_starting_population = False
        if keep_starting_population:
            for k in int_results:
                if k.smiles not in seen:
                    results.append(k)
                    seen.add(k.smiles)

        for epoch in range(1, 1 + n_epochs):

            logger.debug(f'Starting Epoch: {epoch}')
            t0 = time.time()
            # get a sample of molecules from the model
            logger.debug(f'Get sample: {mols_to_sample}')

            samples = self.sample(mols_to_sample)
            t1 = time.time()

            # clean up any non valid molecules
            logger.debug(f'Canonicalize')
            canonicalized_samples = set(canonicalize_list(samples, include_stereocenters=True))

            # new molecules seen
            payload = list(canonicalized_samples.difference(seen))
            payload.sort()  # necessary for reproducibility between different runs

            # add the new stuff to tracker
            seen.update(canonicalized_samples)

            # score the new molecules
            logger.debug(f'Score {len(payload)} samples')

            # if self.n_jobs > 1:
            #     logger.debug(f'Using pool of size {self.n_jobs}')
            #     scores = self.score_pool(payload, objective)
            # else:
            if ind_scorers:
                raw_scores = get_raw_scores(payload, ind_scorers, aggregate_scoring_function=objective)
                scores = raw_scores['Aggregate'].tolist()
            else:
                scores = objective.score_list(payload)


            logger.debug(f'Got scores')

            int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(payload, scores)]
            int_results = sorted(int_results, reverse=True)
            t2 = time.time()


            # store the molecules
            results.extend(sorted(int_results, reverse=True)[0:keep_top])
            results.sort(reverse=True)
            subset = [i.smiles for i in results][0:keep_top]

            # alternatively, only retrain on best new molecules
            #subset = [i.smiles for i in int_results][0:keep_top]


            np.random.shuffle(subset)

            # split into train and test sets at 75%
            train_set = subset[0:int(3 * len(subset) / 4)]
            test_set = subset[int(3 * len(subset) / 4):]

            # override the batch size if the training set is smaller than
            # specified
            opt_batch_size = min(len(train_set), optimize_batch_size)

            # run training
            logger.debug(f'Train. Size={len(train_set)}')

            if optimize_n_epochs > 0:
                self.trainer.fit(train_set, test_set,
                                    n_epoch=optimize_n_epochs,
                                    batch_size=opt_batch_size,
                                    save_frequency=None)

            t3 = time.time()

            # update information
            logger.info(f'Generation {epoch} --- timings: '
                        f'sample: {(t1 - t0):.3f} s, '
                        f'score: {(t2 - t1):.3f} s, '
                        f'finetune: {(t3 - t2):.3f} s')


            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in results[:4])
            logger.info(f'Top 4:\n{top4}')

            # save the current epoch model
            if (self.model_save is not None) and \
                    (epoch % save_frequency == 0):
                base_name = os.path.join(self.output_dir, f'GDM_{epoch}.pt')
                #self._save_model(self.trainer.model, base_name)
                save_model(self.trainer.model, base_name)


            # write out payload
            logger.debug(f'Writing')
            if save_payloads:
                if ind_scorers:
                    ofp = os.path.join(self.output_dir, f'GDM_{epoch}_payload.csv')
                    raw_scores.to_csv(ofp, index=False, header=True, sep=',')
                else:
                    with open(os.path.join(self.output_dir, f'GDM_{epoch}_payload.txt'),'w') as handle:
                        for smiles in payload:
                            handle.write(f'{smiles}\n')

            # write out the current results
            to_write = [i.smiles for i in results][0:keep_top]
            with open(os.path.join(self.output_dir, f'GDM_ongoing_top_scoring_molecules.txt'),'w') as handle:
                for smiles in to_write:
                    handle.write(f'{smiles}\n')

            # write top scores
            # with open(self.output_dir+'/top_scores.csv','a') as ff:
            #     tmp_line = "{}\n".format(",",join([results[i].score for i in range(5)]))
            #     ff.write(tmp_line)

        # return the molecules
        return sorted(results, reverse=True)

    def pretrain_on_initial_population(self, scoring_function: ScoringFunction,
                                       start_population, pretrain_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param scoring_function: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param pretrain_epochs: number of epochs to finetune with start_population
        :return: Candidate molecules
        """
        seed: List[OptResult] = []

        start_population_size = len(start_population)

        training = canonicalize_list(start_population, include_stereocenters=True)

        # if len(training) != start_population_size:
        #     logger.warning("Some entries for the start population are invalid or duplicated")
        #     start_population_size = len(training)

        if start_population_size == 0:
            return seed

        # logger.info("finetuning with {} molecules for {} epochs".format(start_population_size, pretrain_epochs))

        # scores = scoring_function.score_list(training)
        # seed.extend(OptResult(smiles=smiles, score=score) for smiles, score in zip(training, scores))

        # train_seqs, _ = load_smiles_from_list(training, max_len=self.max_len)
        # train_set = get_tensor_dataset(train_seqs)

        batch_size = min(int(len(training)), 32)

        # print_every = len(training) / batch_size
        logger.info("Pretraining on Starting Population")
        losses = self.trainer.fit(training,
                                  batch_size=batch_size,
                                  n_epoch=pretrain_epochs)
        logger.info("Finished Pretraining")
        return seed



