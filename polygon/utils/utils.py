import pandas as pd
import re
import os
import glob
import sys
import multiprocessing
import numpy as np
import logging
import itertools
import pickle
import torch
from collections import Counter, defaultdict
from typing import Optional, List, Iterable, Collection, Tuple


from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

#from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from polygon.utils.scoring_function import MoleculewiseScoringFunction 
from polygon.utils.scoring_function import ScoringFunctionBasedOnRdkitMol
from polygon.utils.scoring_function import ArithmeticMeanScoringFunction
from polygon.utils.scoring_function import MinMaxGaussianModifier
from polygon.utils.scoring_function import ThresholdedLinearModifier
from polygon.utils.scoring_function import LinearModifier




# custom scoring
from polygon.utils.custom_scoring_fcn import QED_custom 
from polygon.utils.custom_scoring_fcn import SAScorer
from polygon.utils.custom_scoring_fcn import CellLine
from polygon.utils.custom_scoring_fcn import LatentDistance
from polygon.utils.custom_scoring_fcn import LogP
from polygon.utils.custom_scoring_fcn import MW
from polygon.utils.custom_scoring_fcn import TaniSim
from polygon.utils.custom_scoring_fcn import LigandEfficancy






def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

################################################################################
# IO STUFF
################################################################################
def load_model(model_class, model_definition, device, model_params=None, copy_to_cpu=True):
    """
        Args:
            model_class: class defintion of model to load
            model_definition: path to model pickle
            device: cuda or cpu
            copy_to_cpu: bool

        Returns: an VAE model

        """
    # load model, optionally with configuration
    if model_params:
        model = model_class(**model_params).to(device)
    else:
        model = model_class().to(device)

    # load state dict
    if "cpu" in device:
        model.load_state_dict(torch.load(model_definition, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_definition))

    # place in eval mode
    model.eval()
    return model

def save_model(model,output_path, device=None):
    """ Save a model
        Args: 
            a model class:
            device: device to return the model to
            output_path (str): file location to save model to
        Returns:
            None
    """
    if device is None:
        try:
            device = model.device
        except:
            raise RuntimeError("Model must have a `device` parametere if none is specified.")

    # convert to cpu for storage
    model = model.to('cpu')
    # write the model to disk
    torch.save(model.state_dict(), output_path)
    # convert back to proper device
    model = model.to(device)

def load_smiles_from_file(smi_file):
    with open(smi_file) as f:
        return [canonicalize(s.strip()) for s in f]


####################################################################################
# scaffold substructure
####################################################################################
def remove_intermediates(d,t,to_ignore = ['c1ccccc1']):
    temp = d.copy()
    p = temp[t]
    if len(p)==0:
        return None
    if len(p)==1:
        return p[0]
    
    p = [i for i in p if i not in to_ignore]
    x = 0
    while len(p)>1:
        x+=1
        children = [r for q in p for r in temp[q] if r not in to_ignore]
        intermediates = set(p).difference(children)
        for i in intermediates:
            p.remove(i)
        if x>100:
            return None
    if len(p)<1:
        return None
    else:
        return p[0]
        
def get_lowest_level_substructure(scaffolds):
    uniq_scaffolds = np.unique(scaffolds)
    ms = [Chem.MolFromSmiles(s) for s in uniq_scaffolds]
    pairs = itertools.permutations(range(len(ms)),r=2)
    d = defaultdict(list)
    for i,j in pairs:
        has = ms[i].HasSubstructMatch(ms[j])
        opp = ms[j].HasSubstructMatch(ms[i])
        if has:
            if opp:
                # isomer? pass
                continue
            d[uniq_scaffolds[i]].append(uniq_scaffolds[j])
    result = []
    for scaffold in d.keys():
        
        p = remove_intermediates(d, scaffold)
        result.append((scaffold, p))
    result = pd.DataFrame(result, columns=['scaffold','substruct'])
    return result

def annotate_with_known_substrutures(smiles,
                                    substructures={ 'Biphenyl':'c1ccc(-c2ccccc2)cc1',
                                                    'Benzimidazole':'c1ccc2[nH]cnc2c1'},
                                    return_dataframe=True):
    """ Test each smile string for 
    """
    substructure_mols = {k: Chem.MolFromSmiles(v) for k,v in substructures.items()}
    membership = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        has = {k: m.HasSubstructMatch(v) for k,v in substructure_mols.items()}
        membership[s]=has

    if return_dataframe:
        return pd.DataFrame(membership).transpose()
    else:
        return membership

def get_scaffolds_counts(smiles):
    """
    Get Murcko Scaffold counts for a list of smiles
    """
    scaffold_counts = Counter(MurckoScaffoldSmilesFromSmiles(s) for s in smiles)
    return scaffold_counts

def pick_diverse_set(smiles,n_diverse=100):
    """
    Get diverse set of molecules from a list
    """
    logger = logging.getLogger()


    ms = smiles
    fps = [GetMorganFingerprint(Chem.MolFromSmiles(x),3) for x in ms]
    nfps = len(fps)
    picker = MaxMinPicker()
    logger.debug(f'Len FPS: {nfps}')

    if n_diverse>nfps:
        logger.warning("Exceeded Size")
        n_diverse=nfps

    f = lambda i,j:  1-DataStructs.DiceSimilarity(fps[i],fps[j])
    pickIndices = picker.LazyPick(f,nfps,n_diverse)
    picks = [ms[x] for x in pickIndices]
    
    #top = dff[dff['smiles'].isin(picks)]
    return picks

def get_fingerprint_similarity(scaffolds, return_long = False):
    ms = [Chem.MolFromSmiles(s) for s in scaffolds]
    fps = [FingerprintMols.FingerprintMol(x) for x in ms]
    pairs = itertools.permutations(range(len(ms)),r=2)
    sim = []
    for i,j in pairs:
        s = DataStructs.FingerprintSimilarity(fps[i],fps[j])
        sim.append((i,j,s))
    sim = pd.DataFrame(sim)
    if return_long:
        return sim
    sim_wide = sim.pivot_table(index=0,columns=1, values=2)
    np.fill_diagonal(sim_wide.values,1)
    return sim_wide

def get_fingerprint_similarity_pair(a,b):
    am = Chem.MolFromSmiles(a)
    bm = Chem.MolFromSmiles(b)
    af = FingerprintMols.FingerprintMol(am)
    bf = FingerprintMols.FingerprintMol(bm)
    return DataStructs.FingerprintSimilarity(af,bf)


####################################################################################
# scoring
####################################################################################
def build_scoring_function( scoring_definition,
                            encoding,
                            cell_line_model,
                            fscores,
                            opti='gauss',
                            return_individual = False, 
                            vae_model=None):
    """ Build scoring function """

    # scoring definition has columns:
    # category, name, minimize, mu, sigma, file, model, n_top
    df = pd.read_csv(scoring_definition, sep=",",header=0)
    scorers = {}

    # Get CpG encodings
    with open(encoding,'rb') as handle:
        encoding = pickle.load(handle)

    for i, row in df.iterrows():
        name = row['name']
        if row.category == "cell_line":
            cell_encodings = encoding[name]
            if opti == "gauss":
                cell_line_modifier =  MinMaxGaussianModifier(mu=row.mu,
                                                            sigma=row.sigma,
                                                            minimize=row.minimize)
            elif opti == "linear":
                cell_line_modifier = ThresholdedLinearModifier_min(threshold=row.mu)

            scorers[name] = CellLine(cell_encodings,
                                        score_modifier = cell_line_modifier,
                                        model_path=cell_line_model)


        elif row.category == "qed":
            scorers[name] = QED_custom(score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize))
        elif row.category == "sa":
            scorers[name] = SAScorer( 
                                    score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    fscores=fscores  
                                    )
        elif row.category == "latent_distance":
            if vae_model == None:
                raise RuntimeError("No vae class defined.  TODO: I need to fix this")
            # file 
            with open(row.file) as handle:
                smiles_targets = [line.rstrip() for line in handle]
                smiles_targets = canonicalize_list(smiles_targets)
            model = load_model(vae_model, row.model, "cpu") 
            scorers[name] = LatentDistance( smiles_targets=smiles_targets,
                                            model=model,
                                            n_top=row.n_top,
                                            agg="mean",
                                            score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                                    sigma=row.sigma,
                                                                                    minimize=row.minimize),
                                            ) 
        elif row.category == "tanimoto_similarity":
            # file 
            with open(row.file) as handle:
                smiles_targets = [line.rstrip() for line in handle]
                smiles_targets = canonicalize_list(smiles_targets)
            scorers[name] = TaniSim(smiles_targets=smiles_targets,
                                    n_top=row.n_top,
                                    agg="mean",
                                    score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    )  
        elif row.category == 'ligand_efficiency':
            scorers[name] = LigandEfficancy(
                                    score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    model_path=row.file
                                    )                           
        else:
            print("WTF Did not understand category: {}".format(row.category))
 
    scoring_function = ArithmeticMeanScoringFunction([scorers[i] for i in scorers])

    if return_individual:
        return scorers, scoring_function
    else:
        return scoring_function

def filter_results(df, mean=True, verbose=False):

    # cell lines
    LUNG_CELL_LINES = ['A549_LUNG','NCIH23_LUNG','NCIH1944_LUNG']#,'NCIH2122_LUNG']
    TOX_CELL_LINES = ['JURKAT_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','HEPG2_LIVER']

    if verbose:
        print("Starting: ", df.shape)
    # lung cell line max
    lc_max = 0.3
    if mean:
        df = df[df[LUNG_CELL_LINES].mean(axis=1)<lc_max]
    else:
        df = df[(df[LUNG_CELL_LINES]<lc_max).all(axis=1)]
    
    if verbose:
        print("After LUNG: ", df.shape)
    # tox cell line min
    tc_min = 0.9
    if mean:
        df = df[df[TOX_CELL_LINES].mean(axis=1)>tc_min]
    else:
        df = df[(df[TOX_CELL_LINES]>tc_min).all(axis=1)]
    
    if verbose:
        print("After TOX: ", df.shape)
    # qed min
    qed_min = 0.7
    df = df[df['qed']>qed_min]
    if verbose:
        print("After QED: ", df.shape)
    return df

def get_raw_scores(molecules, scorers, aggregate_scoring_function=None):
    """
    """
    #pass

    raw_scores = {}
    raw_scores['smiles'] = molecules

    for n, sf in scorers.items():
        l = []
        for m in molecules:
            try:
                s = sf.raw_score(m)
            except:
                # some error occured
                s = -1
            l.append(s)
        raw_scores[n] = l
        #raw_scores[n] = [sf.raw_score(m) for m in molecules]
    df = pd.DataFrame(raw_scores)
    if aggregate_scoring_function:
        df['Aggregate'] = aggregate_scoring_function.score_list(df['smiles'])
        df = df.sort_values('Aggregate', ascending=False)
    return df


####################################################################################
# smiles validation
####################################################################################
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None

def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()
        index += 1

    return balanced and len(s) == 0

def matched_ring(s):
    """ Check if rings are matched """
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0

def fast_verify(s):
    """ Quickly check if smiles might be valid """
    return matched_ring(s) and balanced_parentheses(s)

def random_enumerate_smiles(smil):
    """ randomly shuffle to get new smiles """
    # https://sourceforge.net/p/rdkit/mailman/message/36382511/
    # Construct a molecule from aSMILES string.
    m = Chem.MolFromSmiles(smil)
    if m is None:
        return smil
    N = m.GetNumAtoms()
    if N==0:
        return smil
    aids = list(range(N))
    random.shuffle(aids)
    m = Chem.RenumberAtoms(m,aids)
    try:
        n= random.randint(0,N-1)
        t= Chem.MolToSmiles(m, rootedAtAtom=n, canonical=False)
    except :
        return smil
    return t



def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return canonicalized_smiles

####################################################################################
# Random
####################################################################################
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rev_sort_by_len(x):
    """ Sort strings by length reverse """
    return sorted(x, key=len,reverse=True)

def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)

def torch_device(arg):
    """ from moses
    """
    if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
        raise TypeError(
            'Wrong device format: {}'.format(arg)
        )
    if arg != 'cpu':
        splited_device = arg.split(':')
        if (not torch.cuda.is_available()) or \
                (len(splited_device) > 1 and
                 int(splited_device[1]) > torch.cuda.device_count()):
            raise TypeError(
                'Wrong device: {} is not available'.format(arg)
            )
    return arg
