import numpy as np
import pandas as pd
#from test import numBridgeheadsAndSpiro
import math
import gzip
import pickle

#import xgboost as xgb
import joblib as skjoblib

# guacamol
from polygon.utils.scoring_function import MoleculewiseScoringFunction
from polygon.utils.scoring_function import ArithmeticMeanScoringFunction
from polygon.utils.scoring_function import ScoringFunctionBasedOnRdkitMol
from polygon.utils.scoring_function import MinMaxGaussianModifier

# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Mol
from rdkit.six import iteritems
#from rdkit.six.moves import cPickle
import _pickle as cPickle
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs

import torch


class LigandEfficancy(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, score_modifier, model_path):
        super().__init__(score_modifier=score_modifier)
        with open(model_path,'rb') as handle:
            self.rfr = pickle.load(handle)
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        m = Chem.MolFromSmiles(smiles)
        mph = Chem.AddHs(m)
        N = mph.GetNumAtoms() - mph.GetNumHeavyAtoms()
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2)
        fp = np.array([fp])
        pic50 = self.rfr.predict(fp)
        LE = 1.4*(pic50)/N
        return LE[0]

class ToxicityScore(MoleculewiseScoringFunction):
    """
    Scoring function that determines the toxicity score for molecules from SMILES strings with a XGB model
    """
    def __init__(self, model, scaler):
        super().__init__()
        self.model  = model
        self.scaler = scaler
        self.names = ['MolWt','TPSA', 'NumHDonors','NumHAcceptors', 'MolLogP', 'HeavyAtomCount', 'NumRotatableBonds', 'RingCount']
        self.des_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.names)
        self.threshold = 250
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        npfp = np.array( list(fp.ToBitString())).astype('int8')
        z2=self.des_calc.CalcDescriptors(mol)
        zf=np.concatenate( (npfp,z2) )
        zf = zf.reshape(1,-1)
        zf = self.scaler.transform(zf)
        preds=self.model.predict(zf)[0]
        return np.minimum( preds, self.threshold)/self.threshold




class DualTani(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, target_string1, target_string2, thresh):
        super().__init__()
        self.target1 =target_string1
        self.target2 =target_string2
        target_mol1  =Chem.MolFromSmiles(self.target1)
        target_mol2  =Chem.MolFromSmiles(self.target2)
        #if target_mol is None:
        #    raise RuntimeError(f'The similarity target {target} is not a valid molecule.')
        self.ref_fp1 = AllChem.GetMorganFingerprintAsBitVect(target_mol1,3,nBits=4096)
        self.ref_fp2 = AllChem.GetMorganFingerprintAsBitVect(target_mol2,3,nBits=4096)
        self.threshold = thresh
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        tani1 = TanimotoSimilarity(fp, self.ref_fp1)
        tani2 = TanimotoSimilarity(fp, self.ref_fp2)
        tani = np.minimum(tani1, tani2)
        return np.minimum(tani, self.threshold)/self.threshold

class QED_custom(MoleculewiseScoringFunction):
    def __init__(self, score_modifier):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        qed = Descriptors.qed(mol)
        return qed

    


class TaniSim(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, smiles_targets, score_modifier, n_top=None, agg="mean"):
        super().__init__(score_modifier=score_modifier)
        self.targets = smiles_targets
        self.mols = [Chem.MolFromSmiles(s) for s in self.targets]
        self.fp_targets = [ AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) for mol in self.mols]
        if n_top is None:
            n_top = len(self.targets)
        self.n_top=int(n_top)
        self.agg = agg
   
    def raw_score(self, smiles):
        """ Ge distance to set """
        mol = Chem.MolFromSmiles(smiles)
        fp =  AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        tani = np.array([TanimotoSimilarity(fp, fp_targets_i) for fp_targets_i in self.fp_targets])
        if self.n_top:
            tani.sort()
            tani = tani[::-1] # reverse because we what high tanimoto similarity
            tani = tani[:self.n_top]
        if self.agg == "mean":
            return tani.mean()
        if self.agg == "max":
            return tani.max()
        if self.agg == "min":
            return tani.min()
         
class LatentDistance(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, smiles_targets, model, score_modifier, n_top=None, agg="mean"):
        super().__init__(score_modifier=score_modifier)
        self.model = model
        self.collate_fn = model.get_collate_fn()
        self.x_targets = self.collate_fn(smiles_targets)
        self.z_targets = self.model.encode(self.x_targets)
        
        if n_top is None:
            self.n_top = len(smiles_targets)
        else:
            self.n_top = int(n_top)
        self.agg = agg
   
    def raw_score(self, smiles):
        """ Ge distance to set """

        x = self.collate_fn([smiles])
        z = self.model.encode(x)
        norm = torch.norm(self.z_targets - z, dim=1)
        norm = norm.sort()[0][:self.n_top]
        norm = norm.detach().numpy()
        if self.agg == "mean":
            return norm.mean()
        if self.agg == "max":
            return norm.max()
        if self.agg == "min":
            return norm.min()
    
    def get_all(self, smiles):
        z = self.model.encode(smiles)
        norm = torch.norm(self.z_targets - z, dim=1)
        return norm
        


class CellLine(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, cellline_cnv, score_modifier, model_path=None):
        super().__init__(score_modifier=score_modifier)
        self.cellline_cnv = cellline_cnv

        #with open('/gpfs/amarolab/cdparks/test/brent/brent.pickle','rb') as handle:
        if model_path is None:
            model_path = '../../data/MODEL_allData_DNAmeth_iterations23.pickle'
        with open(model_path,'rb') as handle:
            self.mlp = pickle.load(handle, encoding='Latin1')
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        d = Chem.MolFromSmiles(smiles)
        fingerprint = list(SimilarityMaps.GetMorganFingerprint(d, fpType='bv',radius=2))
        X = fingerprint + self.cellline_cnv
        X = np.asarray(X).reshape(1,-1)
        preds = self.mlp.predict(X)[0]
        return preds



class LigandSets(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, data_target1, score_modifier):
        super().__init__(score_modifier=score_modifier)

        print('we are now loading:', data_target1)
        self.data1 = pd.read_csv(data_target1, index_col=None, header=None )
        self.data1.columns = ['smiles']
        self.smiles1 = list( self.data1['smiles'] )
        self.fp1 =[]

        for s in self.smiles1:
            mol = Chem.MolFromSmiles(s)
            self.fp1.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )

        #self.threshold = thresh
        print(self.data1.shape)

    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        scores=[]
        for fps in self.fp1:
            scores.append( TanimotoSimilarity(fp, fps) )
        
        #tani = np.median( np.array(scores) )
        tani = np.max(np.array(scores) )

        return tani
        #tani = np.max(np.array(scores) )
        #tani = np.min(np.array(scores) )
        #return np.minimum(tani, self.threshold)/self.threshold


class LigandSets_dual(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, data_target1, data_target2, thresh):
        super().__init__()

        print('we are now loading:', data_target1)
        self.data1 = pd.read_csv(data_target1, index_col=None, header=None )
        self.data1.columns = ['smiles']
        print('we are now loading:', data_target2)
        self.data2 = pd.read_csv(data_target2, index_col=None, header=None )
        self.data2.columns = ['smiles']

        self.smiles1 = list( self.data1['smiles'] )
        self.smiles2 = list( self.data2['smiles'] )

        self.fp1 =[]
        self.fp2 =[]

        for s in self.smiles1:
            mol = Chem.MolFromSmiles(s)
            self.fp1.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )


        for s in self.smiles2:
            mol = Chem.MolFromSmiles(s)
            self.fp2.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )

        self.threshold = thresh
        print(self.data1.shape)
        print(self.data2.shape)

    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        maxs2 = 0
        for fps in self.fp2:
            tani2 = TanimotoSimilarity(fp, fps)
            if( tani2 > maxs2):
                maxs2 = tani2
        maxs1 = 0
        for fps in self.fp1:
            tani1 = TanimotoSimilarity(fp, fps)
            if( tani1 > maxs1):
                maxs1 = tani1
        
        tani = np.minimum(maxs1, maxs2)
        return np.minimum(tani, self.threshold)/self.threshold



class LigandSets_truel(MoleculewiseScoringFunction):
    """
    Scoring function that determines the toxicity score for molecules from SMILES strings with a XGB model
    """
    def __init__(self, data_target1, data_target2, data_target3, thresh):
        super().__init__()

        print('we are now loading:', data_target1)
        self.data1 = pd.read_csv(data_target1, index_col=None, header=None )
        self.data1.columns = ['smiles']
        print('we are now loading:', data_target2)
        self.data2 = pd.read_csv(data_target2, index_col=None, header=None )
        self.data2.columns = ['smiles']
        print('we are now loading:', data_target3)
        self.data3 = pd.read_csv(data_target3, index_col=None, header=None )
        self.data3.columns = ['smiles']

        self.smiles1 = list( self.data1['smiles'] )
        self.smiles2 = list( self.data2['smiles'] )
        self.smiles3 = list( self.data3['smiles'] )

        self.fp1 =[]
        self.fp2 =[]
        self.fp3 =[]

        for s in self.smiles1:
            mol = Chem.MolFromSmiles(s)
            self.fp1.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )

        for s in self.smiles2:
            mol = Chem.MolFromSmiles(s)
            self.fp2.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )

        for s in self.smiles3:
            mol = Chem.MolFromSmiles(s)
            self.fp3.append(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096) )


        self.threshold = thresh
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)

        maxs1 = 0
        for fps in self.fp1:
            tani1 = TanimotoSimilarity(fp, fps)
            if( tani1 > maxs1):
                maxs1 = tani

        maxs2 = 0
        for fps in self.fp2:
            tani2 = TanimotoSimilarity(fp, fps)
            if( tani2 > maxs2):
                maxs2 = tani

        maxs3 = 0
        for fps in self.fp3:
            tani3 = TanimotoSimilarity(fp, fps)
            if( tani3 > maxs3):
                maxs3 = tani

        tani = np.minimum(maxs1, maxs2)
        tani = np.minimum(max3,  tani)

        return np.minimum(tani, self.threshold)/self.threshold



class CustomTani(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, target_string):
        super().__init__()
        #self.target ='CC(C)CC1=CC=C(C=C1)C(C)C(O)=O'
        self.target =target_string
        target_mol = Chem.MolFromSmiles(self.target)
        if target_mol is None:
            raise RuntimeError(f'The similarity target {target} is not a valid molecule.')
        self.ref_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol,3,nBits=4096)
        self.threshold = 0.8
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=4096)
        tani = TanimotoSimilarity(fp, self.ref_fp)
        return np.minimum(tani, self.threshold)/self.threshold



class SAScorer(MoleculewiseScoringFunction):
    def __init__(self, score_modifier, fscores=None):
        super().__init__(score_modifier=score_modifier)
        if fscores is None:
            fscores = '../../data/fpscores.pkl.gz'
        self.fscores = cPickle.load(gzip.open(fscores ))
        outDict = {}
        for i in self.fscores:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self.fscores = outDict
    
    def numBridgeheadsAndSpiro(self, mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro


    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        m = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprint(m,2)  # <- 2 is the *radius* of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in iteritems(fps):
            nf += v
            sfp = bitId
            score1 += self.fscores.get(sfp, -4) * v
        score1 /= nf
        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        ri = m.GetRingInfo()
        nBridgeheads, nSpiro = self.numBridgeheadsAndSpiro(m, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        
        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)
        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5
        sascore = score1 + score2 + score3
        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        return sascore
        #return self.threshold/np.maximum(sascore, self.threshold)

class LogP(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        logp = Descriptors.MolLogP(mol)
        return logp

class MW(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        try:
            mw=  Descriptors.ExactMolWt( Chem.MolFromSmiles(smiles) )
            return mw
        except:
            #print('we cant calculate molecular weight', smiles )
            return -1.
