import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pathlib import Path

import logging

def train_ligand_binding_model(target_unit_pro_id,binding_db_path,output_path):
    binddb = pd.read_csv(binding_db_path, sep="\t",header=0,low_memory=False,error_bad_lines=False)


    d = binddb[binddb['UniProt (SwissProt) Primary ID of Target Chain']==target_unit_pro_id]
    d = d[['Ligand SMILES','IC50 (nM)','Kd (nM)']]
    d.columns = ['smiles','ic50','kd50']

    logging.debug(f'Number of obs: {d.shape[0]}:')
    logging.debug(f'{d.head()}')

    vs = []
    for i,j in d[['ic50','kd50']].values:
        try:
            v = float(i)
        except ValueError:
            v = float(i[1:])
        try:
            w = float(j)
        except ValueError:
            w = float(j[1:])       

        t = pd.Series([v,w]).dropna().min()
        t = -np.log10(t*1E-9) 
        vs.append(t)
        
    d['metric_value'] = vs
    d = d[['smiles','metric_value']]
    d['metric_value'] = d['metric_value'].astype(float)
    d = d.drop_duplicates(subset='smiles')
    d = d.dropna()

    logging.debug(f'Number of obs: {d.shape[0]}:')

    if d.shape[0]<10:
        logging.info('Less than 10 compound-target pairs. Not fitting a model')
        return 1
    # convert to fingerprint
    fps = []
    values = []
    for x,y in d[['smiles','metric_value']].values:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2)
        except:
            continue
        
        fps.append(fp)
        values.append(y)

    X = np.array(fps)
    y = np.array(values)

    regr = RandomForestRegressor(n_estimators=1000,random_state=0,n_jobs=-1)
    regr.fit(X,y)
    regr.score(X,y)

    logging.debug(regr.score(X,y))

    if output_path is None:
        output_path = f'{target_unit_pro_id}_rfr_ligand_model.pt'

    with open(output_path, 'wb') as handle:
        s = pickle.dump(regr, handle)

    return 1
