import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
import sys
import itertools
pd.set_option('display.max_rows', 500)

import math
import umap
from sklearn.cluster import DBSCAN
import scipy.stats
import scipy.optimize as opt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import PandasTools
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

import useful_rdkit_utils as uru

from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import Descriptors
from IPython.display import HTML
from rdkit import Chem
import requests
# lib_file = requests.get("https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py")
# ofs = open("scaffold_finder.py","w")
# print(lib_file.text,file=ofs)
# ofs.close()
#from scaffold_finder import generate_fragments, find_scaffolds, get_molecules_with_scaffold, cleanup_fragment

import mols2grid
import datamol as dm
from sort_and_slice_ecfp_featuriser import create_sort_and_slice_ecfp_featuriser
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics.pairwise import pairwise_distances
import brewer2mpl
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE 
import colorcet as cc


import numba
@numba.njit()

def show_df(df):
    '''Display DataFrame with HTML table'''
    return HTML(df.to_html(notebook=True))

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None
    
def mol_to_smiles(mol):
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None

def remove_salts(smiles):
    """
    Remove salts, counterions, solvents from SMILES and return the largest fragment.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. isolate fragments
        fragmenter = rdMolStandardize.LargestFragmentChooser()
        mol = fragmenter.choose(mol)

        # 2. remove redundant charge for canonical SMILES
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def canon_scaffold(x):
    # Case 1: RDKit Mol
    if isinstance(x, Chem.Mol):
        return mol_to_smiles(x)  # canonical by default

    # Case 2: SMILES string
    elif isinstance(x, str):
        m = smiles_to_mol(x)
        return mol_to_smiles(m) if m is not None else None

    # Case 3: Other types
    else:
        return None

def key_sorter(df):
    sorted_keys = sorted(df["Key"].unique())
    # Create a mapping from the sorted values to a sequential range starting from 1
    mapping = {old_key: new_key for new_key, old_key in enumerate(sorted_keys, start=1)}
    # Apply the mapping to the 'Key' column
    df["Key"] = df["Key"].map(mapping)
    return(df)


def Cluster_PLSR(cl, preprocessed_df, SAR_df,
                 comp_disp=False):
    #---1. Filtering and Identifying scaffold by cluster (input as cl)---
    pp_df = preprocessed_df[preprocessed_df["Cluster"].isin(cl)] # Select all molecules belonging to the given cluster
    pp_df.reset_index(inplace=True,drop=True)
    mol_df,scaffold_df = find_scaffolds(pp_df) # searches for possible scaffolds within this subset
    
    #---2. find a representative scaffold and map with other candidates for common structure---
    CORE_SMILES = scaffold_df.head(1)["Scaffold"].iloc[0] # Pick one representative: the first row pf the table
    scaffold_mol = Chem.MolFromSmiles(CORE_SMILES)
    scaffold_smiles_list1, mols_df1 = get_molecules_with_scaffold(CORE_SMILES,mol_df,pp_df) # identify which molecules contain this scaffold
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles_list1[0])
    if comp_disp==True:display(scaffold_mol)
    
    #---3. Remove atom map number for pure chemical structure
    clean_scaffold_mol, _ = cleanup_fragment(scaffold_mol) 
    Compute2DCoords(clean_scaffold_mol) # get 2D coords to align the molecule consistently for drawing
    pp_df['mol'] = pp_df.SMILES.apply(Chem.MolFromSmiles) # SMILE strings to RDkit object
    
    #---4. (optional) align molecules with scaffolds for better visualization
    list_not_sim=[]
    for i,mol in enumerate(pp_df.mol):
        try:
            AllChem.GenerateDepictionMatching2DStructure(mol,clean_scaffold_mol)
        except:
            list_not_sim.append(pp_df["Name"][i]) # record those cannot be aligned
            continue
            
    pre_df = pp_df[~pp_df['Name'].isin(list_not_sim)] #
    pre_df.reset_index(inplace=True,drop=True)
    
    #---5. perform RGD
    rgd,failed = rdRGroupDecomposition.RGroupDecompose([clean_scaffold_mol],pp_df["mol"].values,asRows=False) # RUN RGD using the cleaned scaffold as the core
    '''
    rgd: a dictionary with keys 'Core', 'R1', 'R2'...
    failed: indices of molecules that could not be matched to the scaffold
    '''

    #---6. Separate successful/failed molecules
    RGD_df = pp_df.drop(failed,axis=0) # RGD_df = only successful molecules
    failed_df = pp_df[pp_df.index.isin(failed)] # split the dataset into successfully decomposed molecules and those failed
    display(f"{len(pp_df)}>>>>>>>{len(RGD_df)}")
    
    #---7. Re-run RGD and collect representative scaffold within each cluster for presentation
    mol_df,scaffold_df = find_scaffolds(RGD_df)
    CORE_SMILES = scaffold_df.head(1)["Scaffold"].iloc[0]
    scaffold_mol = Chem.MolFromSmiles(CORE_SMILES)
    scaffold_smiles_list1, mols_df1 = get_molecules_with_scaffold(CORE_SMILES,mol_df,pp_df)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles_list1[0])
    preclean_scaffold = scaffold_mol # for visualization
    display(scaffold_mol)
    used_scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True)
    
    r_groups = [x for x in rgd.keys() if x != "Core"] # identify how many R-groups are detected
    R_length = len(r_groups)
    
    RGD_df.reset_index(inplace=True,drop=True)
    
    #Building RGD_df
    for r in sorted(r_groups):
        RGD_df[r] = rgd[r]
        RGD_df[r] = RGD_df[r].apply(Chem.MolToSmiles) # convert each R-goup to its SMILE string

#   RGD_df =RGD_df[['Name', 'pIC50','SMILES', 'mol', "R1","R2"]] 
    
    Rgroups_mls_df = RGD_df.copy(deep=True)
    Rgroups_mls_df.drop(["pIC50","mol","Name"],axis=1,inplace=True)
    
    # Convert all SMILES columns to molecule columns
    for col in Rgroups_mls_df.columns:
        Rgroups_mls_df[col] = Rgroups_mls_df[col].apply(smiles_to_mol)
        
    Rgroups_mls_df_raw =Rgroups_mls_df.copy(deep=True)
    for i in range(len(Rgroups_mls_df)):
        for i2 in range(1,R_length+1):
            if Rgroups_mls_df[f"R{i2}"][i] == None:
                continue
            else:
                cleanup_fragment(Rgroups_mls_df[f"R{i2}"][i])
                
    Rgroups_mls_df["Name"] = RGD_df["Name"]


    # Apply the function to create a new column 'SMILES'
    for i2 in range(1,(R_length+1)):
        Rgroups_mls_df[f'R{i2}_SMILES'] = Rgroups_mls_df[f'R{i2}'].apply(mol_to_smiles)
        
    #---8. output
    cluster_df_out = SAR_df[SAR_df["ECFP_Cluster"].isin(cl)]
    cluster_df_out = cluster_df_out.drop(['pIC50'],axis=1)
    Rgroups_mls_df.drop("SMILES",axis=1,inplace=True)

    Rgroups_mls_df.rename(columns={'Name': 'Compound name'}, inplace=True)
    
    final_output = pd.merge( cluster_df_out,Rgroups_mls_df, on=['Compound name'], how='inner')
    
    return(final_output,R_length, clean_scaffold_mol,preclean_scaffold,failed_df, used_scaffold)


def indi_Cluster_PLSR(cl,preprocessed_df,SAR_df,
                 comp_disp=False):
    #Filtering and Identifying scaffold
    pp_df = preprocessed_df[preprocessed_df["Cluster"]==cl]
    pp_df.reset_index(inplace=True,drop=True)
    mol_df,scaffold_df = find_scaffolds(pp_df)
    
    #Scaffolding
    CORE_SMILES = scaffold_df.head(1)["Scaffold"].iloc[0]
    scaffold_mol = Chem.MolFromSmiles(CORE_SMILES)
    scaffold_smiles_list1, mols_df1 = get_molecules_with_scaffold(CORE_SMILES,mol_df,pp_df)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles_list1[0])
    if comp_disp==True:display(scaffold_mol)
    
   
    clean_scaffold_mol, _ = cleanup_fragment(scaffold_mol)
    Compute2DCoords(clean_scaffold_mol)
    pp_df['mol'] = pp_df.SMILES.apply(Chem.MolFromSmiles)
    
    #Perform R group decomposition
    j=0
    list_not_sim=[]
    for i,mol in enumerate(pp_df.mol):
        try:
            AllChem.GenerateDepictionMatching2DStructure(mol,clean_scaffold_mol)
        except:
            j=j+1
            list_not_sim.append(pp_df["Name"][i])
            continue
            
            
    pre_df = pp_df[~pp_df['Name'].isin(list_not_sim)]
    pre_df.reset_index(inplace=True,drop=True)
    
    rgd,failed = rdRGroupDecomposition.RGroupDecompose([clean_scaffold_mol],pp_df["mol"].values,asRows=False)
    RGD_df = pp_df.drop(failed,axis=0)
    failed_df = pp_df[pp_df.index.isin(failed)]
    display(f"{len(pp_df)}>>>>>>>{len(RGD_df)}")
    
    
    mol_df,scaffold_df = find_scaffolds(RGD_df)
    CORE_SMILES = scaffold_df.head(1)["Scaffold"].iloc[0]
    scaffold_mol = Chem.MolFromSmiles(CORE_SMILES)
    scaffold_smiles_list1, mols_df1 = get_molecules_with_scaffold(CORE_SMILES,mol_df,pp_df)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles_list1[0])
    preclean_scaffold = scaffold_mol
    display(scaffold_mol)
    
    r_groups = [x for x in rgd.keys() if x != "Core"]
    R_length = len(r_groups)
    
    RGD_df.reset_index(inplace=True,drop=True)
    
    #Building RGD_df
    for r in sorted(r_groups):
        RGD_df[r] = rgd[r]
        RGD_df[r] = RGD_df[r].apply(Chem.MolToSmiles)

#     RGD_df =RGD_df[['Name', 'pIC50','SMILES', 'mol', "R1","R2"]] 
    
    Rgroups_mls_df = RGD_df.copy(deep=True)
    Rgroups_mls_df.drop(["pIC50","mol","Name"],axis=1,inplace=True)


    # Convert all SMILES columns to molecule columns
    for col in Rgroups_mls_df.columns:
        Rgroups_mls_df[col] = Rgroups_mls_df[col].apply(smiles_to_mol)
        
    Rgroups_mls_df_raw =Rgroups_mls_df.copy(deep=True)
    for i in range(len(Rgroups_mls_df)):
        for i2 in range(1,R_length+1):
            if Rgroups_mls_df[f"R{i2}"][i] == None:
                continue
            else:
                cleanup_fragment(Rgroups_mls_df[f"R{i2}"][i])
                
    Rgroups_mls_df["Name"] = RGD_df["Name"]
    
    # Apply the function to create a new column 'SMILES'
    for i2 in range(1,(R_length+1)):
        Rgroups_mls_df[f'R{i2}_SMILES'] = Rgroups_mls_df[f'R{i2}'].apply(mol_to_smiles)
        
    cluster_df_out = SAR_df[SAR_df["ECFP_Cluster"]==cl]
    cluster_df_out = cluster_df_out.drop(['pIC50'],axis=1)
    Rgroups_mls_df.drop("SMILES",axis=1,inplace=True)

    Rgroups_mls_df.rename(columns={'Name': 'Compound name'}, inplace=True)
    
    final_output = pd.merge( cluster_df_out,Rgroups_mls_df, on=['Compound name'], how='inner')
    
    return(final_output,R_length, clean_scaffold_mol,preclean_scaffold,failed_df)


def tanimoto_dist(a,b):
    dotprod = np.dot(a,b)
    tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
    return 1.0-tc


def FP_For_UMAP_Generator(mols, radius=3, nBits=2048):
    """
    Generate a 0/1 Morgan fingerprint matrix (n_mols x nBits) for a set of RDKit Mol objects, suitable for UMAP or other dimensionality reduction methods.
    - Uses the updated API: rdFingerprintGenerator.GetMorganGenerator (no deprecation warning)
    - Uses dtype uint8 to save memory
    """
    gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    fps_np = []

    for mol in mols:
        if mol is None:
            continue  
        fp = gen.GetFingerprint(mol)  # RDKit ExplicitBitVect
        # convert FP object (bit vector) to np array
        # bit vector: A binary bit vector that internally stores 0/1 bits, cannot directly feed into UMAP, t-SNE
        arr = np.zeros((nBits,), dtype=np.uint8)  # Create an empty NumPy array of length nBits, pre-filled with zeros
        DataStructs.ConvertToNumpyArray(fp, arr)  # Copy the bits of the bit vector (fp) into the arr. This operation modifies the contents of arr in place.
        fps_np.append(arr)

    X = np.vstack(fps_np) if fps_np else np.empty((0, nBits), dtype=np.uint8)
    print(f"{X.shape[0]} mols loaded")
    return X

# def FP_For_UMAP_Generator(mols):

#     for mol in mols:
#         AllChem.Compute2DCoords(mol) # compute 2D coordinations for each [Mol]
#     X_molecular_fingerprint = []
#     for mol in mols:
#         arr = np.zeros((0,))
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) # generate Morgan fingerprint for each [Mol]
#         DataStructs.ConvertToNumpyArray(fp, arr) # convert the RDKit fingerprint bit vector into a NumPy array (arr)
#         X_molecular_fingerprint.append(arr)
#     print('{} mols loaded'.format(len(X_molecular_fingerprint)))
#     return(X_molecular_fingerprint)

def umap_opt(some_df,X, nn, md,Hue_fil="Butina_clustering",legend=False):
    umap_X = umap.UMAP(n_neighbors=nn, #30,0.10 ORIGINAL
                   min_dist=md, 
                   metric=tanimoto_dist, 
                   random_state=2).fit_transform(X)

    pd.options.display.max_rows = 150
    embed = pd.DataFrame(umap_X)
    embed = embed.rename(columns={0:"X",1:"Y"})
    
    plot = sns.scatterplot(x='X', y='Y',data=embed,hue = some_df[Hue_fil], palette='tab20',legend=legend)
    
    return(plot)
#     merged = pd.concat([embed, table1], axis=1, sort=False)
#     merged.columns
    
    
def activity_status(row):
    if pd.isna(row['Means_pc_Fib']) or row['Means_pc_Fib'] > 10:
        return('Inactive')
    else:
        return 'Active'
    
def activity_status_Col(row):
    if pd.isna(row['Means_pc_Col']) or row['Means_pc_Col'] > 10:
        return('Inactive')
    else:
        return 'Active'

#Tanimoto distance calculator with alternative for np.nan or np.inf values
def tanimoto_distance(x, y):
    inter = np.dot(x, y)
    union = np.sum(x) + np.sum(y) - inter
    if union == 0:
        return 0.0
    return 1.0 - inter / union


###SIMPLE MATRIXPLOT
def plot_clustering_heatmap(distance_matrix, labels):
    # Create the clustermap
    g = sns.clustermap(distance_matrix, xticklabels=labels, yticklabels=labels, cmap='magma', method='average')

    # Reduce the size of the axis text by accessing the ax_heatmap and adjusting tick labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=2)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=2)
    
    # Reduce the size of the colorbar label text if needed
    g.cax.set_ylabel('Colorbar', fontsize=8)
    
    # Optionally adjust the colorbar tick labels
    g.cax.tick_params(labelsize=8)
    
    plt.title('Clustering Heatmap', fontsize=10)  # Adjust title font size
    plt.show()
    
#CLUSTERING ON HEATMAP
def perform_clustering(distance_matrix, num_clusters):
    # Compute the linkage matrix
    linkage_matrix = linkage(distance_matrix, method='average',)

    # Assign cluster identities
    cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    return cluster_labels, linkage_matrix

# HEATMAP with color labels, no legend
def plot_clustering_heatmap_wLabels(distance_matrix, labels, row_colors, col_colors):
    # Create the clustermap
    g = sns.clustermap(
        distance_matrix, 
        xticklabels=False, 
        yticklabels=False, 
        cmap='magma', 
        method='average', 
        row_colors=row_colors, 
        col_colors=col_colors,
    )

    # Remove x and y axis labels
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # Reduce the size of the colorbar label text if needed
    g.cax.set_ylabel('Colorbar', fontsize=8)

    # Optionally adjust the colorbar tick labels
    g.cax.tick_params(labelsize=8)

    plt.title('Clustering Heatmap', fontsize=10)  # Adjust title font size
    
    # Add a legend for the row colors before showing the plot
#     for cluster, color in cluster_colors.items():
#         g.ax_heatmap.bar(0, 0, color=color, label=f'Cluster {cluster}', linewidth=0)
#     g.ax_heatmap.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=8)
    
#     plt.show()

#LEGEND plotter
def plot_only_legend(cluster_colors):
    fig, ax = plt.subplots(figsize=(5, 2))  # Adjust the figsize as needed
    for cluster, color in cluster_colors.items():
        ax.bar(0, 0, color=color, label=f'Cluster {cluster}', linewidth=0)
    ax.legend(loc="center", ncol=3, bbox_to_anchor=(0.5, 0.5), fontsize=8, frameon=False)
    ax.axis('off')  # Hide the axes
    plt.show()

def ensure_mol(x):
    """
    If x is a SMILES string: convert to Mol;
    if x is already RDKit Mol: return as is;
    otherwise return None.
    """
    if x is None:
        return None
    if isinstance(x, Chem.Mol):
        return x
    try:
        # assume it's SMILES string
        mol = Chem.MolFromSmiles(x)
        return mol
    except Exception:
        return None

def ecfp4_fp(mol, n_bits=2048, radius=2):
    """
    Generate ECFP4 fingerprint (Morgan radius=2) as a numpy array of bits.
    Safe version using the new RDKit MorganGenerator API.
    """
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    try:
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        # Return zero vector if fingerprinting fails
        return np.zeros((n_bits,), dtype=int)


def cleanup_fragment(mol):
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count


def generate_fragments(mol):
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  Extract the largest fragment from each molecule
    flat_frag_list = [uru.get_largest_fragment(x) for x in flat_frag_list]
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number fragments in
    # input molecule
    num_mol_atoms = mol.GetNumAtoms()
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
    # remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df


def find_scaffolds(df_in):
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    # Loop over molecules and generate fragments, fragments for each molecule are returned as a Pandas dataframe
    df_list = []
    for smiles, name, mol in tqdm(df_in[["SMILES", "Name", "mol"]].values):
        tmp_df = generate_fragments(mol).copy()
        tmp_df['Name'] = name
        tmp_df['SMILES'] = smiles
        df_list.append(tmp_df)
    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    # Collect scaffolds
    scaffold_list = []
    for k, v in mol_df.groupby("Scaffold"):
        scaffold_list.append([k, len(v.Name.unique()), v.NumAtoms.values[0]])
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms"])
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    num_df_rows = len(df_in)
    scaffold_df = scaffold_df.query("Count <= @num_df_rows")
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    return mol_df, scaffold_df


def get_molecules_with_scaffold(scaffold, mol_df, activity_df):
    """
    Associate molecules with scaffolds
    :param scaffold: scaffold SMILES
    :param mol_df: dataframe with molecules and scaffolds, returned by find_scaffolds()
    :param activity_df: dataframe with [SMILES, Name, pIC50] columns
    :return: list of core(s) with R-groups labeled, dataframe with [SMILES, Name, pIC50]
    """
    match_df = mol_df.query("Scaffold == @scaffold")
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    rgroup_match, rgroup_miss = RGroupDecompose(scaffold_mol, merge_df.mol, asSmiles=True)
    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        return rgroup_df.Core.unique(), merge_df[["SMILES", "Name", "pIC50"]]
    else:
        return [], merge_df[["SMILES", "Name", "pIC50"]]