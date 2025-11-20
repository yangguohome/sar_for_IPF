import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from czifile import * # for loading lsm-files
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import pandas as pd
import seaborn as sns

import scipy.stats
from matplotlib import pyplot as plt
import math

import anndata as ad
from scipy.sparse import csr_matrix

import scipy.optimize as opt
#import scikit_posthocs
import sys,os
from matplotlib import colors
from matplotlib import cm as cmx
from scipy.stats import linregress
import ast

from Code import Main_v2
from Code import Plate_Plotter
from Code.Plate_Viz import *
from Code.Quality_Control import *
from Code.Image_To_MFI import*

# #########################


def Bulk_QC_Function(bulk_qc_df, metric):
    """
    Set 'Var/PL4 Selection by Slope' per (Cell line, Compound, Norm_Type, Experiment, Replicate).
    Uses the mean Var Hill_Slope to decide:
      - if Var Hill_Slope < 0  → prefer Var (Var=True, PL4=False)
      - else                   → prefer PL4 (PL4=True, Var=False)

    Params
    ----------
    bulk_qc_df : DataFrame
        Must contain columns:
        ['Cell line','Compound name','Norm_Type','Experiment','Replicate',
         'Curve_Type','Hill_Slope', metric] where metric is 'IC50' or 'LD50'.

    Returns
    -------
    DataFrame
        Input df with/updated column 'Var/PL4 Selection by Slope' (bool).
    """
    if metric not in bulk_qc_df.columns:
        raise ValueError(f"Column '{metric}' not found in bulk_qc_df.")

    out = bulk_qc_df.copy()
    col_sel = "Var/PL4 Selection by Slope"

    # optional: initialize column
    if col_sel not in out.columns:
        out[col_sel] = np.nan

    for cell_line in out["Cell line"].dropna().unique():
        cell_df = out[out["Cell line"] == cell_line]
        for cmpd in cell_df["Compound name"].dropna().unique():
            cmp_df = cell_df[cell_df["Compound name"] == cmpd]
            for normtype in cmp_df["Norm_Type"].dropna().unique():
                norm_sdf = cmp_df[cmp_df["Norm_Type"] == normtype]
                for exp in norm_sdf["Experiment"].dropna().unique():
                    exp_df = norm_sdf[norm_sdf["Experiment"] == exp]
                    for rep in exp_df["Replicate"].dropna().unique():
                        rep_df = exp_df[exp_df["Replicate"] == rep]   # 

                        # compute mean slope on Var curves
                        var_hillslope = rep_df.loc[rep_df["Curve_Type"] == "Var", "Hill_Slope"].mean()

                        # build mask to write back
                        m = (
                            (out["Cell line"] == cell_line) &
                            (out["Compound name"] == cmpd) &
                            (out["Norm_Type"] == normtype) &
                            (out["Experiment"] == exp) &
                            (out["Replicate"] == rep)
                        )

                        # decide by slope (fallback to PL4 if slope is NaN)
                        prefer_var = np.isfinite(var_hillslope) and (var_hillslope < 0)

                        if prefer_var:
                            out.loc[m & (out["Curve_Type"] == "Var"), col_sel] = True
                            out.loc[m & (out["Curve_Type"] == "PL4"), col_sel] = False
                        else:
                            out.loc[m & (out["Curve_Type"] == "PL4"), col_sel] = True
                            out.loc[m & (out["Curve_Type"] == "Var"), col_sel] = False

    return bulk_qc_df

def summarize_qic(df, target, # target=IC50/LD50
                qc_col="QC_Score_ZFactor_l",
                top_n=3, long_n=6,
                enforce_slope=True,
                special_compounds=("cmp4","cmp18","Tranilast"),
                special_cell_lines=("MLT018","MLT115")
                ):
    """
    df requires columns:
      'Compound name','Cell line','Curve_Type','Rsq','IC50/LD50','Experiment','By','Replicate',
      'Var/PL4 Selection by Slope', and 'QC_Score_ZFactor_l'
    return dict: shortlist, longlist, Result_List_l, Result_List_Final, MetaSAR_df
    """

    # For each compound–cell line pair: select the top_n/top_long Var curves ranked by plate quality (qc_col), 
    # plus the top_n/top_long curves overall ranked by R² (Rsq); 
    # then merge the two sets and remove duplicates
    local_loop_comb, local_loop_comb_longer = [], []
    for cmpd in df["Compound name"].dropna().unique():
        sub_cmp = df[df["Compound name"] == cmpd]
        for cl in sub_cmp["Cell line"].dropna().unique():
            loop_df = sub_cmp[sub_cmp["Cell line"] == cl]
            if loop_df.empty: continue
            loop_df_var = loop_df[loop_df["Curve_Type"] == "Var"].sort_values(qc_col, ascending=True)
            best_var, best_var_longer = loop_df_var.head(top_n), loop_df_var.head(long_n)
            best_rsq, best_rsq_longer   = loop_df.sort_values("Rsq",ascending=False).head(top_n), loop_df.sort_values("Rsq",ascending=False).head(long_n)
            local_loop_comb.append(pd.concat([best_var, best_rsq]))
            local_loop_comb_longer.append(pd.concat([best_var_longer, best_rsq_longer]))

    shortlist   = pd.concat(local_loop_comb).drop_duplicates(ignore_index=True) if local_loop_comb else pd.DataFrame()
    longlist    = pd.concat(local_loop_comb_longer).drop_duplicates(ignore_index=True) if local_loop_comb_longer else pd.DataFrame()

    # Aggregate at the cell-line level from the shortlist: require BySlope = True; 
    # if there are ≥ 3 Var curves, use Var only; then rank by R² (Rsq) and take the top_n
    results_l = []
    if not shortlist.empty:
        for cmpd in shortlist["Compound name"].dropna().unique():
            sub_cmp = shortlist[shortlist["Compound name"] == cmpd]
            for cl in sub_cmp["Cell line"].dropna().unique():
                g = sub_cmp[sub_cmp["Cell line"] == cl].copy()
                if enforce_slope and "Var/PL4 Selection by Slope" in g.columns:
                    g = g[g["Var/PL4 Selection by Slope"] == True]
                if g.empty: continue
                if (g["Curve_Type"].eq("Var")).sum() > 2:
                    g = g[g["Curve_Type"] == "Var"]
                g = g.sort_values("Rsq", ascending=False).head(top_n)
                if g.empty: continue
                results_l.append({
                    "Compound name": cmpd,
                    "Cell line": cl,
                    f"Mean_{target}": np.round(g[target].mean(), 2),
                    f"Stdev_{target}": np.round(g[target].std(), 2),
                    "Rsq": np.round(g["Rsq"].mean(), 2),
                    "Meta_Data": [g["Experiment"].tolist(), g["By"].tolist(), g["Replicate"].tolist(), g["Curve_Type"].tolist()],
                    "Ns": int(len(g))
                })

    Result_List_l = pd.DataFrame(results_l).dropna(subset=[f"Mean_{target}"]).reset_index(drop=True)

    # Compound-level aggregation: for special compounds use only the specified cell lines; 
    # otherwise, select the top_n by R² (Rsq)
    Final_List, Result_List_Final = [], pd.DataFrame()
    if not Result_List_l.empty:
        for cmps in Result_List_l["Compound name"].unique():
            loop_df = Result_List_l[Result_List_l["Compound name"] == cmps].copy()
            if cmps in special_compounds:
                loop_df = loop_df[loop_df["Cell line"].isin(special_cell_lines)]
            else:
                loop_df = loop_df.sort_values("Rsq", ascending=False).head(top_n)
            if loop_df.empty: continue

            mean_m, stdev_m = loop_df[f"Mean_{target}"].mean(), loop_df[f"Mean_{target}"].std()

            mean_Rsq, Ns = loop_df["Rsq"].mean(), int(loop_df["Ns"].sum())
            
            bestfit_m = loop_df.iloc[0][f"Mean_{target}"]
            bestfit_m_stdev = loop_df.iloc[0][f"Stdev_{target}"]
            bestfit_m_CL = loop_df.iloc[0]["Cell line"]

            Final_List.append({
                "Compound name": cmps,
                target: mean_m,
                "Stdev": stdev_m,
                "Mean_Rsq": mean_Rsq,
                "Ns": Ns,
                f"Bestfit_{target}": bestfit_m,
                f"Bestfit_{target}_stdev": bestfit_m_stdev,
                f"Bestfit_{target}_CL": bestfit_m_CL
            })
            Result_List_Final = pd.concat([Result_List_Final, loop_df], ignore_index=True)

    MetaSAR_df = pd.DataFrame(Final_List).sort_values(target).reset_index(drop=True)

    return {
        "shortlist": shortlist,
        "longlist": longlist,
        "Result_List_l": Result_List_l,          
        "Result_List_Final": Result_List_Final,  
        "MetaSAR_df": MetaSAR_df                 
    }


def summarize_qic_metric(df, metric='IC50',
                         top_n=3, long_n=6,
                         enforce_slope=True,
                         internal_controls =("cmp4","cmp18","Tranilast"),
                         special_cell_lines=("MLT018","MLT115"),
                         tox_from_col="LD50",
                         tox_threshold=60):
    # shortlist (top 3 Rsq candidates) / longlist (top 6 Rsq candidates)
    local_loop_comb, local_loop_comb_longer = [], []
    for cmpd in df["Compound name"].dropna().unique():
        sub_cmp = df[df["Compound name"] == cmpd]
        for cl in sub_cmp["Cell line"].dropna().unique():
            loop_df = sub_cmp[sub_cmp["Cell line"] == cl]
            if loop_df.empty: continue
            # Pick top_n / long_n based on Rsq
            best_rsq        = loop_df.sort_values("Rsq", ascending=False).head(top_n)
            best_rsq_longer = loop_df.sort_values("Rsq", ascending=False).head(long_n)
            local_loop_comb.append(best_rsq)
            local_loop_comb_longer.append(best_rsq_longer)

    shortlist = pd.concat(local_loop_comb).drop_duplicates(ignore_index=True) if local_loop_comb else pd.DataFrame()
    longlist  = pd.concat(local_loop_comb_longer).drop_duplicates(ignore_index=True) if local_loop_comb_longer else pd.DataFrame()

    # cell-line
    results_l = []
    for cmpd in shortlist["Compound name"].unique():
        sub_cmp = shortlist[shortlist["Compound name"] == cmpd]
        for cl in sub_cmp["Cell line"].unique():
            g = sub_cmp[sub_cmp["Cell line"] == cl].copy()
            if enforce_slope and "Var/PL4 Selection by Slope" in g.columns:
                g = g[g["Var/PL4 Selection by Slope"] == True]
            if g.empty: continue
            if (g["Curve_Type"].eq("Var")).sum() > 2:
                g = g[g["Curve_Type"] == "Var"]
            g = g.sort_values("Rsq", ascending=False).head(top_n)
            if g.empty: continue
            results_l.append({
                "Compound name": cmpd,
                "Cell line": cl,
                f"Mean_{metric}": np.round(g[metric].mean(), 2),
                f"Stdev_{metric}": np.round(g[metric].std(), 2),
                "Rsq": np.round(g["Rsq"].mean(), 2),
                "Meta_Data": [g["Experiment"].tolist(), g["By"].tolist(), g["Replicate"].tolist(), g["Curve_Type"].tolist()],
                "Ns": int(len(g))
            })
    Result_List_l = pd.DataFrame(results_l).dropna(subset=[f"Mean_{metric}"]).reset_index(drop=True)

    # compound 
    Final_List, Result_List_Final = [], pd.DataFrame()
    for cmps in Result_List_l["Compound name"].unique():
        loop_df = Result_List_l[Result_List_l["Compound name"] == cmps].copy()
        if cmps in internal_controls:
            loop_df = loop_df[loop_df["Cell line"].isin(special_cell_lines)]
        else:
            loop_df = loop_df.sort_values("Rsq", ascending=False).head(top_n)
        if loop_df.empty: continue

        mean_m, stdev_m = loop_df[f"Mean_{metric}"].mean(), loop_df[f"Mean_{metric}"].std()
        mean_Rsq, Ns = loop_df["Rsq"].mean(), int(loop_df["Ns"].sum())
        bestfit_m = loop_df.iloc[0][f"Mean_{metric}"]
        bestfit_m_stdev = loop_df.iloc[0][f"Stdev_{metric}"]
        bestfit_m_CL = loop_df.iloc[0]["Cell line"]

        Final_List.append({
            "Compound name": cmps,
            metric: mean_m,
            "Stdev": stdev_m,
            "Mean_Rsq": mean_Rsq,
            "Ns": Ns,
            f"Bestfit_{metric}": bestfit_m,
            f"Bestfit_{metric}_stdev": bestfit_m_stdev,
            f"Bestfit_{metric}_CL": bestfit_m_CL
        })
        Result_List_Final = pd.concat([Result_List_Final, loop_df], ignore_index=True)

    MetaSAR_df = pd.DataFrame(Final_List).sort_values(metric).reset_index(drop=True)

    # LD50 toxicity label
    if tox_from_col in df.columns:
        tox_pair = (df.groupby(["Compound name","Cell line"], as_index=False)[tox_from_col]
                      .mean().rename(columns={tox_from_col: f"Mean_{tox_from_col}"}))
        tox_pair["LD50_Toxic"] = tox_pair[f"Mean_{tox_from_col}"] < float(tox_threshold)

        if not shortlist.empty:
            shortlist = shortlist.merge(tox_pair[["Compound name","Cell line","LD50_Toxic"]],
                                        on=["Compound name","Cell line"], how="left")
            shortlist["LD50_Toxic"] = shortlist["LD50_Toxic"].fillna(False)
        if not longlist.empty:
            longlist = longlist.merge(tox_pair[["Compound name","Cell line","LD50_Toxic"]],
                                      on=["Compound name","Cell line"], how="left")
            longlist["LD50_Toxic"] = longlist["LD50_Toxic"].fillna(False)
        if not Result_List_l.empty:
            Result_List_l = Result_List_l.merge(tox_pair[["Compound name","Cell line","LD50_Toxic"]],
                                                on=["Compound name","Cell line"], how="left")
            Result_List_l["LD50_Toxic"] = Result_List_l["LD50_Toxic"].fillna(False)
        if not Result_List_Final.empty:
            Result_List_Final = Result_List_Final.merge(tox_pair[["Compound name","Cell line","LD50_Toxic"]],
                                                        on=["Compound name","Cell line"], how="left")
            Result_List_Final["LD50_Toxic"] = Result_List_Final["LD50_Toxic"].fillna(False)

        
        if not MetaSAR_df.empty:
            any_tox = (tox_pair.groupby("Compound name")["LD50_Toxic"]
                              .any().rename("Any_LD50_Toxic").reset_index())
            MetaSAR_df = MetaSAR_df.merge(any_tox, on="Compound name", how="left")
            MetaSAR_df["Any_LD50_Toxic"] = MetaSAR_df["Any_LD50_Toxic"].fillna(False)

    return {
        "shortlist": shortlist,
        "longlist": longlist,
        "Result_List_l": Result_List_l,
        "Result_List_Final": Result_List_Final,
        "MetaSAR_df": MetaSAR_df,
        "tox_cutoff": {tox_from_col: tox_threshold}
    }


def label_toxicity_from_nuclei(
    QIC_Nuc_post3,               
    QIC_Col_post3,               
    QIC_Fib_post3,               
    fib_out,                     
    label_col="Nuc_Toxicity",    
    ld50_threshold=60.0          
):
    
    for df in [QIC_Nuc_post3,
               QIC_Col_post3,
               QIC_Fib_post3]:
        need_cols = ['Compound name','Cell line']
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{df} lacks: {miss}")

    if 'LD50' not in QIC_Nuc_post3.columns:
        raise ValueError("QIC_Nuc_postQC lacks 'LD50' column.")

    
    nuc = QIC_Nuc_post3.copy()
    nuc['LD50'] = pd.to_numeric(nuc['LD50'], errors='coerce')
    nuc = nuc.dropna(subset=['LD50'])

    
    nuc_tox = nuc.loc[nuc['LD50'] < float(ld50_threshold),
                      ['Compound name','Cell line']].drop_duplicates()

    
    tested_pairs = pd.concat([
        QIC_Col_post3[['Compound name','Cell line']],
        QIC_Fib_post3[['Compound name','Cell line']]
    ], ignore_index=True).drop_duplicates()

    labels = nuc_tox.merge(tested_pairs, on=['Compound name','Cell line'], how='inner')
    labels[label_col] = True
    labels = labels[['Compound name','Cell line', label_col]]

    
    cutoffs = pd.DataFrame({
        'Cell line': sorted(QIC_Nuc_post3['Cell line'].dropna().astype(str).unique()),
        'cutoff': float(ld50_threshold)
    })

    
    updated = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in fib_out.items()}
    for key in ["shortlist","longlist","Result_List_l","Result_List_Final"]:
        if key in updated and isinstance(updated[key], pd.DataFrame) and not updated[key].empty:
            updated[key] = (updated[key]
                            .merge(labels, on=['Compound name','Cell line'], how='left'))
            updated[key][label_col] = updated[key][label_col].fillna(False)

    
    if "MetaSAR_df" in updated and isinstance(updated["MetaSAR_df"], pd.DataFrame) and not updated["MetaSAR_df"].empty:
        any_tox = (labels.groupby("Compound name")[label_col]
                         .any()
                         .rename(f"LD50<{ld50_threshold}uM")
                         .reset_index())
        updated["MetaSAR_df"] = updated["MetaSAR_df"].merge(any_tox, on="Compound name", how="left")
        updated["MetaSAR_df"][f"LD50<{ld50_threshold}uM"] = updated["MetaSAR_df"][f"LD50<{ld50_threshold}uM"].fillna(False)

    return updated, cutoffs, labels



import numpy as np
import pandas as pd

def label_toxicity_from_ti(
    QIC_Nuc_post3,
    QIC_Col_post3,
    QIC_Fib_post3,
    fib_out,
    target="IC50",        # must match summarize_qic(..., target=...)
    ti_col="TI_log10",    # name of column storing log10(LD50/IC50)
    marker_col="TI_Marker",
    ti_toxic=0.3,         # TI_log10 < 0.3   -> 'cross'
    ti_safe=1.0           # 0.3–1           -> 'triangle';  >=1 -> 'circle'
):
    """
    Label toxicity based on therapeutic window TI_log10 = log10(LD50/IC50).

    LD50 comes from nuclei data (QIC_Nuc_post3),
    IC50 comes from fib_out["Result_List_l"] (Mean_<target>).

    IMPORTANT change:
        - We now treat the IC50 table (Result_List_l) as the reference:
          all compound–cell line pairs that have IC50 will be kept.
        - For pairs with missing LD50, LD50 is set to np.inf,
          so TI_log10 -> +inf and they are classified as 'circle' (safe).

    Categories:
        TI_log10 < ti_toxic        -> 'cross'
        ti_toxic ≤ TI_log10 < 1.0  -> 'triangle'
        TI_log10 ≥ ti_safe         -> 'circle'
    """

    # --- Basic column checks ---
    for df in [QIC_Nuc_post3, QIC_Col_post3, QIC_Fib_post3]:
        required = ['Compound name', 'Cell line']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe {df} lacks required columns: {missing}")

    if 'LD50' not in QIC_Nuc_post3.columns:
        raise ValueError("QIC_Nuc_post3 must contain an 'LD50' column.")

    # --- 1) Summarize LD50 from nuclei (per compound × cell line) ---
    nuc = QIC_Nuc_post3.copy()
    nuc['LD50'] = pd.to_numeric(nuc['LD50'], errors='coerce')
    nuc = nuc.dropna(subset=['LD50'])

    nuc_summary = (
        nuc.groupby(['Compound name', 'Cell line'])['LD50']
           .median()
           .reset_index()
    )

    # --- 2) Extract IC50 from fib_out["Result_List_l"] ---
    if "Result_List_l" not in fib_out or not isinstance(fib_out["Result_List_l"], pd.DataFrame):
        raise ValueError("fib_out must contain a non-empty 'Result_List_l' DataFrame.")

    rl = fib_out["Result_List_l"].copy()
    ic50_col = f"Mean_{target}"

    if ic50_col not in rl.columns:
        raise ValueError(f"'Result_List_l' lacks required column '{ic50_col}' to extract IC50.")

    rl[ic50_col] = pd.to_numeric(rl[ic50_col], errors='coerce')

    # Average IC50 per compound × cell line if multiple rows exist
    ic50_df = (
        rl.groupby(['Compound name', 'Cell line'])[ic50_col]
          .mean()
          .reset_index()
          .rename(columns={ic50_col: "IC50"})
    )
    ic50_df = ic50_df.dropna(subset=['IC50'])

    # --- 3) LEFT-MERGE using IC50 table as reference ---
    # All IC50 pairs are kept; LD50 can be missing (NaN) after the merge.
    ti_table = ic50_df.merge(
        nuc_summary,
        on=['Compound name', 'Cell line'],
        how='left'          # <-- key change: IC50 as reference
    )

    # Ensure numeric LD50 and replace missing LD50 with +inf
    ti_table['LD50'] = pd.to_numeric(ti_table['LD50'], errors='coerce')

    # Only enforce IC50 > 0 (for log10); LD50 can be +inf
    ti_table = ti_table[ti_table['IC50'] > 0]

    # Set missing LD50 (no nuclei LD50 available) to +inf
    ld50_missing = ti_table['LD50'].isna()
    ti_table.loc[ld50_missing, 'LD50'] = np.inf

    if ti_table.empty:
        updated = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in fib_out.items()}
        return updated, ti_table, pd.DataFrame(columns=['Compound name', 'Cell line', ti_col, marker_col])

    # --- 4) Compute TI_log10 = log10(LD50/IC50) ---
    ti_table[ti_col] = np.log10(ti_table['LD50']) - np.log10(ti_table['IC50'])

    # --- 5) Assign markers based on TI thresholds ---
    def _assign_marker(x):
        if x < ti_toxic:
            return "cross"
        elif x < ti_safe:
            return "triangle"
        else:
            return "circle"

    ti_table[marker_col] = ti_table[ti_col].apply(_assign_marker)

    # --- 6) Restrict labels to pairs actually tested in Col/Fib (for plotting/consistency) ---
    tested_pairs = pd.concat([
        QIC_Col_post3[['Compound name', 'Cell line']],
        QIC_Fib_post3[['Compound name', 'Cell line']]
    ], ignore_index=True).drop_duplicates()

    labels = tested_pairs.merge(
        ti_table[['Compound name', 'Cell line', ti_col, marker_col]],
        on=['Compound name', 'Cell line'],
        how='left'
    )

    # --- 7) Merge TI information back into fib_out tables ---
    updated = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in fib_out.items()}

    for key in ["shortlist", "longlist", "Result_List_l", "Result_List_Final"]:
        if key in updated and isinstance(updated[key], pd.DataFrame) and not updated[key].empty:
            updated[key] = updated[key].merge(
                labels[['Compound name', 'Cell line', ti_col, marker_col]],
                on=['Compound name', 'Cell line'],
                how='left'
            )

    # --- 8) Compound-level TI summary for MetaSAR_df ---
    if "MetaSAR_df" in updated and isinstance(updated["MetaSAR_df"], pd.DataFrame) and not updated["MetaSAR_df"].empty:

        comp_ti = (
            ti_table.groupby("Compound name")[ti_col]
                    .agg(['min', 'mean', 'max'])
                    .rename(columns={
                        'min': f"{ti_col}_min",
                        'mean': f"{ti_col}_mean",
                        'max': f"{ti_col}_max"
                    })
                    .reset_index()
        )

        def _assign_comp_marker(x):
            if x < ti_toxic:
                return "cross"
            elif x < ti_safe:
                return "triangle"
            else:
                return "circle"

        comp_ti[f"{marker_col}_compound"] = comp_ti[f"{ti_col}_min"].apply(_assign_comp_marker)

        updated["MetaSAR_df"] = updated["MetaSAR_df"].merge(
            comp_ti,
            on="Compound name",
            how="left"
        )

    return updated, ti_table, labels




def curve_fetching_function(
    data_file,
    alternate_df=None,
    COI=None,                              # if None = batch mode（all cmp in one Cell_line）
    Cell_line="MLT018",
    display_plot=True,
    Data_dir="Data/Database_Files/AllData/datasummary_AllData.xlsx",
    Nuclei_dir="Data/Database_Files/AllData/AllData_NucleiDB.xlsx",
    Plate_dir="Data/Database_Files/AllData/AllData_Plates.xlsx",
    stain_w_nuc="MFI/Nuc",  # or "MFI"
    use_alternate=False
):
    """
    batch/single compound: read AllData(plate/nuclei/data) → select stain/norm → table & GraphPad table → curve fitting
    return:
      plot_df_all:      table (all cmps combined), including ['dose','response','series','compound'], series=Experiment_Rep
      graphpad_wide:    dict[compound] = wide_table
      curvefit_results: curvefitting table, including ['compound','series','IC50','SE','Hill_Slope','Rsq','MSE']
    """
    if use_alternate==True:
        data_file = alternate_df.copy()
        
    data_stain = data_file["Stain"].unique()[0]

    if COI is None:
        compounds_to_run = sorted(data_file.loc[data_file["Cell line"] == Cell_line, "Compound name"].unique().tolist())
        if not compounds_to_run:
            raise ValueError(f"No compounds found for Cell line = {Cell_line}.")
    else:
        compounds_to_run = [COI]
    
    meta_data_str = data_file[data_file["Compound name"] == compounds_to_run][data_file["Cell line"] == Cell_line]
    meta_data_str = meta_data_str.iloc[0]["Meta_Data"]
    meta_data_list = ast.literal_eval(meta_data_str)

    exp_to_fetch = meta_data_list[0]
    exp_by= meta_data_list[1]
    exp_replicates = meta_data_list[2]
    exp_curve_types = meta_data_list[3]
    other_information = meta_data_list[4:]
        
    empty_df = pd.DataFrame(columns=['dose', 'compound', 'response', 'Exp'])
    display(exp_to_fetch)
    
    for i in range(len(exp_to_fetch)):
        if exp_by[i] != "MW":
            
            Experiment = exp_to_fetch[i]
            By = exp_by[i]
            Replicate=exp_replicates[i]

            data_df = pd.read_excel(Data_dir,index_col=0)
            data_df = data_df[(data_df["Experiment"] == Experiment) 
                              & (data_df["By"] == By)]

            plate_df = pd.read_excel(Plate_dir,index_col=0)
            plate_df=plate_df[(plate_df["Experiment"] == Experiment) 
                              & (plate_df["By"] == By)]

            nuclei_df = pd.read_excel(Nuclei_dir,index_col=0)
            nuclei_df=nuclei_df[(nuclei_df["Experiment"] == Experiment) 
                              & (nuclei_df["By"] == By)]            

            Main_Object= Main_v2.Main(dataframe_dir     = None, 
                                          plate_dir     = None,
                                          nuclei_dir    = None,
                                          data_df       =  data_df,
                                          plate_df      =  plate_df,
                                          nuclei_df     = nuclei_df,
                                          Experiment    = Experiment, 
                                                By      = By)
            if data_stain  == "FBLN1":
                pre_df = Main_Object.normalized_dataframe("Fibulin")
                stain = "Fibulin"
            else:
                pre_df = Main_Object.normalized_dataframe("ECM")
                stain = "ECM"

            # use isin() to process multiple cmps rather than ==
            processed_vals,names_of_cols=select_stain(pre_df[pre_df["Compound name"].isin(compounds_to_run)], 
                             norm_type="HighLow", 
                             stain_type_here=stain_w_nuc, 
                             stain=stain,thresholding=True)

            processed_vals=  processed_vals[processed_vals["Replicate"]==Replicate]
            processed_vals = processed_vals[processed_vals["Cell line"]==Cell_line]

            saved_df = processed_vals[["Compound Concentration"]+names_of_cols]
            exp_naming =Experiment + "_"+str(Replicate)
            saved_df.rename(columns={"Compound Concentration": "Compound Concentration",
                                      names_of_cols[0]:exp_naming+"_1",
                                      names_of_cols[1]:exp_naming+"_2",
                                      names_of_cols[2]:exp_naming+"_3"
                                    }, inplace=True)

            if i == 0 :
                output_df = saved_df["Compound Concentration"]

            output_df = pd.merge(output_df,saved_df, on = "Compound Concentration")
            ploting_df = preprocessor_function(processed_vals,names_of_cols)
            ploting_df["Exp"]= exp_naming

            empty_df = pd.concat([empty_df,ploting_df])

    empty_df.drop("compound",axis=1,inplace=True)
    empty_df.rename(columns={'Exp': 'compound'}, inplace=True)

    if use_alternate==True:
        data_file = alternate_df.copy()
        
    data_stain = data_file["Stain"].unique()[0]
    
    meta_data_str = data_file[data_file["Compound name"] == compounds_to_run][data_file["Cell line"] == Cell_line]
    meta_data_str = meta_data_str.iloc[0]["Meta_Data"]
    meta_data_list = ast.literal_eval(meta_data_str)

    exp_to_fetch = meta_data_list[0]
    exp_by= meta_data_list[1]
    exp_replicates = meta_data_list[2]
    exp_curve_types = meta_data_list[3]
    other_information = meta_data_list[4:]
    
    if "MW" in exp_by:
        display("Caution: MW experiments are not part of the plotting databse.")
        
    empty_df = pd.DataFrame(columns=['dose', 'compound', 'response', 'Exp'])
    display(exp_to_fetch)
    
    for i in range(len(exp_to_fetch)):
        if exp_by[i] != "MW":
            
            Experiment = exp_to_fetch[i]
            By = exp_by[i]
            Replicate=exp_replicates[i]

            data_df = pd.read_excel(Data_dir,index_col=0)
            data_df = data_df[(data_df["Experiment"] == Experiment) 
                              & (data_df["By"] == By)]

            plate_df = pd.read_excel(Plate_dir,index_col=0)
            plate_df=plate_df[(plate_df["Experiment"] == Experiment) 
                              & (plate_df["By"] == By)]

            nuclei_df = pd.read_excel(Nuclei_dir,index_col=0)
            nuclei_df=nuclei_df[(nuclei_df["Experiment"] == Experiment) 
                              & (nuclei_df["By"] == By)]            

            Main_Object= Main_v2.Main(dataframe_dir     = None, 
                                          plate_dir     = None,
                                          nuclei_dir    = None,
                                          data_df       =  data_df,
                                          plate_df      =  plate_df,
                                          nuclei_df     = nuclei_df,
                                          Experiment    = Experiment, 
                                                By      = By)
            if data_stain  == "FBLN1":
                pre_df = Main_Object.normalized_dataframe("Fibulin")
                stain = "Fibulin"
            else:
                pre_df = Main_Object.normalized_dataframe("ECM")
                stain = "ECM"

            
            processed_vals,names_of_cols=select_stain(pre_df[pre_df["Compound name"]==compounds_to_run], 
                             norm_type="HighLow", 
                             stain_type_here=stain_w_nuc, 
                             stain=stain,thresholding=True)

            processed_vals=  processed_vals[processed_vals["Replicate"]==Replicate]
            processed_vals = processed_vals[processed_vals["Cell line"]==Cell_line]

            saved_df = processed_vals[["Compound Concentration"]+names_of_cols]
            exp_naming =Experiment + "_"+str(Replicate)
            saved_df.rename(columns={"Compound Concentration": "Compound Concentration",
                                      names_of_cols[0]:exp_naming+"_1",
                                      names_of_cols[1]:exp_naming+"_2",
                                      names_of_cols[2]:exp_naming+"_3"
                                    }, inplace=True)

            if i == 0 :
                output_df = saved_df["Compound Concentration"]

            output_df = pd.merge(output_df,saved_df, on = "Compound Concentration")
            ploting_df = preprocessor_function(processed_vals,names_of_cols)
            ploting_df["Exp"]= exp_naming

            empty_df = pd.concat([empty_df,ploting_df])

    empty_df.drop("compound",axis=1,inplace=True)
    empty_df.rename(columns={'Exp': 'compound'}, inplace=True)

    if display_plot == True:

        bar = []
        j=0
        fig,axes= plt.subplots()
        for compound in empty_df.compound.unique():
            plotter = empty_df[empty_df["compound"]==compound]
            info = IC50_generator(exp_curve_types[j],plotter,
                           compounds_to_run,display_plot=True,ax=axes)
            bar.append(info)
            j=j=1
    columns = ['IC50', 'SE', 'Hill_Slope', 'Rsq', 'MSE']
    curvefit_result_df = pd.DataFrame(bar, columns=columns)
    
    return(empty_df, #For Plotting
            output_df,curvefit_result_df) #For Graphpad

