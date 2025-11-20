import warnings
warnings.filterwarnings('ignore')
import traceback
import matplotlib.pyplot as plt
from czifile import * # for loading lsm-files
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import pandas as pd
import seaborn as sns

import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math
import anndata as ad
from scipy.sparse import csr_matrix
import scipy.optimize as opt

import sys,os
from matplotlib import colors
from matplotlib import cm as cmx
from tqdm import tqdm 

from Code import Main_v2
from Code import Plate_Plotter
from Code.Plate_Viz import *
from Code.Quality_Control import *
from Code.Image_To_MFI import*
from Code.Plate_Plotter import *

import os
import pandas as pd

def Plate_Bulk_Analysis(
    dataframe_dir=None,           # root dir containing multiple "*MFI values.xlsx"
    nuclei_dir=None,              # root dir containing multiple "*Nuclei.xlsx"
    plate_dir=None,               # dir containing one plate map
    dataframe_df=None,            # directly input concat "*MFI values.xlsx"
    nuclei_df=None,               # directly input concat "*Nuclei.xlsx"
    plate_df=None,                # directly input plate map
    Experiment=None,              # None/All/list or just name one 'Exp_i'
    By=None,
    norm_type="HighLow",
    metric="MFI/Nuc",
    models=("Var", "PL4"),
    ylim=(0, 110),
    save_values=True,
    display_plot=False,
):
    """
    Read/use data excels -> split by Experiment -> per-Experiment run:
    QC -> normalization -> Fib/ECM (Var/PL4) + Nuclei (Var/PL4) -> merge

    Returns
    -------
    result_df : concatenated per-experiment curve results
    QC_final  : concatenated per-experiment QC (Z')
    QIC       : per-experiment toxicity flags (if available)
    """
    if By is None:
        raise ValueError("Specify the name initials of experimenter.")

    # ---------- read excel under root dir ----------
    def _read_dirs(root_dir, substring):
        """Go through every files under root dir and vertically concat"""
        if root_dir is None:
            raise ValueError(f"{root_dir} does not exist.")
        hits = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".xlsx") and substring in fname:
                    hits.append(pd.read_excel(os.path.join(dirpath, fname)))
        if not hits:
            raise ValueError(f"No excels exist under {root_dir}.")
        return pd.concat(hits, ignore_index=True)


    # ----- Input handling: If all 3 dfs are provided, use them; otherwise, try to read from dir. -----
    if (dataframe_df is not None) and (plate_df is not None) and (nuclei_df is not None):
        data_df  = dataframe_df
        plate_df = plate_df
        nuclei_df = nuclei_df
    else:
        data_df  = _read_dirs(dataframe_dir, "MFI values")
        nuclei_df = _read_dirs(nuclei_dir,   "Nuclei")
        plate_df  = pd.read_excel(plate_dir)

        if data_df is None or plate_df is None or nuclei_df is None:
            raise ValueError(
                "Need either all three DataFrames (dataframe_df, plate_df, nuclei_df) "
                "OR three readable dirs/files (dataframe_dir, plate_dir, nuclei_dir)."
            )
    
    def _unique_exps_from_data(data_df, nuclei_df):
        """Collet 'Experiment index' values from data_df and nuclei_df."""
        exps = set(data_df["Experiment"].unique()) | set(nuclei_df["Experiment"].unique())
        exps = sorted(exps, key=lambda s: int(s.split("_")[1]))
        return exps

    def _subset_by_exp(df, exp):
        """Separate df by 'Experiment Index'. """
        return df[df["Experiment"] == exp].copy()
    
    def build_plate_results(Fibulin_df, Collagen_df):
        ucls = Fibulin_df["Cell line"].dropna().unique()
        res_all = []

        if len(ucls) == 1:
            cl = ucls[0]
            splits = [
                {
                    "cell_line": cl, "replicate": 1,
                    "fib": Fibulin_df[Fibulin_df["Replicate"] == 1],
                    "col": Collagen_df[Collagen_df["Replicate"] == 1],
                },
                {
                    "cell_line": cl, "replicate": 2,
                    "fib": Fibulin_df[Fibulin_df["Replicate"] == 2],
                    "col": Collagen_df[Collagen_df["Replicate"] == 2],
                },
            ]
        elif len(ucls) == 2:
            cl1, cl2 = list(ucls)
            splits = [
                {
                    "cell_line": cl1, "replicate": 1,
                    "fib": Fibulin_df[Fibulin_df["Cell line"] == cl1],
                    "col": Collagen_df[Collagen_df["Cell line"] == cl1],
                },
                {
                    "cell_line": cl2, "replicate": 1,
                    "fib": Fibulin_df[Fibulin_df["Cell line"] == cl2],
                    "col": Collagen_df[Collagen_df["Cell line"] == cl2],
                },
            ]
        else:
            raise ValueError(f"Expect 1 or 2 unique cell lines, got {len(ucls)}")

        for s in splits:
            fib_sub, col_sub = s["fib"], s["col"]
            if fib_sub.empty or col_sub.empty:
                continue
            for model in models:
                out = plate_curve_generator(
                    fib_sub, col_sub,
                    model,             # "Var" or "PL4"
                    norm_type,         # e.g. "HighLow"
                    metric,            # e.g. "MFI/Nuc"
                    ylim=ylim,
                    save_values=save_values,
                    display_plot=display_plot
                ).copy()
                out["Cell line"]  = s["cell_line"]
                out["Replicate"]  = s["replicate"]
                out["Curve_Type"] = model
                out["Readout"]    = metric
                res_all.append(out)

        return pd.concat(res_all, ignore_index=True) if res_all else pd.DataFrame()
    
    def build_nuclei_results(Nuc_df):
        res = []
        for model in models:
            out = nuclei_curves(
                Nuc_df,
                curve=model,
                norm_type=norm_type,
                col_type="Nuclei_HighLow",
                save_values=save_values,
                display_plot=display_plot,
            )
            df = out if not display_plot else out[2]  # (fig, axes, df)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["Curve_Type"] = model
            df["Readout"]    = "Nuclei"
            res.append(df)
        return pd.concat(res, ignore_index=True) if res else pd.DataFrame()


    if Experiment is None or str(Experiment).lower() == "all":
        experiments = _unique_exps_from_data(data_df, nuclei_df)
    elif isinstance(Experiment, (list, tuple, set)):
        experiments = list(Experiment)
    else:
        experiments = [Experiment]
    
    ECM_all = []
    Nuclei_all = []
    qc_all = []
    qic_all = []
    skip_logs = []

    for exp in experiments:
        try: 
            data_sub   = _subset_by_exp(data_df,   exp)
            nuclei_sub = _subset_by_exp(nuclei_df, exp)

            # ---------- Initialize Main Object ---------------------------
            Main_Object = Main_v2.Main(
                dataframe_dir=None,
                plate_dir=None,
                nuclei_dir=None,
                data_df=data_sub,
                plate_df=plate_df,
                nuclei_df=nuclei_sub,
                Experiment=exp,
                By=By
            )

            # ---------- QC for Fibulin, Collagen, and Nuclei ----------
            _, Fib_processed_plate = Main_Object.get_plate("Fibulin", outlier_detection=True)
            QC_Fib = QC_Zfactor(Fib_processed_plate, exp, By)

            _, Col_processed_plate = Main_Object.get_plate("ECM", outlier_detection=True)
            QC_Col = QC_Zfactor(Col_processed_plate, exp, By)

            _, Nuc_processed_plate = Main_Object.get_plate("Nuclei", outlier_detection=True)
            QC_Nuc = QC_Zfactor(Nuc_processed_plate, exp, By)

            # ---------- ECM & Nuclei Normalization ----------
            Fibulin_df  = Main_Object.normalized_dataframe("Fibulin")
            Collagen_df = Main_Object.normalized_dataframe("ECM")

            Nuclei_df = HighLow_Nuc_Normalization(plate=Nuc_processed_plate)
            Nuc_df    = Main_Object.integration_call(Nuclei_df, "Nuclei", norm_type)

            # ---------- Build and collect Fib/ECM/Nuclei results ----------
            ECM_results    = build_plate_results(Fibulin_df, Collagen_df)
            nuclei_results = build_nuclei_results(Nuc_df)

            ECM_results = ECM_results.copy()
            ECM_results["Experiment"] = exp
            ECM_results["By"]         = By
            ECM_results["Norm_Type"]  = norm_type
            ECM_all.append(ECM_results)

            nuclei_results = nuclei_results.copy()
            nuclei_results["Experiment"] = exp
            nuclei_results["By"]         = By
            nuclei_results["Norm_Type"]  = norm_type
            Nuclei_all.append(nuclei_results)

            # ---------- Build and collect QC results ----------
            for plate_name, qc_df in [("Fibulin", QC_Fib), ("ECM", QC_Col), ("Nuclei", QC_Nuc)]:
                df = qc_df.copy() if isinstance(qc_df, pd.DataFrame) else pd.DataFrame()
                df["Plate"] = plate_name
                qc_all.append(df)

            # ---------- Build and collect Cytotoxicity QIC results ----------
            try:
                CV_df = Cell_Viability_analysis(Nuc_df)
                if CV_df is not None and not CV_df.empty and "status" in CV_df.columns and "Compound name" in CV_df.columns:
                    comp_union = pd.unique(pd.concat([
                        Fibulin_df.get("Compound name", pd.Series(dtype=object)),
                        Collagen_df.get("Compound name", pd.Series(dtype=object)),
                        Nuc_df.get("Compound name", pd.Series(dtype=object)),
                    ], ignore_index=True).dropna())
                    QIC_exp = pd.DataFrame({"Compound name": comp_union})
                    QIC_exp["Toxicity"] = False
                    tox_names = (CV_df.loc[CV_df["status"].str.casefold().eq("toxic"), "Compound name"]
                                .astype(str).unique())
                    if tox_names.size:
                        QIC_exp.loc[QIC_exp["Compound name"].astype(str).isin(tox_names), "Toxicity"] = True
                    QIC_exp["Experiment"] = exp
                    qic_all.append(QIC_exp)
            except Exception as e_qic:
                msg = f"[SKIP-QIC] Experiment {exp}: {type(e_qic).__name__}: {e_qic}"
                print(msg)
                skip_logs.append({"Experiment": exp, "stage": "QIC", "error": f"{type(e_qic).__name__}: {e_qic}"})

        except Exception as e:
            err = ''.join(traceback.format_exception_only(type(e), e)).strip()
            print(f"[SKIP] Experiment {exp} skipped: {err}")
            skip_logs.append({"Experiment": exp, "stage": "main", "error": err})
            continue


    ECM_final  = pd.concat(ECM_all,  ignore_index=True) if ECM_all  else pd.DataFrame()
    Nuclei_final = pd.concat(Nuclei_all, ignore_index=True) if Nuclei_all else pd.DataFrame()
    QC_final = pd.concat(qc_all, ignore_index=True) if qc_all else pd.DataFrame()
    QIC_final = pd.concat(qic_all, ignore_index=True) if qic_all else pd.DataFrame()
    
    return ECM_final, Nuclei_final, QC_final, QIC_final

