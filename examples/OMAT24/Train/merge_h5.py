import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import h5py
import argparse
from pathlib import Path


STRUCT_DATASETS = ["Z", "R", "forces", "energy", "stress", "pbc", "cells", "n_idx", "ID",
                   "edipole", "sdipole", "polarizability", "hyperpolarizability", "alfa", "beta"]
VAR_DATASETS = ["dst_idx", "src_idx", "offsets"]


def merge_h5_files(h5_list, output_file):
    print(f"\n[Fusion] Merging {len(h5_list)} H5 files...", flush=True)

    # Pass 1: read N + variable-size dataset lengths per file
    struct_offsets = []
    var_offsets = []
    struct_acc = 0
    var_acc = 0
    all_N = []
    detected_struct = set()
    detected_var = set()

    for i, h5_path in enumerate(h5_list):
        print(f"[Fusion]   Pass 1: {i + 1}/{len(h5_list)} {h5_path.name}", flush=True)
        with h5py.File(h5_path, "r") as f:
            n_mol = len(f["N"][()])
            all_N.append(f["N"][()])
            struct_offsets.append(struct_acc)
            var_offsets.append(var_acc)
            struct_acc += n_mol

            n_var = 0
            for name in VAR_DATASETS:
                if name in f and f[name][()] is not None:
                    n_var = len(f[name])
                    detected_var.add(name)
            var_acc += n_var

            for name in STRUCT_DATASETS:
                if name in f and f[name][()] is not None:
                    detected_struct.add(name)

    all_N = np.concatenate(all_N)
    nmax_global = int(all_N.max())
    total_structures = len(all_N)
    total_var = var_acc

    print(f"[Fusion] {total_structures} structures, nmax = {nmax_global}", flush=True)
    print(f"[Fusion] Detected struct datasets: {sorted(detected_struct)}", flush=True)
    print(f"[Fusion] Detected var datasets:    {sorted(detected_var)}", flush=True)

    # Create output file with pre-allocated datasets
    with h5py.File(output_file, "w") as h5out:
        h5out.create_dataset("N", data=all_N, compression="lzf")

        # Detect dtypes from first file that has each dataset
        dtypes = {}
        with h5py.File(h5_list[0], "r") as f0:
            for name in STRUCT_DATASETS:
                if name in f0 and f0[name][()] is not None:
                    dtypes[name] = f0[name].dtype

        for name in STRUCT_DATASETS:
            if name not in detected_struct:
                continue
            dtype = dtypes.get(name, "float64")
            if name in ("Z", "R", "forces"):
                if name == "Z":
                    shape = (total_structures, nmax_global)
                    chunks = (min(total_structures, 1024), nmax_global)
                else:
                    shape = (total_structures, nmax_global, 3)
                    chunks = (min(total_structures, 1024), nmax_global, 3)
            elif name == "energy":
                shape = (total_structures,)
                chunks = (min(total_structures, 4096),)
            elif name == "stress":
                shape = (total_structures, 6)
                chunks = (min(total_structures, 1024), 6)
            elif name == "pbc":
                shape = (total_structures, 3)
                chunks = (min(total_structures, 4096), 3)
            elif name == "cells":
                shape = (total_structures, 3, 3)
                chunks = (min(total_structures, 1024), 3, 3)
            elif name == "n_idx":
                shape = (total_structures,)
                chunks = (min(total_structures, 4096),)
            elif name == "ID":
                shape = (total_structures,)
                chunks = (min(total_structures, 4096),)
            elif name in ("edipole", "sdipole"):
                shape = (total_structures, 3)
                chunks = (min(total_structures, 1024), 3)
            elif name in ("polarizability", "alfa"):
                shape = (total_structures, 6)
                chunks = (min(total_structures, 1024), 6)
            elif name == "hyperpolarizability":
                shape = (total_structures, 6, 6)
                chunks = (min(total_structures, 256), 6, 6)
            elif name == "beta":
                shape = (total_structures, 6, 6, 6)
                chunks = (min(total_structures, 128), 6, 6, 6)
            else:
                continue
            h5out.create_dataset(name, shape=shape, dtype=dtype,
                                 chunks=chunks, compression="lzf")

        var_dtypes = {}
        with h5py.File(h5_list[0], "r") as f0:
            for name in VAR_DATASETS:
                if name in f0 and f0[name][()] is not None:
                    var_dtypes[name] = f0[name].dtype

        for name in VAR_DATASETS:
            if name not in detected_var:
                continue
            dtype = var_dtypes.get(name, "int64")
            if name == "offsets":
                shape = (total_var, 3)
                chunks = (min(total_var, 4096), 3)
            else:
                shape = (total_var,)
                chunks = (min(total_var, 4096),)
            h5out.create_dataset(name, shape=shape, dtype=dtype,
                                 chunks=chunks, compression="lzf")

        # Pass 2: stream one file at a time
        print("[Fusion] Streaming data...", flush=True)
        for i, h5_path in enumerate(h5_list):
            s_off = struct_offsets[i]
            v_off = var_offsets[i]

            with h5py.File(h5_path, "r") as f:
                n_mol = len(f["N"][()])

                # --- Struct-size datasets ---
                z_raw = f["Z"][()]
                r_raw = f["R"][()]
                local_max = z_raw.shape[1]

                if local_max < nmax_global:
                    z_pad = np.zeros((n_mol, nmax_global), dtype=z_raw.dtype)
                    z_pad[:, :local_max] = z_raw
                    r_pad = np.zeros((n_mol, nmax_global, 3), dtype=r_raw.dtype)
                    r_pad[:, :local_max, :] = r_raw
                else:
                    z_pad = z_raw
                    r_pad = r_raw

                h5out["Z"][s_off:s_off + n_mol] = z_pad
                h5out["R"][s_off:s_off + n_mol] = r_pad

                if "forces" in f and f["forces"][()] is not None:
                    f_raw = f["forces"][()]
                    if local_max < nmax_global:
                        f_pad = np.zeros((n_mol, nmax_global, 3), dtype=f_raw.dtype)
                        f_pad[:, :local_max, :] = f_raw
                    else:
                        f_pad = f_raw
                    h5out["forces"][s_off:s_off + n_mol] = f_pad

                for name in ("energy", "stress", "pbc", "cells", "n_idx", "ID",
                             "edipole", "sdipole", "polarizability",
                             "hyperpolarizability", "alfa", "beta"):
                    if name in f and f[name][()] is not None:
                        h5out[name][s_off:s_off + n_mol] = f[name][()]

                # --- Variable-size datasets ---
                for name in VAR_DATASETS:
                    if name in f and f[name][()] is not None:
                        n_var = len(f[name])
                        h5out[name][v_off:v_off + n_var] = f[name][()]

            print(f"[Fusion]   {i + 1}/{len(h5_list)} {h5_path.name}", flush=True)

    print(f"[Fusion] Final file: {output_file} ({total_structures} structures)", flush=True)


def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--input", type=str, nargs="+", required=True, help="List of intermediate .h5 files to merge")
    parser.add_argument("--output", type=str, required=True, help="Output merged .h5 filename")

    config_file = 'config.txt'
    fromFile = False
    if len(sys.argv) == 1:
        fromFile = False
    if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
        config_file = sys.argv[1]
        fromFile = True

    if fromFile is True:
        print("Try to read configuration from", config_file, "file", flush=True)
        if os.path.isfile(config_file):
            args = parser.parse_args(["@" + config_file])
        else:
            args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = getArguments()

    h5_list = [Path(f) for f in args.input]
    for p in h5_list:
        if not p.is_file():
            print(f"ERROR: file {p} not found", flush=True)
            sys.exit(1)

    merge_h5_files(h5_list, args.output)
    print("Done.", flush=True)
