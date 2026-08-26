import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import csv
import argparse
from pathlib import Path
from ase.data import chemical_symbols


def read_aselmdb(filename):
    import lmdb, zlib, json
    from ase import Atoms

    env = lmdb.open(filename, readonly=True, lock=False, subdir=False)
    atoms_list = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            obj = json.loads(zlib.decompress(value).decode("utf-8"))
            if type(obj) != dict:
                continue
            atoms = Atoms(
                numbers=obj["numbers"],
                positions=obj["positions"],
                cell=obj.get("cell"),
                pbc=obj.get("pbc", False)
            )
            atoms.info["energy"] = obj["energy"]
            atoms.info["forces"] = obj["forces"]
            atoms.info["stress"] = np.asarray(obj["stress"])
            atoms.calc = None
            atoms_list.append(atoms)

    return atoms_list


def contains_only(atoms, latoms):
    if latoms is None:
        return True
    unique_elements = set(atoms.get_chemical_symbols())
    return unique_elements.issubset(set(latoms))


def compute_global_e0s(directories, nmolsmax=-1, latoms=None, natoms=None):
    if isinstance(directories, str):
        directories = [directories]

    db_files = []
    for directory in directories:
        folder = Path(directory)
        db_files.extend(sorted(folder.glob('db_*.aselmdb')))
    print(f"Reading {len(db_files)} files to compute global E0s...", flush=True)

    A_rows = []
    B_vals = []

    n_total = len(db_files)
    for i_db, db_path in enumerate(db_files, 1):
        print(f"\t[{i_db}/{n_total}] {db_path.name}...", end=" ", flush=True)
        structures = read_aselmdb(str(db_path))
        print(f"{len(structures)} structures", flush=True)
        for atoms in structures:
            if nmolsmax > 0 and len(A_rows) >= nmolsmax:
                break
            if atoms is None:
                continue
            if not contains_only(atoms, latoms):
                continue
            if natoms is not None and atoms.get_atomic_numbers().shape[0] != natoms:
                continue
            B_vals.append(atoms.info["energy"])
            A_rows.append(atoms.get_atomic_numbers())
        del structures

    if not A_rows:
        print("No molecules found for E0 computation.", flush=True)
        return {}

    print("Sorting of z table...", flush=True)
    all_z = set()
    for z_arr in A_rows:
        all_z.update(z_arr.tolist())
    z_table = sorted(list(all_z))

    print("Build A matrix...", flush=True)
    len_train = len(A_rows)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.array(B_vals)

    for i, z_arr in enumerate(A_rows):
        for j, z in enumerate(z_table):
            A[i, j] = np.count_nonzero(z_arr == z)

    print(f"Solving system A·E0s = B ({len(A_rows)} equations, {len(z_table)} unknowns)...", flush=True)
    try:
        E0s_solution = np.linalg.lstsq(A, B, rcond=None)[0]
        e0s_dict = {z: E0s_solution[i] for i, z in enumerate(z_table)}
    except np.linalg.LinAlgError:
        print("Warning: E0 computation failed. Using 0.0.", flush=True)
        e0s_dict = {z: 0.0 for z in z_table}

    print(f"Global E0s computed for {len(e0s_dict)} elements.", flush=True)
    for z, e0 in sorted(e0s_dict.items()):
        print(f"  Z={z} ({chemical_symbols[z]}): E0 = {e0:.6f}", flush=True)
    return e0s_dict


def save_e0s_csv(e0s_dict, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Z", "element", "E0"])
        for z in sorted(e0s_dict.keys()):
            writer.writerow([z, chemical_symbols[z], e0s_dict[z]])
    print(f"E0s saved to {output_file}", flush=True)


def load_e0s_csv(csv_file):
    e0s_dict = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e0s_dict[int(row["Z"])] = float(row["E0"])
    print(f"Loaded E0s from {csv_file}: {len(e0s_dict)} elements", flush=True)
    return e0s_dict


def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--directories", type=str, required=True, help="Comma-separated list of directories containing db_*.aselmdb files (e.g. --directories=folder1,folder2)")
    parser.add_argument("--output", type=str, default="e0s.csv", help="Output CSV filename. Default=e0s.csv")
    parser.add_argument("--n_atoms", default=-1, type=int, help="Keep only structures with exactly N atoms. -1 => all")
    parser.add_argument("--num_structures", default=-1, type=int, help="Max number of structures per file. -1 => all")

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

    natoms = args.n_atoms
    if natoms < 0:
        natoms = None
    nmolsmax = args.num_structures

    e0s_dict = compute_global_e0s(
        args.directories.split(','),
        nmolsmax=nmolsmax,
        latoms=None,
        natoms=natoms
    )

    if e0s_dict:
        save_e0s_csv(e0s_dict, args.output)
    else:
        print("No E0s computed.", flush=True)
