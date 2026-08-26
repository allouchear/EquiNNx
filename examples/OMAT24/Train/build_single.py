import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
from joblib import Parallel, delayed
from joblib import cpu_count
import warnings
from ase.neighborlist import neighbor_list
import h5py
import csv
from ase import Atoms
from ase.data import chemical_symbols
import argparse
from pathlib import Path


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


def load_e0s_csv(csv_file):
    e0s_dict = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e0s_dict[int(row["Z"])] = float(row["E0"])
    print(f"Loaded E0s from {csv_file}: {len(e0s_dict)} elements", flush=True)
    return e0s_dict


def compute_local_e0s(mols):
    A_rows = []
    B_vals = []
    for atoms in mols:
        B_vals.append(atoms.info["energy"])
        A_rows.append(atoms.get_atomic_numbers())

    if not A_rows:
        return {}

    all_z = set()
    for z_arr in A_rows:
        all_z.update(z_arr.tolist())
    z_table = sorted(list(all_z))

    len_train = len(A_rows)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.array(B_vals)

    for i, z_arr in enumerate(A_rows):
        for j, z in enumerate(z_table):
            A[i, j] = np.count_nonzero(z_arr == z)

    try:
        E0s_solution = np.linalg.lstsq(A, B, rcond=None)[0]
        e0s_dict = {z: E0s_solution[i] for i, z in enumerate(z_table)}
    except np.linalg.LinAlgError:
        print("Warning: E0 computation failed. Using 0.0.", flush=True)
        e0s_dict = {z: 0.0 for z in z_table}

    print(f"Local E0s computed for {len(e0s_dict)} elements:", flush=True)
    for z, e0 in sorted(e0s_dict.items()):
        print(f"  Z={z} ({chemical_symbols[z]}): E0 = {e0:.6f}", flush=True)
    return e0s_dict


def sparse_pairwise_indices(num):
    if num < 1:
        raise ValueError(f'num must be larger than 0, received {num}')
    dst_idx = np.repeat(np.arange(num), num)
    src_idx = np.tile(np.arange(num), num)
    return dst_idx, src_idx


def gather_idx(R, idx):
    return np.take(R, idx, axis=0, out=None, mode='raise')


def get_distances(R, dst_idx, src_idx):
    dst_R = gather_idx(R, dst_idx)
    src_R = gather_idx(R, src_idx)
    distances = np.linalg.norm(src_R - dst_R, axis=-1)
    return distances


def cut_idx(R, dst_idx, src_idx, cutoff=None):
    if cutoff is None:
        return dst_idx, src_idx
    distances = get_distances(R, dst_idx, src_idx)
    dst_idx = dst_idx[distances <= cutoff]
    src_idx = src_idx[distances <= cutoff]
    return dst_idx, src_idx


def get_idx_list(atoms, cutoff=None):
    if any(atoms.get_pbc()):
        if cutoff is None:
            print("cutoff is needed for periodic system")
            sys.exit()
        dst_idx, src_idx, S = neighbor_list('ijS', atoms, cutoff, self_interaction=False)
        offsets = np.dot(S, atoms.get_cell())
    else:
        N = len(atoms.get_atomic_numbers())
        dst_idx, src_idx = sparse_pairwise_indices(N)
        offsets = None
        if cutoff is not None:
            dst_idx, src_idx = cut_idx(atoms.get_positions(), dst_idx, src_idx, cutoff=cutoff)
    return dst_idx, src_idx, offsets


def getNBNE(njobs, nmols):
    m = nmols // njobs
    M = [m] * njobs
    nr = nmols % njobs
    for i in range(nr):
        M[i] += 1
    NB = [0] * njobs
    NE = [0] * njobs
    NE[0] = M[0]
    for i in range(1, njobs):
        NB[i] = NE[i - 1]
        NE[i] = NB[i] + M[i]
    return NB, NE


def set_idx(mols, cutoff, iBegin, iEnd, iWorker, njobs=1, total=0):
    a_dst_idx = []
    a_src_idx = []
    a_offsets = []
    n = iEnd - iBegin
    step = max(1, n // 100)
    for ii, im in enumerate(range(iBegin, iEnd)):
        atoms = mols[im]
        dst_idx, src_idx, offsets = get_idx_list(atoms, cutoff=cutoff)
        a_dst_idx.append(dst_idx[:])
        a_src_idx.append(src_idx[:])
        a_offsets.append(offsets)
        done = iBegin + ii + 1
        if (ii+1) % step == 0:
            msg = f"  Progress: {done}/{total}" if njobs == 1 else f"  Worker {iWorker}: {ii+1}/{n}"
            print(f"\r{msg:<40}", end="\r", flush=True)
    print(f"\r{'':40}", end="\r", flush=True)
    return a_dst_idx, a_src_idx, a_offsets


def build_idx_list(mols, nmols, label="", njobs=1, cutoff=None):
    all_dst_idx = []
    all_src_idx = []
    all_offsets = []
    print(f"Setting of neighbor lists ({label})...", flush=True)
    print("Number of structures =", nmols)
    if njobs > nmols:
        njobs = nmols
    print("Number of workers =", njobs)
    if njobs == 1:
        dst_idx, src_idx, offsets = set_idx(mols, cutoff, 0, nmols, 1, njobs=1, total=nmols)
        all_dst_idx.extend(dst_idx)
        all_src_idx.extend(src_idx)
        all_offsets.extend(offsets)
    else:
        NB, NE = getNBNE(njobs, nmols)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            r = Parallel(n_jobs=njobs, verbose=0)(
                delayed(set_idx)(mols, cutoff, NB[i], NE[i], i + 1, njobs=njobs, total=nmols) for i in range(njobs)
            )
        for a in r:
            dst_idx, src_idx, offsets = a
            all_dst_idx.extend(dst_idx)
            all_src_idx.extend(src_idx)
            all_offsets.extend(offsets)
    nnone = 0
    for offset in all_offsets:
        if offset is None:
            nnone += 1
    if nnone == len(all_offsets):
        all_offsets = None
    print(f"End setting of neighbor lists ({label})", flush=True)
    return all_dst_idx, all_src_idx, all_offsets


def build_index(mols, label="", cutoff=5, njobs=1):
    nmols = len(mols)
    all_dst_idx, all_src_idx, all_offsets = build_idx_list(mols, nmols, label=label, njobs=njobs, cutoff=cutoff)
    print(f"End build index ({label})", flush=True)

    for im in range(nmols):
        atoms = mols[im]
        atoms.info['dst_idx'] = all_dst_idx[im]
        atoms.info['src_idx'] = all_src_idx[im]
        if all_offsets is not None and len(all_offsets) > 1:
            atoms.info['offsets'] = all_offsets[im]
        else:
            atoms.info['offsets'] = 'NONE'
    return mols


def subtract_e0s(mols, e0s_dict):
    for atoms in mols:
        total_energy = atoms.info["energy"]
        interaction_energy = total_energy
        for z, e0 in e0s_dict.items():
            n_atoms = np.count_nonzero(atoms.get_atomic_numbers() == z)
            interaction_energy -= n_atoms * e0
        atoms.info["energy"] = interaction_energy
    return mols


def contains_only(atoms, latoms):
    if latoms is None:
        return True
    unique_elements = set(atoms.get_chemical_symbols())
    return unique_elements.issubset(set(latoms))


def getData(mols):
    positions = []
    edipole = []
    polar = []
    alfa = []
    beta = []
    hpolar = []
    Z = []
    pbc = []
    cells = []
    dst_idx = []
    src_idx = []
    offsets = []
    ID = []
    n_idx = []

    energy = []
    forces = []
    stress = []

    for atoms in mols:
        Z.append(atoms.get_atomic_numbers())
        pbc.append(atoms.get_pbc())
        cells.append(atoms.get_cell()[:])
        positions.append(np.asarray(atoms.get_positions()))
        if 'energy' in atoms.info.keys():
            energy.append(atoms.info['energy'])
        if 'forces' in atoms.info.keys():
            forces.append(atoms.info['forces'])
        if 'stress' in atoms.info.keys():
            stress.append(atoms.info['stress'])
        if 'ID' in atoms.info.keys():
            ID.append(atoms.info['ID'])
        if 'edipole' in atoms.info.keys():
            edipole.append(atoms.info['edipole'])
        if 'polarizability' in atoms.info.keys():
            polar.append(atoms.info['polarizability'])
        if 'hyperpolarizability' in atoms.info.keys():
            hpolar.append(atoms.info['hyperpolarizability'])
        if 'dst_idx' in atoms.info.keys():
            dst_idx.extend(np.asarray(atoms.info['dst_idx']))
            n_idx.append(len(atoms.info['dst_idx']))
        if 'src_idx' in atoms.info.keys():
            src_idx.extend(np.asarray(atoms.info['src_idx']))
        if 'offsets' in atoms.info.keys():
            if atoms.info['offsets'] is None or (type(atoms.info['offsets']) is str and atoms.info['offsets'] == 'NONE'):
                offsets.extend([None])
            else:
                offsets.extend(np.asarray(atoms.info['offsets']))
        if 'alfa' in atoms.info.keys():
            alfa.append(atoms.info['alfa'])
        if 'beta' in atoms.info.keys():
            beta.append(atoms.info['beta'])

    data = {}
    data['N'] = []
    for z in Z:
        data['N'].append(z.shape[0])
    data['N'] = np.asarray(data['N'])

    nmax = 0
    for z in Z:
        if nmax < z.shape[0]:
            nmax = z.shape[0]
    ZZ = []
    for i, z in enumerate(Z):
        nres = nmax - z.shape[0]
        if nres > 0:
            zz = z.tolist()
            zz += [0] * nres
            ZZ.append(zz)
        else:
            ZZ.append(z.tolist())
    Z = np.asarray(ZZ)
    R = []
    for i, r in enumerate(positions):
        nres = nmax - r.shape[0]
        if nres > 0:
            zeros = np.zeros((nres, 3))
            R.append(np.vstack((r, zeros)))
        else:
            R.append(r)
    R = np.asarray(R)

    pbc = np.asarray(pbc)
    cells = np.asarray(cells)
    data['Z'] = Z
    data['R'] = R
    if len(ID) >= 1:
        data['ID'] = np.asarray(ID)
    else:
        data['ID'] = None

    if len(alfa) >= 1:
        data['alfa'] = np.asarray(alfa)
    else:
        data['alfa'] = None
    if len(beta) >= 1:
        data['beta'] = np.asarray(beta)
    else:
        data['beta'] = None
    if len(edipole) >= 1:
        edipole = np.asarray(edipole)
        data['edipole'] = edipole
        data['sdipole'] = np.linalg.norm(edipole, ord=None, axis=1, keepdims=False)
    else:
        data['sdipole'] = None

    if len(edipole) >= 1:
        edipole = np.asarray(edipole)
        data['edipole'] = edipole
    else:
        data['edipole'] = None
    if len(polar) >= 1:
        polar = np.asarray(polar)
        data['polarizability'] = polar
    else:
        data['polarizability'] = None
    if len(hpolar) >= 1:
        hpolar = np.asarray(hpolar)
        data['hyperpolarizability'] = hpolar
    else:
        data['hperpolarizability'] = None
    if len(energy) >= 1:
        data['energy'] = np.asarray(energy)
    else:
        data['energy'] = None

    if len(forces) >= 1:
        nmax = 0
        for f in forces:
            if len(f) > nmax:
                nmax = len(f)
        forcesnew = []
        for f in forces:
            nadd = nmax - len(f)
            if nadd > 0:
                for i in range(nadd):
                    f.append([0.0, 0.0, 0.0])
            forcesnew.append(f)
        data['forces'] = np.asarray(forcesnew)
    else:
        data['forces'] = None

    if len(stress) >= 1:
        stress = np.asarray(stress)
        data['stress'] = stress
    else:
        data['stress'] = None

    data['cells'] = cells
    if len(n_idx) >= 1:
        data['n_idx'] = np.asarray(n_idx)
    else:
        data['n_idx'] = None
    if len(dst_idx) >= 1:
        data['dst_idx'] = dst_idx
    else:
        data['dst_idx'] = None

    if len(src_idx) >= 1:
        data['src_idx'] = src_idx
    else:
        data['src_idx'] = None

    if len(offsets) >= 1:
        if np.all(np.array(offsets) == None):
            data['offsets'] = None
        else:
            data['offsets'] = offsets
    else:
        data['offsets'] = None

    return data


def saveDatah5(data, filename):
    with h5py.File(filename, "w") as h5file:
        for key, value in data.items():
            if value is not None:
                h5file.create_dataset(key, data=value, compression="lzf")


def getArguments():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--file", type=str, required=True, help="Path to a single db_*.aselmdb file")
    parser.add_argument("--e0s_file", type=str, default=None, help="CSV file with E0s (columns: Z, element, E0). If not given, E0s are computed from this file.")
    parser.add_argument("--no_e0s", action="store_true", help="Disable E0 subtraction entirely")
    parser.add_argument('--cutoff', type=float, default=-1, help="Cutoff for neighbor list. Default=-1 (no cutoff)")
    parser.add_argument('--n_atoms', default=-1, type=int, help="Keep only structures with exactly N atoms. -1 => all")
    parser.add_argument('--num_structures', default=-1, type=int, help="Max number of structures. -1 => all")
    parser.add_argument('--njobs', default=-1, type=int, help="Workers for build_index. -1 => all cores")
    parser.add_argument("--output", default=None, type=str, help="Output .h5 filename. Default: derived from input filename")

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

    db_path = Path(args.file)
    if not db_path.is_file():
        print(f"ERROR: file {db_path} not found", flush=True)
        sys.exit(1)

    natoms = args.n_atoms
    if natoms < 0:
        natoms = None
    nmolsmax = args.num_structures

    cutoff_idx = None
    if args.cutoff > 0:
        cutoff_idx = args.cutoff

    njobs = args.njobs
    if njobs <= 0:
        njobs = cpu_count()

    # Load E0s from CSV if provided
    e0s_dict = {}
    if args.e0s_file is not None:
        if not os.path.isfile(args.e0s_file):
            print(f"ERROR: E0s file {args.e0s_file} not found", flush=True)
            sys.exit(1)
        e0s_dict = load_e0s_csv(args.e0s_file)

    # Read structures
    print(f"Reading {db_path.name}...", flush=True)
    structures = read_aselmdb(str(db_path))
    print(f"  {len(structures)} structures read", flush=True)

    mols = []
    i = 0
    for atoms in structures:
        if nmolsmax > 0 and i + 1 > nmolsmax:
            break
        if atoms is None:
            continue
        if not contains_only(atoms, None):
            continue
        if natoms is not None and atoms.get_atomic_numbers().shape[0] != natoms:
            continue
        mols.append(atoms)
        i += 1
        atoms.info['ID'] = int(i)
        atoms.info['name'] = atoms.symbols
    del structures

    if len(mols) == 0:
        print("No structures matched filters.", flush=True)
        sys.exit(0)

    print(f"  {len(mols)} structures kept after filters", flush=True)

    # Compute local E0s if no CSV provided and --no_e0s not set
    if not e0s_dict and not args.no_e0s:
        print("Computing local E0s from this file...", flush=True)
        e0s_dict = compute_local_e0s(mols)

    # Subtract E0s
    if e0s_dict:
        mols = subtract_e0s(mols, e0s_dict)

    # Determine cutoff
    local_cutoff = cutoff_idx
    if local_cutoff is None and any(mols[0].get_pbc()):
        local_cutoff = 5.0

    # Build index (parallelized with joblib)
    mols = build_index(mols, label=db_path.name, cutoff=local_cutoff, njobs=njobs)

    # Extract data and save
    print("Extract data...", flush=True)
    data = getData(mols)

    if args.output is not None:
        h5_path = Path(args.output)
    else:
        db_name = db_path.stem
        data_name = f"{db_name}_n_structures_{len(mols)}_cutidx_{local_cutoff}"
        if e0s_dict:
            data_name += "_e0s_global"
        else:
            data_name += "_e0s_none"
        data_name = data_name.replace(".", "p")
        h5_path = db_path.parent / f"{data_name}.h5"

    print("Save data...", flush=True)
    saveDatah5(data, str(h5_path))
    print(f"Done: {len(mols)} molecules -> {h5_path.name}", flush=True)
