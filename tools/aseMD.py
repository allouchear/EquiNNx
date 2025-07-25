from ase import io
from ase.cell import Cell
from ase import Atoms
from ase.io.vasp import read_vasp
from ase.io.extxyz import read_extxyz
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
#from ase.io.trajectory import TrajectoryReader
from ase.io.trajectory import Trajectory
from ase import units
from ase.units import Bohr,Hartree,kcal,mol, GPa
# Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.

import numpy as np
from Utils.Predictor import *
import os
import sys


def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--input_file_name', default=None, type=str, help="Input file name xyz, POSCAR, ....")
	parser.add_argument('--output_traj_name', default=None, type=str, help="name of  output trajectory file, ....")
	parser.add_argument('--nmax', default=10000, type=int, help="nb of total steps during the simulation")
	parser.add_argument('--tstep', default=1.0, type=float, help="timestep in fs")
	parser.add_argument('--verbose', type=int, default=0, help="verbose (default=0)")
	parser.add_argument('--temperature', type=float, default=300, help="temperature in K (default=300)")
	parser.add_argument('--friction', type=float, default=0.001, help="friction for Langevin MD")
	parser.add_argument('--andersen_prob', type=float, default=0.001, help="Andersen probability MD")
	parser.add_argument('--taut', type=float, default=0.5, help="Time constant for Berendsen Temperature (in ps), Default=0.5")
	parser.add_argument('--taup', type=float, default=1.0, help="Time constant for Berendsen pressure(in ps), Default=1.0")
	parser.add_argument('--pressure', type=float, default=1.01325, help="Pressure in bar. Default=1,01325")
	parser.add_argument('--compressibility', type=float, default=4.57e-5, help="Compressibility en bar^-1.")
	parser.add_argument('--external_stress', type=float, default=1., help="external stress for Nose Hoover NPT : symmetric 3x3 tensor, 6-vector, or scalar representing the pressure (in eV/A^3)")
	parser.add_argument('--pfactor', type=float, default=25, help="set for NPT Nose Hoover : cst in the barostat differential eq (pfactor ~ ptime^2 *Bulk modulus)")
	parser.add_argument('--ttime', type=float, default=1, help="set for NVT Nose Hoover : characteristic timescale of thermostat (in ASE internal unit)")
	parser.add_argument('--mask', type=int, nargs="+",default=None, help="optional for NPT Nose Hoover : indicate which direction are allowed to change ex: 1,1,0 dissalow move in z axis")
	parser.add_argument('--mdType', type=str, default="NVE", help="dynamic type : NVE, LANGEVIN , ANDERSEN , NVTBerendsen, NPTBerendsen, NVTNoseHoover, NPTNoseHoover, (default=NVE)")
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--unit_models', default="Hartree,Bohr", type=str, help="the unit of model, that used during train")
	parser.add_argument('--print_nsteps', default=1, type=int, help="Print result for nsteps. Default = 1")


	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=False
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			args = parser.parse_args(["@"+config_file])
		else:
			args = parser.parse_args(["--help"])
	else:
		args = parser.parse_args()

	return args


args = getArguments()

inputfname=args.input_file_name
trajfname=args.output_traj_name
lmodels=args.list_models
lmodels=lmodels[0].split()
mdType=args.mdType.upper()
friction=args.friction
print(lmodels)

scale_distance=None
scale_output=None
if args.unit_models.upper()=="KCAL/MOL,ANG":
	scale_distance=1.0
	scale_output={'energy':kcal/mol,'forces':kcal/mol}
elif args.unit_models.upper()=="HARTREE,BOHR":
	scale_distance=1.0/Bohr
	scale_output={'energy':Hartree,'forces':Hartree/Bohr, 'stress':Hartree/(Bohr*Bohr*Bohr)}
elif args.unit_models.upper()=="EV,ANG":
	scale_distance=1.0
	scale_output={'energy':1.0,'forces':1.0, 'stress':1.0}
else:
		raise ValueError('Unknown unit_models. Please use kcal/mol,Ang or Hartree,Bohr or eV,Ang')
	
print(inputfname)
if "traj" in inputfname :
    print(f'RESTART from {inputfname}')
    #atoms=TrajectoryReader(inputfname)
    traj=Trajectory(inputfname, mode='r')
    #atoms=[]
    for ats in traj:        
        atoms=Atoms(ats)
    #print(atoms)
elif "POSCAR" in inputfname or "poscar" in inputfname:
    print("reading POSCAR file ....")
    atoms = read_vasp(inputfname)
elif ".xyz" in inputfname :
    print("reading .xyz file ....")
    atoms = io.read(inputfname)
    #at = read_extxyz(inputfname)
    #for a in at:
    #    atoms=Atoms(a)

print("--------- Input geometry --------------------")
print("PBC : ",atoms.get_pbc())
print("Cell : " ,atoms.get_cell())
print("Z : " , atoms.get_atomic_numbers())
print("Positions : ",atoms.get_positions())
print("---------------------------------------------")

atoms.calc = Predictor(lmodels,
			scale_distance=scale_distance,
			scale_output=scale_output,
			)

from ase.optimize import BFGS
from ase.vibrations import Vibrations

#BFGS(atoms).run(fmax=0.001)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

tstep = args.tstep # time step in fs
print("mdType=",mdType)
if mdType=="NVE":
	print("VelocityVerlet: timestep=", tstep, " fs", ", temperature_K=",args.temperature, ", friction=",friction)
	dyn = VelocityVerlet(atoms, tstep * units.fs)
elif mdType=="LANGEVIN":
	print("Langevin: timestep=", tstep , " fs", ", temperature_K=",args.temperature, ", friction=",friction)
	dyn = Langevin(atoms, tstep * units.fs, temperature_K=args.temperature, friction=friction)
elif mdType=="ANDERSEN":
	print("Andersen: timestep=", tstep,  " fs", ", temperature_K=",args.temperature, ", andersen_prob=",args.andersen_prob)
	dyn = Andersen(atoms, tstep * units.fs, temperature_K=args.temperature, andersen_prob=args.andersen_prob)
elif mdType=="NVTBERENDSEN":
	print("NVTBerendsen: timestep=", tstep,  " fs", ", temperature_K=",args.temperature, ", tau temperature=",args.taut, ' ps')
	dyn = NVTBerendsen(atoms, tstep * units.fs, temperature_K=args.temperature, taut=args.taut*1000*units.fs)
elif mdType=="NPTBERENDSEN":
	print("NPTBerendsen: timestep=", tstep , " fs", "temperature_K=",args.temperature, ", tau temperature=",args.taut, ' ps',
		", taup=",args.taup, " ps", ", compressibility=",args.compressibility, " bar^-1")
	dyn = NPTBerendsen(atoms, tstep * units.fs, temperature_K=args.temperature, 
		taut=args.taut*1000*units.fs, pressure_au=args.pressure * units.bar,
		taup=args.taup*1000*units.fs, compressibility=args.compressibility / units.bar)
elif mdType=="NVTNOSEHOOVER":
    #set ttime for thermostat and pfactor=None
    print("NVTNoseHoover: timestep=", tstep,  " fs", ", temperature_K=",args.temperature, 
           "ttime =",args.ttime," fs" ,"external stress=", args.external_stress, "eV/A^3", "=", args.external_stress/GPa,"GPa")
    dyn = NPT(atoms, tstep * units.fs, temperature_K=args.temperature, externalstress=args.external_stress, ttime=args.ttime ,pfactor=None)  
elif mdType=="NPTNOSEHOOVER":
    #set pfactor for barostat ttime=None
    print("NPTNoseHoover: timestep=", tstep,  " fs", ", temperature_K=",args.temperature, ", external stress =",args.external_stress,"eV/A^3", "=", args.external_stress/GPa,"GPa"
          ,", pfactor =",args.pfactor )
    if args.mask is not None:
      mask=tuple(args.mask)
    else:
      mask=None
    dyn = NPT(atoms, tstep * units.fs, temperature_K=args.temperature, pfactor=args.pfactor, externalstress=args.external_stress, mask=mask, ttime=None) 
else:
	print("Unknown md type, Only NVE, LANGEVIN, ANDERSEN, NVTBERENDSEN, NPTBERENDSEN , NVTNOSEHOOVER, NPTNOSEHOOVER are accepted")
	sys.exit(1)


niter=0
nsteps=args.print_nsteps
#nmax=10000
nmax=args.nmax
nstepsT=20
def trace_pressure(a=atoms):
    stress_tensor=atoms.get_stress(atoms)
    xx=stress_tensor[0]
    yy=stress_tensor[0]
    zz=stress_tensor[0]
    P_trace = (xx+yy+zz)/3
    return P_trace

def printenergy(a=atoms):
    ndigits=len(str(printenergy.nmax))
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    vol = a.get_volume()
    st ="Iter = {:{ndigits}d} / {:{ndigits}d} ; Epot/atom = {:.3f} eV ; Ekin/atom = {:.3f} eV ; T = {:3.0f} K ; Epot = {:.3f} eV ; Epot= {:0.3f} eV ; Etot= {:0.3f} eV ; V = {:3.0f} Ang^3".format(
		printenergy.niter, printenergy.nmax, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, a.get_potential_energy(), (epot+ekin)*len(a), vol, ndigits=ndigits
		)
    print(st,flush=True)
    a.calc.reset_neighbor_list()
    printenergy.niter += printenergy.nsteps

def printenergymolecule(a=atoms):
    ndigits=len(str(printenergy.nmax))
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    st ="Iter = {:{ndigits}d} / {:{ndigits}d} ; Epot/atom = {:.3f} eV ; Ekin/atom = {:.3f} eV ; T = {:3.0f} K ; Epot = {:.3f} eV ; Epot= {:0.3f} eV ; Etot= {:0.3f} eV".format(
		printenergy.niter, printenergy.nmax, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, a.get_potential_energy(), (epot+ekin)*len(a), ndigits=ndigits
		)
    print(st,flush=True)
    a.calc.reset_neighbor_list()
    printenergy.niter += printenergy.nsteps


def printenergyandpress(a=atoms):
    ndigits=len(str(printenergy.nmax))
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    vol = a.get_volume()
    P=trace_pressure(a)
    st ="Iter = {:{ndigits}d} / {:{ndigits}d} ; Epot/atom = {:.3f} eV ; Ekin/atom = {:.3f} eV ; T = {:3.0f} K ; Etot/atom = {:.3f} eV ; Epot= {:0.3f} eV ; Etot= {:0.3f} eV ; V = {:8.2f} Ang^3 ; trace_P = {:8.3f} GPa".format(
	printenergy.niter, printenergy.nmax, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, a.get_potential_energy(), (epot+ekin)*len(a), vol, P/GPa,ndigits=ndigits)
    print(st, flush=True)
    a.calc.reset_neighbor_list()
    printenergy.niter += printenergy.nsteps

printenergy.niter = 0
printenergy.nmax = nmax
printenergy.nsteps = nsteps


# Now run the dynamics
#printenergy(atoms)
#print("Forces = ", atoms.get_forces())

# Print energy every 10th step
periodic = atoms.pbc.any()
if periodic :
	dyn.attach(printenergyandpress, interval=nsteps)
else:
	dyn.attach(printenergymolecule, interval=nsteps)

# We also want to save the positions of all atoms after every 50th time step.
traj = Trajectory(trajfname, 'w', atoms)
#dyn.attach(traj.write, interval=nstepsT)

# Now run the dynamics
dyn.run(nmax)
