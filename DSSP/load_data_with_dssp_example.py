import numpy as np
import glob

input_folder = '/media/tue/Data/Dropbox/ComputationalGenetics/data/pnet_with_msa_and_dssp_testing/'
search_command = input_folder + "*.npz"
input_files = [f for f in sorted(glob.glob(search_command))]

for input_file in input_files:
    data = np.load(input_file)
    print("variables in data: {:}".format(data.files))

    seq = data['seq']                       #Sequence encoded with the amino acid alphabet given in variable AA_LIST (standard)
    pssm = data['pssm']                     #pssm information (standard)
    entropy = data['entropy']               #entropy (standard)
    rCa = data['rCa']                       #(standard)
    rCb = data['rCb']                       #(standard)
    rN = data['rN']                         #(standard)
    id = data['id']                         #pdb_id
    log_units = data['log_units']           #the unit magnitude of the coordinate information (-9 = Nanometers, -10 = Angstrom, -12 = picometers, ect.)
    AA_LIST = data['AA_LIST']               #The amino acid encoding list
    msa = data['msa']                       #Up to 30000 randomly drawn msas from the pnet raw msa files (if there is less than 30000 it has taken all the msas that were given)
    nmsa_org = data['nmsa_org']             #The number of MSAs given in the raw msa files, from which the msas have been drawn from
    dssp = data['dssp']                     #dssp information as calculated by https://github.com/PDB-REDO/dssp, and encoded by the DSSP_ALPHABET (8 = no dssp information)
    DSSP_ALPHABET = data['DSSP_ALPHABET']   #DSSP alphabet is given as:

    # ├────────────┼──────────────┼─────────────┤
    # │H           │ HELX_RH_AL_P │ Alphahelix  │
    # ├────────────┼──────────────┼─────────────┤
    # │B           │ STRN         │ Betabridge  │
    # ├────────────┼──────────────┼─────────────┤
    # │E           │ STRN         │ Strand      │
    # ├────────────┼──────────────┼─────────────┤
    # │G           │ HELX_RH_3T_P │ Helix_3     │
    # ├────────────┼──────────────┼─────────────┤
    # │I           │ HELX_RH_PI_P │ Helix_5     │
    # ├────────────┼──────────────┼─────────────┤
    # │P           │ HELX_LH_PP_P │ Helix_PPII  │
    # ├────────────┼──────────────┼─────────────┤
    # │T           │ TURN_TY1_P   │ Turn        │
    # ├────────────┼──────────────┼─────────────┤
    # │S           │ BEND         │ Bend        │
    # ├────────────┼──────────────┼─────────────┤
    # │' '(space)  │ OTHER        │ Loop        │



    print("stop")


