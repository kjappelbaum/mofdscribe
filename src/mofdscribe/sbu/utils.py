""""
Routines for the conversion between pymatgen molecules and RDKit. 

Inspired by/copied from 
https://github.com/mjwen/bondnet/blob/b719bd85235012c567298cb0da81ef113e3744bc/bondnet/core/rdmol.py#L265
"""
from collections import defaultdict

from openbabel import openbabel as ob
from pymatgen.core import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem, BondType
from rdkit.Geometry import Point3D


def create_rdkit_mol(
    species, coords, bond_types, formal_charge=None, name=None, force_sanitize=True
):
    """
    Create a rdkit mol from scratch.

    Followed: https://sourceforge.net/p/rdkit/mailman/message/36474923/

    Args:
        species (list): species str of each molecule
        coords (2D array): positions of atoms
        bond_types (dict): with bond indices (2 tuple) as key and bond type
            (e.g. Chem.rdchem.BondType.DOUBLE) as value
        formal_charge (list): formal charge of each atom
        name (str): name of the molecule
        force_sanitize (bool): whether to force the sanitization of molecule.
            If `True` and the sanitization fails, it generally throw an error
            and then stops. If `False`, will try to sanitize first, but if it fails,
            will proceed smoothly giving a warning message.

    Returns:
        rdkit Chem.Mol
    """

    m = Chem.Mol()
    edm = Chem.EditableMol(m)
    conformer = Chem.Conformer(len(species))

    for i, (s, c) in enumerate(zip(species, coords)):
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        if formal_charge is not None:
            cg = formal_charge[i]
            if cg is not None:
                atom.SetFormalCharge(cg)
        atom_idx = edm.AddAtom(atom)
        conformer.SetAtomPosition(atom_idx, Point3D(*c))

    for b, t in bond_types.items():
        edm.AddBond(b[0], b[1], t)

    m = edm.GetMol()
    if force_sanitize:
        Chem.SanitizeMol(m)
    else:
        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            warnings.warn(f"Cannot sanitize molecule {name}, because {str(e)}")
    m.AddConformer(conformer, assignId=False)

    m.SetProp("_Name", str(name))

    return m


def remove_metals(mol):
    """
    Check whether metals are in a pymatgen molecule. If yes, create a new Molecule
    with metals removed.
    Args:
        mol (pymatgen mol): molecule
        metals (dict): with metal specie are key and charge as value
    Returns:
        pymatgen mol
    """
    species = [str(s) for s in mol.species]

    if set(species).intersection(set(METALS.keys())):
        charge = mol.charge

        species = []
        coords = []
        properties = defaultdict(list)
        for site in mol:
            s = str(site.specie)
            if s in METALS:
                charge -= METALS[s]
            else:
                species.append(s)
                coords.append(site.coords)
                for k, v in site.properties:
                    properties[k].append(v)

        # do not provide spin_multiplicity, since we remove an atom
        mol = Molecule(species, coords, charge, site_properties=properties)

    return mol


def create_rdkit_mol_from_mol_graph(
    mol_graph,
    name=None,
    force_sanitize=False,
):
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.
    Done in the below steps:
    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge
    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str): name of the molecule
        force_sanitize (bool): whether to force sanitization of the rdkit mol
    Returns:
        m: rdkit Chem.Mol
        bond_types (dict): bond types assigned to the created rdkit mol
    """

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, attr in mol_graph.graph.edges.data()]

    # create babel mol without metals
    pmg_mol_no_metals = remove_metals(pymatgen_mol)
    adaptor = BabelMolAdaptor(pmg_mol_no_metals)
    ob_mol = adaptor.openbabel_mol

    # get babel bond order of mol without metals
    ob_bond_order = {}
    for bd in ob.OBMolBondIter(ob_mol):
        k = tuple(sorted([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()]))
        v = bd.GetBondOrder()
        ob_bond_order[k] = v

    # create bond type
    atom_idx_mapping = pymatgen_2_babel_atom_idx_map(pymatgen_mol, ob_mol)
    bond_types = {}

    for bd in bonds:
        try:
            ob_bond = [atom_idx_mapping[a] for a in bd]

            # atom not in ob mol
            if None in ob_bond:
                raise KeyError
            # atom in ob mol
            else:
                ob_bond = tuple(sorted(ob_bond))
                v = ob_bond_order[ob_bond]
                if v == 0:
                    tp = BondType.UNSPECIFIED
                elif v == 1:
                    tp = BondType.SINGLE
                elif v == 2:
                    tp = BondType.DOUBLE
                elif v == 3:
                    tp = BondType.TRIPLE
                elif v == 5:
                    tp = BondType.AROMATIC
                else:
                    raise RuntimeError(f"Got unexpected babel bond order: {v}")

        except KeyError:
            atom1_spec, atom2_spec = [species[a] for a in bd]

            if atom1_spec in metals and atom2_spec in metals:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in metals or atom2_spec in metals:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in metals:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds, metals)

    m = create_rdkit_mol(species, coords, bond_types, formal_charge, name, force_sanitize)

    return m, bond_types
