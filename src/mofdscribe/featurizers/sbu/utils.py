# -*- coding: utf-8 -*-
"""
Routines for the conversion between pymatgen molecules and RDKit.

Mostly copied from
https://github.com/mjwen/bondnet/blob/b719bd85235012c567298cb0da81ef113e3744bc/bondnet/core/rdmol.py#L265
which is licensed under CDDLv1.0.

COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0

1. Definitions.

1.1. "Contributor" means each individual or entity that
creates or contributes to the creation of Modifications.

1.2. "Contributor Version" means the combination of the
Original Software, prior Modifications used by a
Contributor (if any), and the Modifications made by that
particular Contributor.

1.3. "Covered Software" means (a) the Original Software, or
(b) Modifications, or (c) the combination of files
containing Original Software with files containing
Modifications, in each case including portions thereof.

1.4. "Executable" means the Covered Software in any form
other than Source Code.

1.5. "Initial Developer" means the individual or entity
that first makes Original Software available under this
License.

1.6. "Larger Work" means a work which combines Covered
Software or portions thereof with code not governed by the
terms of this License.

1.7. "License" means this document.

1.8. "Licensable" means having the right to grant, to the
maximum extent possible, whether at the time of the initial
grant or subsequently acquired, any and all of the rights
conveyed herein.

1.9. "Modifications" means the Source Code and Executable
form of any of the following:

A. Any file that results from an addition to,
deletion from or modification of the contents of a
file containing Original Software or previous
Modifications;

B. Any new file that contains any part of the
Original Software or previous Modification; or

C. Any new file that is contributed or otherwise made
available under the terms of this License.

1.10. "Original Software" means the Source Code and
Executable form of computer software code that is
originally released under this License.

1.11. "Patent Claims" means any patent claim(s), now owned
or hereafter acquired, including without limitation,
method, process, and apparatus claims, in any patent
Licensable by grantor.

1.12. "Source Code" means (a) the common form of computer
software code in which modifications are made and (b)
associated documentation included in or with such code.

1.13. "You" (or "Your") means an individual or a legal
entity exercising rights under, and complying with all of
the terms of, this License. For legal entities, "You"
includes any entity which controls, is controlled by, or is
under common control with You. For purposes of this
definition, "control" means (a) the power, direct or
indirect, to cause the direction or management of such
entity, whether by contract or otherwise, or (b) ownership
of more than fifty percent (50%) of the outstanding shares
or beneficial ownership of such entity.

2. License Grants.

2.1. The Initial Developer Grant.

Conditioned upon Your compliance with Section 3.1 below and
subject to third party intellectual property claims, the
Initial Developer hereby grants You a world-wide,
royalty-free, non-exclusive license:

(a) under intellectual property rights (other than
patent or trademark) Licensable by Initial Developer,
to use, reproduce, modify, display, perform,
sublicense and distribute the Original Software (or
portions thereof), with or without Modifications,
and/or as part of a Larger Work; and

(b) under Patent Claims infringed by the making,
using or selling of Original Software, to make, have
made, use, practice, sell, and offer for sale, and/or
otherwise dispose of the Original Software (or
portions thereof).

(c) The licenses granted in Sections 2.1(a) and (b)
are effective on the date Initial Developer first
distributes or otherwise makes the Original Software
available to a third party under the terms of this
License.

(d) Notwithstanding Section 2.1(b) above, no patent
license is granted: (1) for code that You delete from
the Original Software, or (2) for infringements
caused by: (i) the modification of the Original
Software, or (ii) the combination of the Original
Software with other software or devices.

2.2. Contributor Grant.

Conditioned upon Your compliance with Section 3.1 below and
subject to third party intellectual property claims, each
Contributor hereby grants You a world-wide, royalty-free,
non-exclusive license:

(a) under intellectual property rights (other than
patent or trademark) Licensable by Contributor to
use, reproduce, modify, display, perform, sublicense
and distribute the Modifications created by such
Contributor (or portions thereof), either on an
unmodified basis, with other Modifications, as
Covered Software and/or as part of a Larger Work; and

(b) under Patent Claims infringed by the making,
using, or selling of Modifications made by that
Contributor either alone and/or in combination with
its Contributor Version (or portions of such
combination), to make, use, sell, offer for sale,
have made, and/or otherwise dispose of: (1)
Modifications made by that Contributor (or portions
thereof); and (2) the combination of Modifications
made by that Contributor with its Contributor Version
(or portions of such combination).

(c) The licenses granted in Sections 2.2(a) and
2.2(b) are effective on the date Contributor first
distributes or otherwise makes the Modifications
available to a third party.

(d) Notwithstanding Section 2.2(b) above, no patent
license is granted: (1) for any code that Contributor
has deleted from the Contributor Version; (2) for
infringements caused by: (i) third party
modifications of Contributor Version, or (ii) the
combination of Modifications made by that Contributor
with other software (except as part of the
Contributor Version) or other devices; or (3) under
Patent Claims infringed by Covered Software in the
absence of Modifications made by that Contributor.

3. Distribution Obligations.

3.1. Availability of Source Code.

Any Covered Software that You distribute or otherwise make
available in Executable form must also be made available in
Source Code form and that Source Code form must be
distributed only under the terms of this License. You must
include a copy of this License with every copy of the
Source Code form of the Covered Software You distribute or
otherwise make available. You must inform recipients of any
such Covered Software in Executable form as to how they can
obtain such Covered Software in Source Code form in a
reasonable manner on or through a medium customarily used
for software exchange.

3.2. Modifications.

The Modifications that You create or to which You
contribute are governed by the terms of this License. You
represent that You believe Your Modifications are Your
original creation(s) and/or You have sufficient rights to
grant the rights conveyed by this License.

3.3. Required Notices.

You must include a notice in each of Your Modifications
that identifies You as the Contributor of the Modification.
You may not remove or alter any copyright, patent or
trademark notices contained within the Covered Software, or
any notices of licensing or any descriptive text giving
attribution to any Contributor or the Initial Developer.

3.4. Application of Additional Terms.

You may not offer or impose any terms on any Covered
Software in Source Code form that alters or restricts the
applicable version of this License or the recipients'
rights hereunder. You may choose to offer, and to charge a
fee for, warranty, support, indemnity or liability
obligations to one or more recipients of Covered Software.
However, you may do so only on Your own behalf, and not on
behalf of the Initial Developer or any Contributor. You
must make it absolutely clear that any such warranty,
support, indemnity or liability obligation is offered by
You alone, and You hereby agree to indemnify the Initial
Developer and every Contributor for any liability incurred
by the Initial Developer or such Contributor as a result of
warranty, support, indemnity or liability terms You offer.

3.5. Distribution of Executable Versions.

You may distribute the Executable form of the Covered
Software under the terms of this License or under the terms
of a license of Your choice, which may contain terms
different from this License, provided that You are in
compliance with the terms of this License and that the
license for the Executable form does not attempt to limit
or alter the recipient's rights in the Source Code form
from the rights set forth in this License. If You
distribute the Covered Software in Executable form under a
different license, You must make it absolutely clear that
any terms which differ from this License are offered by You
alone, not by the Initial Developer or Contributor. You
hereby agree to indemnify the Initial Developer and every
Contributor for any liability incurred by the Initial
Developer or such Contributor as a result of any such terms
You offer.

3.6. Larger Works.

You may create a Larger Work by combining Covered Software
with other code not governed by the terms of this License
and distribute the Larger Work as a single product. In such
a case, You must make sure the requirements of this License
are fulfilled for the Covered Software.

4. Versions of the License.

4.1. New Versions.

Sun Microsystems, Inc. is the initial license steward and
may publish revised and/or new versions of this License
from time to time. Each version will be given a
distinguishing version number. Except as provided in
Section 4.3, no one other than the license steward has the
right to modify this License.

4.2. Effect of New Versions.

You may always continue to use, distribute or otherwise
make the Covered Software available under the terms of the
version of the License under which You originally received
the Covered Software. If the Initial Developer includes a
notice in the Original Software prohibiting it from being
distributed or otherwise made available under any
subsequent version of the License, You must distribute and
make the Covered Software available under the terms of the
version of the License under which You originally received
the Covered Software. Otherwise, You may also choose to
use, distribute or otherwise make the Covered Software
available under the terms of any subsequent version of the
License published by the license steward.

4.3. Modified Versions.

When You are an Initial Developer and You want to create a
new license for Your Original Software, You may create and
use a modified version of this License if You: (a) rename
the license and remove any references to the name of the
license steward (except to note that the license differs
from this License); and (b) otherwise make it clear that
the license contains terms which differ from this License.

5. DISCLAIMER OF WARRANTY.

COVERED SOFTWARE IS PROVIDED UNDER THIS LICENSE ON AN "AS IS"
BASIS, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
INCLUDING, WITHOUT LIMITATION, WARRANTIES THAT THE COVERED
SOFTWARE IS FREE OF DEFECTS, MERCHANTABLE, FIT FOR A PARTICULAR
PURPOSE OR NON-INFRINGING. THE ENTIRE RISK AS TO THE QUALITY AND
PERFORMANCE OF THE COVERED SOFTWARE IS WITH YOU. SHOULD ANY
COVERED SOFTWARE PROVE DEFECTIVE IN ANY RESPECT, YOU (NOT THE
INITIAL DEVELOPER OR ANY OTHER CONTRIBUTOR) ASSUME THE COST OF
ANY NECESSARY SERVICING, REPAIR OR CORRECTION. THIS DISCLAIMER OF
WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS LICENSE. NO USE OF
ANY COVERED SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER THIS
DISCLAIMER.

6. TERMINATION.

6.1. This License and the rights granted hereunder will
terminate automatically if You fail to comply with terms
herein and fail to cure such breach within 30 days of
becoming aware of the breach. Provisions which, by their
nature, must remain in effect beyond the termination of
this License shall survive.

6.2. If You assert a patent infringement claim (excluding
declaratory judgment actions) against Initial Developer or
a Contributor (the Initial Developer or Contributor against
whom You assert such claim is referred to as "Participant")
alleging that the Participant Software (meaning the
Contributor Version where the Participant is a Contributor
or the Original Software where the Participant is the
Initial Developer) directly or indirectly infringes any
patent, then any and all rights granted directly or
indirectly to You by such Participant, the Initial
Developer (if the Initial Developer is not the Participant)
and all Contributors under Sections 2.1 and/or 2.2 of this
License shall, upon 60 days notice from Participant
terminate prospectively and automatically at the expiration
of such 60 day notice period, unless if within such 60 day
period You withdraw Your claim with respect to the
Participant Software against such Participant either
unilaterally or pursuant to a written agreement with
Participant.

6.3. In the event of termination under Sections 6.1 or 6.2
above, all end user licenses that have been validly granted
by You or any distributor hereunder prior to termination
(excluding licenses granted to You by any distributor)
shall survive termination.

7. LIMITATION OF LIABILITY.

UNDER NO CIRCUMSTANCES AND UNDER NO LEGAL THEORY, WHETHER TORT
(INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE, SHALL YOU, THE
INITIAL DEVELOPER, ANY OTHER CONTRIBUTOR, OR ANY DISTRIBUTOR OF
COVERED SOFTWARE, OR ANY SUPPLIER OF ANY OF SUCH PARTIES, BE
LIABLE TO ANY PERSON FOR ANY INDIRECT, SPECIAL, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES OF ANY CHARACTER INCLUDING, WITHOUT
LIMITATION, DAMAGES FOR LOST PROFITS, LOSS OF GOODWILL, WORK
STOPPAGE, COMPUTER FAILURE OR MALFUNCTION, OR ANY AND ALL OTHER
COMMERCIAL DAMAGES OR LOSSES, EVEN IF SUCH PARTY SHALL HAVE BEEN
INFORMED OF THE POSSIBILITY OF SUCH DAMAGES. THIS LIMITATION OF
LIABILITY SHALL NOT APPLY TO LIABILITY FOR DEATH OR PERSONAL
INJURY RESULTING FROM SUCH PARTY'S NEGLIGENCE TO THE EXTENT
APPLICABLE LAW PROHIBITS SUCH LIMITATION. SOME JURISDICTIONS DO
NOT ALLOW THE EXCLUSION OR LIMITATION OF INCIDENTAL OR
CONSEQUENTIAL DAMAGES, SO THIS EXCLUSION AND LIMITATION MAY NOT
APPLY TO YOU.

8. U.S. GOVERNMENT END USERS.

The Covered Software is a "commercial item," as that term is
defined in 48 C.F.R. 2.101 (Oct. 1995), consisting of "commercial
computer software" (as that term is defined at 48 C.F.R.
252.227-7014(a)(1)) and "commercial computer software
documentation" as such terms are used in 48 C.F.R. 12.212 (Sept.
1995). Consistent with 48 C.F.R. 12.212 and 48 C.F.R. 227.7202-1
through 227.7202-4 (June 1995), all U.S. Government End Users
acquire Covered Software with only those rights set forth herein.
This U.S. Government Rights clause is in lieu of, and supersedes,
any other FAR, DFAR, or other clause or provision that addresses
Government rights in computer software under this License.

9. MISCELLANEOUS.

This License represents the complete agreement concerning subject
matter hereof. If any provision of this License is held to be
unenforceable, such provision shall be reformed only to the
extent necessary to make it enforceable. This License shall be
governed by the law of the jurisdiction specified in a notice
contained within the Original Software (except to the extent
applicable law, if any, provides otherwise), excluding such
jurisdiction's conflict-of-law provisions. Any litigation
relating to this License shall be subject to the jurisdiction of
the courts located in the jurisdiction and venue specified in a
notice contained within the Original Software, with the losing
party responsible for costs, including, without limitation, court
costs and reasonable attorneys' fees and expenses. The
application of the United Nations Convention on Contracts for the
International Sale of Goods is expressly excluded. Any law or
regulation which provides that the language of a contract shall
be construed against the drafter shall not apply to this License.
You agree that You alone are responsible for compliance with the
United States export administration regulations (and the export
control laws and regulation of any other countries) when You use,
distribute or otherwise make available any Covered Software.

10. RESPONSIBILITY FOR CLAIMS.

As between Initial Developer and the Contributors, each party is
responsible for claims and damages arising, directly or
indirectly, out of its utilization of rights under this License
and You agree to work with Initial Developer and Contributors to
distribute such responsibility on an equitable basis. Nothing
herein is intended or shall be deemed to constitute any admission
of liability.
"""
from collections import defaultdict
from ctypes import Structure
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike
from openbabel import openbabel as ob
from pymatgen.core import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Geometry import Point3D

METALS = {
    # first group
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    # second group
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Al": 3,
    "Sc": 1,
    "Ti": 4,
    "V": 2,
    "Cr": 2,
    "Mn": 2,
    "Fe": 2,
    "Co": 2,
    "Ni": 2,
    "Cu": 2,
    "Zn": 2,
    "Zr": 4,
    # Lanthanides and Actinides
    "La": 3,
    "Ce": 3,
    "Pr": 3,
    "Nd": 3,
    "Pm": 3,
    "Sm": 3,
    "Eu": 3,
    "Gd": 3,
    "Tb": 3,
    "Dy": 3,
    "Ho": 3,
    "Er": 3,
    "Tm": 3,
    "Yb": 3,
    "Lu": 3,
    "Hf": 4,
    "Ta": 4,
    "U": 4,
    "W": 4,
    "Re": 4,
    "Os": 4,
    "Ir": 4,
    "Pt": 4,
}


def pymatgen_2_babel_atom_idx_map(pmg_mol: Molecule, ob_mol: ob.OBMol) -> Dict[int, int]:
    """
    Create an atom index mapping between pymatgen mol and openbabel mol.

    This does not require pymatgen mol and ob mol has the same number of atoms.
    But ob_mol can have smaller number of atoms.

    Args:
        pmg_mol (Molecule): pymatgen molecule
        ob_mol (ob.OBMol): openbabel molecule

    Returns:
        Dict[int, int]: with atom index in pymatgen mol as key and atom index in babel mol as
            value. Value is `None` if there is not corresponding atom in babel.

    Raises:
        RuntimeError: if mapping is not possible.
    """
    pmg_coords = pmg_mol.cart_coords
    ob_coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(ob_mol)]
    ob_index = [a.GetIdx() for a in ob.OBMolAtomIter(ob_mol)]

    mapping = {i: None for i in range(len(pmg_coords))}

    for idx, oc in zip(ob_index, ob_coords):
        for i, gc in enumerate(pmg_coords):
            if np.allclose(oc, gc):
                mapping[i] = idx
                break
        else:
            raise RuntimeError("Cannot create atom index mapping pymatgen and ob mols")

    return mapping


def create_rdkit_mol(
    species: Iterable[str],
    coords: ArrayLike,
    bond_types: Dict[Tuple[int, int], Chem.rdchem.BondType],
    formal_charge: Optional[Iterable[float]] = None,
    name: Optional[str] = None,
    force_sanitize: bool = True,
):
    """
    Create a rdkit mol from scratch.

    Followed: https://sourceforge.net/p/rdkit/mailman/message/36474923/

    Args:
        species (list): species str of each molecule
        coords (ArrayLike): positions of atoms
        bond_types (Dict[Tuple[int, int]): with bond indices (2 tuple) as key and bond type
            (e.g. Chem.rdchem.BondType.DOUBLE) as value
        formal_charge (list, optional): formal charge of each atom.
            Defaults to None.
        name (str, optional): name of the molecule.
            Defaults to None.
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
            logger.warning(f"Cannot sanitize molecule {name}, because {str(e)}")
    m.AddConformer(conformer, assignId=False)

    m.SetProp("_Name", str(name))

    return m


def adjust_formal_charge(species: Iterable[str], bonds: Iterable[Tuple[int, int]]):
    """
    Adjust formal charge of metal atoms.

    Args:
        species (Iterable[str]): species string of atoms
        bonds (Iterable[Tuple[int, int]]): 2-tuple index of bonds

    Returns:
        list: formal charge of atoms. None for non metal atoms.

    ToDo:
        - use something like oximachine to guess oxidation state
    """
    # initialize formal charge first so that atom does not form any bond has its formal
    # charge set
    formal_charge = [METALS[s] if s in METALS else None for s in species]

    # atom_idx: idx of atoms in the molecule
    # num_bonds: number of bonds the atom forms
    atom_idx, num_bonds = np.unique(bonds, return_counts=True)
    for i, ct in zip(atom_idx, num_bonds):
        s = species[i]
        if s in METALS:
            formal_charge[i] = int(formal_charge[i] - ct)

    return formal_charge


def remove_metals(mol: Molecule):
    """
    Check whether metals are in a pymatgen molecule. If yes, create a new Molecule with metals removed.

    Args:
        mol (Molecule): pymatgen molecule

    Returns:
        pymatgen mol
    """
    species = [str(s) for s in mol.species]

    #  metals (dict): with metal specie are key and charge as value

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
    name: Optional[str] = None,
    force_sanitize: bool = False,
) -> Chem.Mol:
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.

    Done in the below steps:
    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge

    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str, optional): name of the molecule.
            Defaults to None.
        force_sanitize (bool): whether to force sanitization of the rdkit mol

    Returns:
        mol (Chem.Mol): rdkit Chem.Mol

    Raises:
        RuntimeError: if it finds and unexpected bond type or a bond between
            two metals
    """
    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, _ in mol_graph.graph.edges.data()]

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

            if atom1_spec in METALS and atom2_spec in METALS:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in METALS or atom2_spec in METALS:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in METALS:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds)

    m = create_rdkit_mol(species, coords, bond_types, formal_charge, name, force_sanitize)

    return m


def boxed_molecule(molecule: Molecule) -> Structure:
    """
    Create a box molecule from a pymatgen molecule.

    Args:
        molecule (Molecule): pymatgen molecule

    Returns:
        structure (Structure): molecule in a box
    """
    max_dist = molecule.distance_matrix.max()

    return molecule.get_boxed_structure(a=max_dist, b=max_dist, c=max_dist)
