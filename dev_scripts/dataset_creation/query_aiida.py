# %%
from collections import OrderedDict, defaultdict
from functools import lru_cache

import pandas as pd
import yaml
from aiida import load_profile
from aiida.orm import CifData, Dict, Group, Node, WorkChainNode
from aiida.orm.querybuilder import QueryBuilder
from frozendict import frozendict

load_profile()
import simplejson

# pylint: disable=invalid-name

# label => projection
CIF_PROJECTIONS = OrderedDict(
    (
        ("mat_id", "label"),
        ("name_conventional", "extras.name_conventional"),
        ("class_mat", "extras.class_material"),
        ("lowest_abundance_value", "extras.lowest_abundance_value"),
        ("lowest_abundance_element", "extras.lowest_abundance_element"),
        ("element_price_kg", "extras.element_price_kg"),
    )
)


# %%
TAG_KEY = "test"

# %%
@lru_cache(maxsize=128)
def get_data_aiida_raw(quantitites, cif_projections=None):
    """Query the AiiDA database
    :param quantities: tuple of quantities to project
    :param cif_projections:  tuple of projections on the original CIF
    Note: this function addes a couple of other properties that are projected by default (CIF_PROJECTIONS).
    TODO: the group version needs to be better rationalized!
    """

    if cif_projections is None:
        cif_projections = list(CIF_PROJECTIONS.values())

    qb = QueryBuilder()
    qb.append(
        Group,
        filters={"label": {"like": r"curated-___\_%\_v_"}},
        tag="curated_groups",
        project="uuid",
    )
    qb.append(
        Node,
        project=cif_projections,
        filters={"extras.{}".format(TAG_KEY): "qmof_cif"},
        with_group="curated_groups",
    )

    for q in quantitites:
        qb.append(
            Dict,
            project=["attributes.{}".format(q["key"])],
            filters={"extras.{}".format(TAG_KEY): q["dict"]},
            with_group="curated_groups",
        )

    return qb.all()


# %%
@lru_cache(maxsize=128)
def get_data_aiida(quantitites, labels=None):
    """Query the AiiDA database
    Like get_data_aiida_raw but:
      * cleans 'None' values
      * returns dictionary
    :param quantities: tuple of quantities to project
    """
    results_dirty = get_data_aiida_raw(quantitites)

    if labels is None:
        if len(quantitites) == 2:
            labels = ("x", "y")
        elif len(quantitites) == 3:
            labels = ("x", "y", "z")
        else:
            raise ValueError(f"Unknown number of labels: {len(labels)}")

    # Clean the query from None values projections
    results = []
    for l in results_dirty:
        if None not in l:
            results.append(l)

    if not results:
        return {}

    result_zip = list(zip(*results))
    result_dict = {}

    for i, label in enumerate(CIF_PROJECTIONS):
        result_dict[label] = result_zip[i]

    for j, label in enumerate(labels):
        result_dict[label] = result_zip[len(CIF_PROJECTIONS) + j]

    return result_dict

    # mat_id, name, class_mat, abundance_val, abundance_el, price_kg, x, y = zip(*results)
    # return {
    #     'mat_id': mat_id,
    #     'name': name,
    #     'class_mat': class_mat,
    #     'abundance_value': list(map(float, abundance_val)),
    #     'abundance_element': abundance_el,
    #     'element_price_kg': price_kg,
    #     'x': list(map(float, x)),
    #     'y': list(map(float, y)),
    # }


@lru_cache(maxsize=8)
def get_isotherm_nodes(mat_id):
    """Query the AiiDA database, to get all the isotherms (Dict output of IsothermWorkChain, with GCMC calculations).
    Returning a dictionary like: {'co2: [Dict_0, Dict_1], 'h2': [Dict_0, Dict_1, Dict_2]}
    """

    # Get all the Isotherms
    qb = QueryBuilder()
    qb.append(
        Group, filters={"label": {"like": r"curated-___\_{}\_v_".format(mat_id)}}, tag="mat_group"
    )
    qb.append(
        Dict, filters={"extras.{}".format(TAG_KEY): {"like": r"isot\_%"}}, with_group="mat_group"
    )

    gas_dict = {}
    for x in qb.all():
        node = x[0]
        gas = node.extras[TAG_KEY].split("_")[1]
        if gas in gas_dict:
            gas_dict[gas].append(node.get_dict())
        else:
            gas_dict[gas] = [node.get_dict()]

    # Quite diry way to get all the isotherms from an IsothermMultiTemp
    qb = QueryBuilder()
    qb.append(
        Group, filters={"label": {"like": r"curated-___\_{}\_v_".format(mat_id)}}, tag="mat_group"
    )
    qb.append(
        Dict,
        filters={"extras.{}".format(TAG_KEY): {"like": r"isotmt\_%"}},
        with_group="mat_group",
        tag="isotmt_out",
        project=["extras.{}".format(TAG_KEY)],
    )
    qb.append(WorkChainNode, with_outgoing="isotmt_out", tag="isotmt_wc")
    qb.append(
        WorkChainNode,
        edge_filters={"label": {"like": "run_isotherm_%"}},
        with_incoming="isotmt_wc",
        tag="isot_wc",
    )
    qb.append(
        Dict, edge_filters={"label": "output_parameters"}, with_incoming="isot_wc", project=["*"]
    )

    for x in qb.all():
        node = x[1]
        gas = x[0].split("_")[1].upper()
        if gas in gas_dict:
            gas_dict[gas].append(node.get_dict())
        else:
            gas_dict[gas] = [node.get_dict()]

    return gas_dict


@lru_cache()
def get_mat_nodes_dict(mat_id):
    """Given a MAT_ID return a dictionary with all the tagged nodes for that material."""

    qb = QueryBuilder()
    qb.append(
        Group,
        filters={"label": {"like": r"curated-___\_{}\_v_".format(mat_id)}},
        tag="curated_groups",
    )
    qb.append(Node, filters={"extras": {"has_key": TAG_KEY}}, with_group="curated_groups")

    mat_nodes_dict = {}
    for q in qb.all():
        n = q[
            -1
        ]  # if more groups are present with different versions, take the last: QB sorts groups by label
        mat_nodes_dict[n.extras[TAG_KEY]] = n

    # add pricing info
    add_price(cif_node=mat_nodes_dict["orig_cif"])

    return mat_nodes_dict


def add_price_all():
    """Add price to extras of all CIF nodes."""
    # LOGGER.info("Computing prices for all materials. This is needed only once.")
    qb = QueryBuilder()
    qb.append(Node, filters={"extras.{}".format(TAG_KEY): "orig_cif"})
    results = qb.all()

    for result in results:
        cif_node = result[0]
        add_price(cif_node, force=True)


def add_price(cif_node, force=False):
    """Set element price information as node extras."""
    from mof_pricer import Price  # pylint: disable=import-outside-toplevel

    keys = [
        "element_abundance",
        "lowest_abundance_element",
        "lowest_abundance_value",
        "element_price_kg",
        "element_price_fractions_kg",
    ]
    if not force and all(k in cif_node.extras for k in keys):
        return

    atoms = cif_node.get_ase()
    price = Price(atoms)
    low_symbol, low_abundance = list(price.element_abundance.items())[0]
    cif_node.set_extra("lowest_abundance_element", low_symbol)
    cif_node.set_extra("lowest_abundance_value", low_abundance)
    cif_node.set_extra("element_abundance", list(price.element_abundance.items()))
    cif_node.set_extra("element_price_kg", price.kg_price)
    cif_node.set_extra("element_price_fractions_kg", list(price.element_price_fractions_kg.items()))


@lru_cache(maxsize=8)
def get_db_nodes_dict():
    """Given return a dictionary with all the curated materials having the material label as key, and a dict of
    curated nodes as value."""

    qb = QueryBuilder()
    qb.append(
        Group, filters={"label": {"like": r"curated-%"}}, tag="curated_groups", project=["label"]
    )
    qb.append(
        Node, filters={"extras": {"has_key": TAG_KEY}}, with_group="curated_groups", project=["*"]
    )

    db_nodes_dict = {}
    for q in qb.all():
        mat_label = q[0].split("_")[1]
        if mat_label not in db_nodes_dict:
            db_nodes_dict[mat_label] = {}
        n = q[1]
        db_nodes_dict[mat_label][n.extras[TAG_KEY]] = n

    return db_nodes_dict


# %%
with open("quantities.yml", "r") as handle:
    params = yaml.load(handle)

# %%
QUANTITIES = OrderedDict([(q["label"], frozendict(q)) for q in params])

# %%
APPLICATIONS = [
    "CO2 Henry coefficient",
    "CO2 adsorption energy",
    "N2 Henry coefficient",
    "N2 adsorption energy",
    "CO2 parasitic energy (coal)",
    "Gravimetric working capacity (coal)",
    "Volumetric working capacity (coal)",
    "CO2 parasitic energy (nat. gas)",
    "Gravimetric working capacity (nat. gas)",
    "Volumetric working capacity (nat. gas)",
    "Final CO2 purity (nat. gas)",
    "CH4 Henry coefficient",
    "CH4 adsorption energy",
    "Enthalphy of Adsorption @ 5.8 bar, 298K",
    "Enthalphy of Adsorption @ 65bar/298K",
    "Working capacity vol. (5.8-65bar/298K)",
    "Working capacity mol. (5.8-65bar/298K)",
    "Working capacity fract. (5.8-65bar/298K)",
    "Working capacity wt% (5.8-65bar/298K)",
    "O2 Henry coefficient",
    "O2 adsorption energy",
    "Enthalphy of Adsorption @ 5 bar, 298K",
    "Enthalphy of Adsorption @ 140bar/298K",
    "Working capacity vol. (5-140bar/298K)",
    "Working capacity mol. (5-140bar/298K)",
    "Working capacity fract. (5-140bar/298K)",
    "Working capacity wt% (5-140bar/298K)",
    "Xe Henry coefficient",
    "Xe adsorption energy",
    "Kr Henry coefficient",
    "Kr adsorption energy",
    "Xe/Kr selectivity @ 298K",
    "Working capacity g/L (5-100bar/298-198K)",
    "Working capacity g/L (5-100bar/77K)",
    "Working capacity g/L (1-100bar/77K)",
    "Working capacity wt% (5-100bar/298-198K)",
    "Working capacity wt% (5-100bar/77K)",
    "Working capacity wt% (1-100bar/77K)",
    "H2S Henry coefficient",
    "H2S adsorption energy",
    "H2O Henry coefficient",
    "H2O adsorption energy",
    "H2S/H2O selectivity @ 298K",
    "CH4/N2 selectivity @ 298K",
    #  'Geometric Void Fraction',
    #  'Accessible Pore Volume',
    #  'Accessible Surface Area',
    #  'Density',
    #  'Largest Free Sphere Diameter',
    #  'Largest Included Sphere Diameter'
]

# %%
NEW_COL_NAMES = [
    "CO2-henry_coefficient-mol/kg/Pa",
    "CO2-adsorption_energy-kJ/mol",
    "N2-henry_coefficient-mol/kg/Pa",
    "N2-adsorption_energy-kJ/mol",
    "CO2-parasitic_energy_coal-MJ/kg",
    "CO2-gravimetric_working_capacity_coal-kgCO2/kg",
    "CO2-volumetric_working_capacity_coal-kgCO2/m3",
    "CO2-parasitic_energy_nat_gas-MJ/kg",
    "CO2-gravimetric_working_capacity_nat_gas-kgCO2/kg",
    "CO2-volumetric_working_capacity_nat_gas-kgCO2/m3",
    "CO2-final_purity_nat_gas-mol/mol",
    "CH4-henry_coefficient-mol/kg/Pa",
    "CH4-adsorption_energy-kJ/mol",
    "CH4-enthalphy_of_adsorption_5.8_bar_298_K-kJ/mol",
    "CH4-enthalphy_of_adsorption_65_bar_298_K-kJ/mol",
    "CH4-working_capacity_vol_5.8_to_65_bar_298_K-cm3_STP/cm3",
    "CH4-working_capacity_mol_5.8_to_65_bar_298_K-mol/kg",
    "CH4-working_capacity_fract_5.8_to_65_bar_298_K-",
    "CH4-working_capacity_wt%_5.8_to_65_bar_298_K-g/g*100",
    "O2-henry_coefficient-mol/kg/Pa",
    "O2-adsorption_energy-kJ/mol",
    "O2-enthalphy_of_adsorption_5_bar_298_K-kJ/mol",
    "O2-enthalphy_of_adsorption_140_bar_298_K-kJ/mol",
    "O2-working_capacity_vol_5_to_140_bar_298_K-cm3_STP/cm3",
    "O2-working_capacity_mol_5_to_140_bar_298_K-mol/kg",
    "O2-working_capacity_fract_5_to_140_bar_298_K-",
    "O2-working_capacity_wt%_5_to_140_bar_298_K-g/g*100",
    "Xe-henry_coefficient-mol/kg/Pa",
    "Xe-adsorption_energy-kJ/mol",
    "Kr-henry_coefficient-mol/kg/Pa",
    "Kr-adsorption_energy-kJ/mol",
    "Xe/Kr-selectivity_298_K-",
    "H2-working_capacity_5_to_100_bar_298_to_198_K-g/L",
    "H2-working_capacity_5_to_100_bar_77_K-g/L",
    "H2-working_capacity_1_to_100_bar_77_K-g/L",
    "H2-working_capacity_wt%_5_to_100_bar_298_to_198_K-g/g100",
    "H2-working_capacity_wt%_5_to_100_bar_77_K-g/g100",
    "H2-working_capacity_wt%_1_to_100_bar_77_K-g/g100",
    "H2S-henry_coefficient-mol/kg/Pa",
    "H2S-adsorption_energy-kJ/mol",
    "H2O-henry_coefficient-mol/kg/Pa",
    "H2O-adsorption_energy-kJ/mol",
    "H2S/H2O-selectivity_298_K-",
    "CH4/N2-selectivity_298_K-",
]

# %%
for o, n in zip(APPLICATIONS, NEW_COL_NAMES):
    print(o, n)

# %%
q_list = tuple([QUANTITIES[label] for label in APPLICATIONS])


# %%
results = get_data_aiida_raw(q_list, cif_projections=("label", "extras.name_conventional"))

# %%
uuids = [r[0] for r in results]

# %%
res = pd.DataFrame(results, columns=["uuid", "id", "name"] + list(APPLICATIONS))

# %%
len(res)

# %%
res.dropna(subset=["H2S Henry coefficient"])

# %%
res.to_csv("qmof_data.csv", index=False)

# %%
res["qmof-id"] = res["name"].apply(lambda x: x.split("/")[-1].replace(".cif", ""))

# %%
rename_dict = dict(zip(APPLICATIONS, NEW_COL_NAMES))

# %%
res.rename(columns=rename_dict, inplace=True)

# %%
res.to_csv("qmof_data.csv", index=False)

# %%
import numpy as np

res_dict = {}

for i, row in res.iterrows():
    subdict = defaultdict(dict)

    name = row["qmof-id"]

    for col in NEW_COL_NAMES:
        parts = col.split("-")
        gas = parts[0]
        prop = parts[1]
        unit = parts[2]
        try:
            value = float(row[col])
        except TypeError:
            value = np.nan
        subdict[gas][prop] = value
        subdict[gas][prop + "_unit"] = unit

    res_dict[name] = subdict

# %%
isotherms = {}

for i, row in res.iterrows():
    isotherms[row["qmof-id"]] = get_isotherm_nodes(row["id"])

# %%
for mof, subdict in res_dict.items():
    for gas, gas_dict in subdict.items():
        try:
            gas_dict.update(isotherms[mof][gas.lower()][0])
        except KeyError:
            pass

# %%
with open("raw.json", "w") as f:
    simplejson.dump(res_dict, f, ignore_nan=True)
