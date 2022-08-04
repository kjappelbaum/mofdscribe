# -*- coding: utf-8 -*-
"""Basic raspa output parser.

code from https://github.com/lsmo-epfl/aiida-raspa/blob/develop/aiida_raspa/utils/base_parser.py

MIT License
Copyright 2018-2022, LABORATORY OF MOLECULAR SIMULATION (LSMO),
ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import re
from math import isinf, isnan

float_base = float


def float(number):
    number = float_base(number)
    return number if not any((isnan(number), isinf(number))) else None


KELVIN_TO_KJ_PER_MOL = float(8.314464919 / 1000.0)  # exactly the same as Raspa

# manage block of the first type
# --------------------------------------------------------------------------------------------
BLOCK_1_LIST = [
    ("Average Volume:", "cell_volume", (1, 2, 4), 0),
    ("Average Density:", "adsorbate_density", (1, 2, 4), 0),
    # ("Average Heat Capacity", "framework_heat_capacity", (1, 2, 4), 0), # misleading property!
    ("Enthalpy of adsorption:", "enthalpy_of_adsorption", (1, 4, 3), 4),
    ("Tail-correction energy:", "tail_correction_energy", (1, 2, 4), 0)
    # ("Total energy:$", "total_energy", (1, 2, 4), 0), # not important property!
]

# block of box properties.
BOX_PROP_LIST = [
    ("Average Box-lengths:", "box"),
]


def parse_block1(flines, result_dict, prop, value=1, unit=2, dev=4):
    """Parse block.

    Parses blocks that look as follows::

        Average Volume:
        =================
            Block[ 0]        12025.61229 [A^3]
            Block[ 1]        12025.61229 [A^3]
            Block[ 2]        12025.61229 [A^3]
            Block[ 3]        12025.61229 [A^3]
            Block[ 4]        12025.61229 [A^3]
            ------------------------------------------------------------------------------
            Average          12025.61229 [A^3] +/-            0.00000 [A^3]

    Args:
        flines (list): list of lines to parse
        result_dict (dict): dictionary to store results
        prop (str): property to store
        value (int): index of value in line
        unit (int): index of unit in line
        dev (int): index of deviation in line
    """
    for line in flines:
        if "Average" in line:
            result_dict[prop + "_average"] = float(line.split()[value])
            result_dict[prop + "_unit"] = re.sub(r"[{}()\[\]]", "", line.split()[unit])
            result_dict[prop + "_dev"] = float(line.split()[dev])
            break


# manage energy reading
# --------------------------------------------------------------------------------------------
ENERGY_CURRENT_LIST = [
    ("Host/Adsorbate energy:", "host/ads", "tot"),
    ("Host/Adsorbate VDW energy:", "host/ads", "vdw"),
    ("Host/Adsorbate Coulomb energy:", "host/ads", "coulomb"),
    ("Adsorbate/Adsorbate energy:", "ads/ads", "tot"),
    ("Adsorbate/Adsorbate VDW energy:", "ads/ads", "vdw"),
    ("Adsorbate/Adsorbate Coulomb energy:", "ads/ads", "coulomb"),
]

ENERGY_AVERAGE_LIST = [
    ("Average Adsorbate-Adsorbate energy:", "ads/ads"),
    ("Average Host-Adsorbate energy:", "host/ads"),
]


def parse_block_energy(flines, res_dict, prop):
    """Parse energy block.

    Parse block that looks as follows::

        Average Adsorbate-Adsorbate energy:
        ===================================
            Block[ 0] -443.23204         Van der Waals: -443.23204         Coulomb: 0.00000            [K]
            Block[ 1] -588.20205         Van der Waals: -588.20205         Coulomb: 0.00000            [K]
            Block[ 2] -538.43355         Van der Waals: -538.43355         Coulomb: 0.00000            [K]
            Block[ 3] -530.00960         Van der Waals: -530.00960         Coulomb: 0.00000            [K]
            Block[ 4] -484.15106         Van der Waals: -484.15106         Coulomb: 0.00000            [K]
            ------------------------------------------------------------------------------
            Average   -516.80566         Van der Waals: -516.805659        Coulomb: 0.00000            [K]
                  +/- 98.86943                      +/- 98.869430               +/- 0.00000            [K]

    Args:
        flines (list): list of lines to parse
        res_dict (dict): dictionary to store results
        prop (str): property to store
    """
    for line in flines:
        if "Average" in line:
            res_dict["energy_{}_tot_average".format(prop)] = (
                float(line.split()[1]) * KELVIN_TO_KJ_PER_MOL
            )
            res_dict["energy_{}_vdw_average".format(prop)] = (
                float(line.split()[5]) * KELVIN_TO_KJ_PER_MOL
            )
            res_dict["energy_{}_coulomb_average".format(prop)] = (
                float(line.split()[7]) * KELVIN_TO_KJ_PER_MOL
            )
        if "+/-" in line:
            res_dict["energy_{}_tot_dev".format(prop)] = (
                float(line.split()[1]) * KELVIN_TO_KJ_PER_MOL
            )
            res_dict["energy_{}_vdw_dev".format(prop)] = (
                float(line.split()[3]) * KELVIN_TO_KJ_PER_MOL
            )
            res_dict["energy_{}_coulomb_dev".format(prop)] = (
                float(line.split()[5]) * KELVIN_TO_KJ_PER_MOL
            )
            return


# manage lines with components
LINES_WITH_COMPONENT_LIST = [
    (" Average Widom Rosenbluth-weight:", "widom_rosenbluth_factor"),
    (" Average chemical potential: ", "chemical_potential"),
    (" Average Henry coefficient: ", "henry_coefficient"),
    (" Average  <U_gh>_1-<U_h>_0:", "adsorption_energy_widom"),
]


def parse_lines_with_component(res_components, components, line, prop):
    """Parse lines that contain components"""
    for i, component in enumerate(components):
        if "[" + component + "]" in line:
            words = line.split()
            res_components[i][prop + "_unit"] = re.sub(r"[{}()\[\]]", "", words[-1])
            res_components[i][prop + "_dev"] = float(words[-2])
            res_components[i][prop + "_average"] = float(words[-4])


def parse_base_output(output_abs_path, system_name, ncomponents):  # noqa: C901
    """Parse RASPA output file: it is divided in different parts, whose start/end is carefully documented."""
    warnings = []
    res_per_component = []
    for _ in range(ncomponents):
        res_per_component.append({})
    result_dict = {"exceeded_walltime": False}

    with open(output_abs_path, "r") as fobj:

        # 1st parsing part: input settings
        # --------------------------------
        # from: start of file
        # to: "Current (initial full energy) Energy Status"

        icomponent = 0
        component_names = []
        res_cmp = res_per_component[0]
        for line in fobj:
            if "Component" in line and "molecule)" in line:
                component_names.append(line.split()[2][1:-1])
                if "(Adsorbate" in line:
                    res_cmp["molecule_type"] = "adsorbate"
                elif "(Cation" in line:
                    res_cmp["molecule_type"] = "cation"

            # Consider to change it with parse_line()
            if "Conversion factor molecules/unit cell -> mol/kg:" in line:
                res_cmp["conversion_factor_molec_uc_to_mol_kg"] = float(line.split()[6])
                res_cmp["conversion_factor_molec_uc_to_mol_kg_unit"] = "(mol/kg)/(molec/uc)"
            # this line was corrected in Raspa's commit c1ad4de (Nov19), since "gr/gr" should read "mg/g"
            if (
                "Conversion factor molecules/unit cell -> gr/gr:" in line
                or "Conversion factor molecules/unit cell -> mg/g:" in line
            ):
                res_cmp["conversion_factor_molec_uc_to_mg_g"] = float(line.split()[6])
                res_cmp["conversion_factor_molec_uc_to_mg_g_unit"] = "(mg/g)/(molec/uc)"
            if "Conversion factor molecules/unit cell -> cm^3 STP/gr:" in line:
                res_cmp["conversion_factor_molec_uc_to_cm3stp_gr"] = float(line.split()[7])
                res_cmp["conversion_factor_molec_uc_to_cm3stp_gr_unit"] = "(cm^3_STP/gr)/(molec/uc)"
            if "Conversion factor molecules/unit cell -> cm^3 STP/cm^3:" in line:
                res_cmp["conversion_factor_molec_uc_to_cm3stp_cm3"] = float(line.split()[7])
                res_cmp[
                    "conversion_factor_molec_uc_to_cm3stp_cm3_unit"
                ] = "(cm^3_STP/cm^3)/(molec/uc)"
            if "MolFraction:" in line:
                res_cmp["mol_fraction"] = float(line.split()[1])
                res_cmp["mol_fraction_unit"] = "-"
            if "Partial pressure:" in line:
                res_cmp["partial_pressure"] = float(line.split()[2])
                res_cmp["partial_pressure_unit"] = "Pa"
            if "Partial fugacity:" in line:
                res_cmp["partial_fugacity"] = float(line.split()[2])
                res_cmp["partial_fugacity_unit"] = "Pa"
                icomponent += 1
                if icomponent < ncomponents:
                    res_cmp = res_per_component[icomponent]
            if "Framework Density" in line:
                result_dict["framework_density"] = line.split()[2]
                result_dict["framework_density_unit"] = re.sub(r"[{}()\[\]]", "", line.split()[3])
            if "Current (initial full energy) Energy Status" in line:
                break

        # 2nd parsing part: initial and final configurations
        # --------------------------------------------------
        # from: "Current (initial full energy) Energy Status"
        # to: "Average properties of the system"

        reading = "initial"
        result_dict["energy_unit"] = "kJ/mol"

        for line in fobj:
            # Understand if it is the initial or final "Current Energy Status" section
            if "Current (full final energy) Energy Status" in line:
                reading = "final"

            # Read the entries of "Current Energy Status" section
            if reading:
                for parse in ENERGY_CURRENT_LIST:
                    if parse[0] in line:
                        result_dict["energy_{}_{}_{}".format(parse[1], parse[2], reading)] = (
                            float(line.split()[-1]) * KELVIN_TO_KJ_PER_MOL
                        )
                        if parse[1] == "ads/ads" and parse[2] == "coulomb":
                            reading = None

            if "Average properties of the system" in line:
                break

        # 3rd parsing part: average system properties
        # --------------------------------------------------
        # from: "Average properties of the system"
        # to: "Number of molecules"

        for line in fobj:
            for parse in BLOCK_1_LIST:
                if parse[0] in line:
                    parse_block1(fobj, result_dict, parse[1], *parse[2])
                    # I assume here that properties per component are present furhter in the output file.
                    # so I need to skip some lines:
                    skip_nlines_after = parse[3]
                    while skip_nlines_after > 0:
                        line = next(fobj)
                        skip_nlines_after -= 1
                    for i, cmpnt in enumerate(component_names):
                        # The order of properties per molecule is the same as the order of molecules in the
                        # input file. So if component name was not found in the next line, I break the loop
                        # immidiately as there is no reason to continue it
                        line = next(fobj)
                        if cmpnt in line:
                            parse_block1(fobj, res_per_component[i], parse[1], *parse[2])
                        else:
                            break
                        skip_nlines_after = parse[3]
                        while skip_nlines_after > 0:
                            line = next(fobj)
                            skip_nlines_after -= 1

                    continue  # no need to perform further checks, propperty has been found already
            for parse in ENERGY_AVERAGE_LIST:
                if parse[0] in line:
                    parse_block_energy(fobj, result_dict, prop=parse[1])
                    continue  # no need to perform further checks, propperty has been found already
            for parse in BOX_PROP_LIST:
                if parse[0] in line:
                    # parse three cell vectors
                    parse_block1(fobj, result_dict, prop="box_ax", value=2, unit=3, dev=5)
                    parse_block1(fobj, result_dict, prop="box_by", value=2, unit=3, dev=5)
                    parse_block1(fobj, result_dict, prop="box_cz", value=2, unit=3, dev=5)
                    # parsee angles between the cell vectors
                    parse_block1(fobj, result_dict, prop="box_alpha", value=3, unit=4, dev=6)
                    parse_block1(fobj, result_dict, prop="box_beta", value=3, unit=4, dev=6)
                    parse_block1(fobj, result_dict, prop="box_gamma", value=3, unit=4, dev=6)

            if "Number of molecules:" in line:
                break

        # 4th parsing part: average molecule properties
        # --------------------------------------------------
        # from: "Number of molecules"
        # to: end of file

        icomponent = 0
        for line in fobj:
            # Consider to change it with parse_line?
            if "Average loading absolute [molecules/unit cell]" in line:
                res_per_component[icomponent]["loading_absolute_average"] = float(line.split()[5])
                res_per_component[icomponent]["loading_absolute_dev"] = float(line.split()[7])
                res_per_component[icomponent]["loading_absolute_unit"] = "molecules/unit cell"
            elif "Average loading excess [molecules/unit cell]" in line:
                res_per_component[icomponent]["loading_excess_average"] = float(line.split()[5])
                res_per_component[icomponent]["loading_excess_dev"] = float(line.split()[7])
                res_per_component[icomponent]["loading_excess_unit"] = "molecules/unit cell"
                icomponent += 1
            if icomponent >= ncomponents:
                break

        for line in fobj:
            for to_parse in LINES_WITH_COMPONENT_LIST:
                if to_parse[0] in line:
                    parse_lines_with_component(
                        res_per_component, component_names, line, to_parse[1]
                    )

        # Assigning to None all the quantities that are meaningless if not running a Widom insertion calculation
        for res_comp in res_per_component:
            for prop in ["henry_coefficient", "widom_rosenbluth_factor", "chemical_potential"]:
                if res_comp["{}_dev".format(prop)] == 0.0:
                    res_comp["{}_average".format(prop)] = None
                    res_comp["{}_dev".format(prop)] = None

            # The section "Adsorption energy from Widom-insertion" is not showing in the output if no widom is performed
            if "adsorption_energy_widom_average" not in res_comp:
                res_comp["adsorption_energy_widom_unit"] = "kJ/mol"
                res_comp["adsorption_energy_widom_dev"] = None
                res_comp["adsorption_energy_widom_average"] = None

    return_dictionary = {"general": result_dict, "components": {}}

    for name, value in zip(component_names, res_per_component):
        return_dictionary["components"][name] = value

    # Parsing all the warning that are printed in the output file, avoiding redoundancy
    with open(output_abs_path, "r") as fobj:
        for line in fobj:
            if "WARNING" in line:
                warning_touple = (system_name, line)
                if warning_touple not in warnings:
                    warnings.append(warning_touple)
    return return_dictionary, warnings
