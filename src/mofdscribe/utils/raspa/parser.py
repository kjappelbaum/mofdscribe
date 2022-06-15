# -*- coding: utf-8 -*-
# noqa: W605
"""RASPA output parser taken from the official repo [1]

[1] https://github.com/iRASPA/RASPA2/blob/f1eb37f3ffd610851220199307d91872789e1254/python/output_parser.py


The MIT License

RASPA: a molecular-dynamics, monte-carlo and optimization code for nanoporous materials
Copyright (C) 2006-2019 David Dubbeldam, Sofia Calero, Thijs Vlugt, Donald E. Ellis, and Randall Q. Snurr.

    D.Dubbeldam@uva.nl            http://www.uva.nl/profiel/d/u/d.dubbeldam/d.dubbeldam.html
    scaldia@upo.es                http://www.upo.es/raspa/
    t.j.h.vlugt@tudelft.nl        http://homepage.tudelft.nl/v9k6y
    don-ellis@northwestern.edu    http://dvworld.northwestern.edu/
    snurr@northwestern.edu        http://zeolites.cqe.northwestern.edu/

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""
import re


def parse(raspa_output):
    """Specific parsing of the output file.
    Args:
        raspa_output: A string representing unparsed RASPA output
    Returns:
        A data structure generated from the RASPA file
    """
    # Reads the string into a newline-separated list, skipping useless lines
    data = [
        row.strip()
        for row in raspa_output.splitlines()
        if row and all(d not in row for d in ["-----", "+++++"])
    ]

    # Generally, categories in the output are delimited by equal signs
    delimiters = [
        i
        for i, row in enumerate(data)
        if "=====" in row and "Exclusion constraints energy" not in data[i - 1]
    ]

    # Append a row for "absolute adsorption:" and "excess adsorption:"
    # These values are separated into two rows
    abs_adsorp_rows = [i for i, row in enumerate(data) if "absolute adsorption:" in row]
    for row in abs_adsorp_rows:
        data[row] += "  " + data[row + 1]
        data[row + 2] += data[row + 3]
        data[row + 1], data[row + 3] = " ", " "

    # Use the delimiters to make a high-level dict. Title is row before
    # delimiter, and content is every row after delimiter, up to the next title
    info = {
        data[n - 1].strip(":"): data[n + 1 : delimiters[i + 1] - 1]
        for i, n in enumerate(delimiters[:-1])
    }

    # Let's PARSE!
    for key, values in info.items():
        d, note_index = {}, 1
        for item in values:
            # Takes care of all "Blocks[ #]", skipping hard-to-parse parts
            if "Block" in item and "Box-lengths" not in key and "Van der Waals:" not in item:
                blocks = _clean(item.split())
                d["".join(blocks[:2])] = blocks[2:]

            # Most of the average data values are parsed in this section
            elif (
                any(s in item for s in ["Average     ", "Surface area:"])
                and "desorption" not in key
            ):
                average_data = _clean(item.split())
                # Average values organized by its unit, many patterns here
                if len(average_data) == 8:
                    del average_data[2:4]
                    d[" ".join(average_data[4:6])] = average_data[1:4]
                elif len(average_data) == 5:
                    d[average_data[-1]] = average_data[1:4]
                elif "Surface" in average_data[0]:
                    d[average_data[-1]] = average_data[2:5]
                # This is the common case
                else:
                    del average_data[2]
                    d[average_data[-1]] = average_data[1:4]

            # Average box-lengths has its own pattern
            elif "Box-lengths" in key:
                box_lengths = _clean(item.split())
                i = 3 if "angle" in item else 2
                d[" ".join(box_lengths[:i])] = box_lengths[i:]

            # "Heat of Desorption" section
            elif "desorption" in key:
                if "Note" in item:
                    notes = re.split("[:\s]{2,}", item)
                    d["%s %d" % (notes[0], note_index)] = notes[1]
                    note_index += 1
                else:
                    heat_desorp = _clean(item.split())
                    # One line has "Average" in front, force it to be normal
                    if "Average" in item:
                        del heat_desorp[0]
                    d[heat_desorp[-1]] = heat_desorp[0:3]

            # Parts where Van der Waals are included
            elif (
                "Host-" in key or "-Cation" in key or "Adsorbate-Adsorbate" in key
            ) and "desorption" not in key:
                van_der = item.split()
                # First Column
                if "Block" in van_der[0]:
                    sub_data = [_clean(s.split(":")) for s in re.split("\s{2,}", item)[1:]]
                    sub_dict = {s[0]: s[1] for s in sub_data[:2]}
                    d["".join(van_der[:2])] = [float(van_der[2]), sub_dict]
                # Average for each columns
                elif "Average" in item:
                    avg = _clean(re.split("\s{2,}", item))
                    vdw, coulomb = [_clean(s.split(": ")) for s in avg[2:4]]
                    d[avg[0]] = avg[1]
                    d["Average %s" % vdw[0]] = vdw[1]
                    d["Average %s" % coulomb[0]] = coulomb[1]
                else:
                    d["standard deviation"] = _clean(van_der)

            # IMPORTANT STUFF
            elif "Number of molecules" in key:
                adsorb_data = _clean(item.rsplit(" ", 12))
                if "Component" in item:
                    gas_name = adsorb_data[2].strip("[]")
                    d[gas_name] = {}
                else:
                    d[gas_name][adsorb_data[0]] = adsorb_data[1:]

            # Henry and Widom
            elif "Average Widom" in item:
                d["Widom"] = _clean(item.rsplit(" ", 5))[1:]

            elif "Average Henry" in item:
                d["Henry"] = _clean(item.rsplit(" ", 5))[1:]

            # Ignore these
            elif any(s in item for s in ["=====", "Starting simulation", "Finishing simulation"]):
                continue

            # Other strings
            else:
                parsed_data = _clean(re.split("[()[\]:,\t]", item))
                d[parsed_data[0]] = parsed_data[1:]
        # Putting subdictionary back into main object
        info[key] = d
    info["HoA_K"] = float(re.findall("\<U_gh\>_1-\<U_h\>_0:\s+([-\d]+.\d+)", raspa_output)[0])
    return info


def _clean(split_list):
    """Strips and attempts to convert a list of strings to floats."""

    def try_float(s):
        try:
            return float(s)
        except ValueError:
            return s

    return [try_float(s.strip()) for s in split_list if s]
