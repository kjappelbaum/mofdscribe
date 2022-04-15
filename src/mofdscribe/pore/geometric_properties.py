from matminer.featurizers.base import BaseFeaturizer
from loguru import logger
import subprocess
import os
import re
from tempfile import TemporaryDirectory
from pymatgen.core import Structure
from ..utils import is_tool
from typing import Union, List, Tuple

ZEOPP_BASE_COMMAND = ["network", "-ha"]
NO_ZEOPP_WARNING = "Did not find the zeo++ network binary in the path. \
            Can not run pore analysis."


PROBE_RADII = {
    "CO2": 1.525,
    "N2": 1.655,
    "H2": 1.48,
    "CH4": 1.865,
    "O2": 1.51,
    "Xe": 1.985,
    "Kr": 1.83,
    "H2O": 1.58,
    "H2S": 1.74,
}


def run_zeopp(structure: Structure, command, parser) -> dict:
    """Run zeopp with network -ha  (http://www.zeoplusplus.org/examples.html)
    to find the pore diameters
    Args:
        structure (Structure): pymatgen Structure object
    Returns:
        dict: pore analysis results
    """
    if not is_tool("network"):
        logger.error(NO_ZEOPP_WARNING)
    with TemporaryDirectory() as tempdir:
        structure_path = os.path.join(tempdir, "structure.cif")
        result_path = os.path.join(tempdir, "result.res")
        structure.to("cif", structure_path)
        cmd = ZEOPP_BASE_COMMAND + command + [str(result_path), str(structure_path)]
        _ = subprocess.run(
            cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        with open(result_path, "r") as handle:
            results = handle.read()

        zeopp_results = parser(results)

        return zeopp_results


def _parse_res_zeopp(filecontent: str) -> Tuple[List[float], List[str]]:
    """Parse the results line of a network -res call to zeopp
    Args:
        filecontent (str): results file
    Returns:
        dict: largest included sphere, largest free sphere,
            largest included sphera along free sphere path
    """
    first_line = filecontent.split("\n")[0]
    parts = first_line.split()

    results = {
        "lis": float(parts[1]),  # largest included sphere
        "lifs": float(parts[2]),  # largest free sphere
        "lifsp": float(parts[3]),  # largest included sphere along free sphere path
    }

    return list(results.values()), list(results.keys())


def _parse_sa_zeopp(filecontent):
    regex_unitcell = re.compile("Unitcell_volume: ((\d+\.\d+)|\d+)")
    regex_density = re.compile("Density: ((\d+\.\d+)|\d+)")
    asa_a2 = re.compile("ASA_A\^2: ((\d+\.\d+)|\d+)")
    asa_m2cm3 = re.compile("ASA_m\^2/cm\^3: ((\d+\.\d+)|\d+)")
    asa_m2g = re.compile("ASA_m\^2/g: ((\d+\.\d+)|\d+)")
    nasa_a2 = re.compile("NASA_A\^2: ((\d+\.\d+)|\d+)")
    nasa_m2cm3 = re.compile("NASA_m\^2/cm\^3: ((\d+\.\d+)|\d+)")
    nasa_m2g = re.compile("NASA_m\^2/g: ((\d+\.\d+)|\d+)")

    d = {
        "unitcell_volume": float(re.findall(regex_unitcell, filecontent)[0][0]),
        "density": float(re.findall(regex_density, filecontent)[0][0]),
        "asa_a2": float(re.findall(asa_a2, filecontent)[0][0]),
        "asa_m2cm3": float(re.findall(asa_m2cm3, filecontent)[0][0]),
        "asa_m2g": float(re.findall(asa_m2g, filecontent)[0][0]),
        "nasa_a2": float(re.findall(nasa_a2, filecontent)[0][0]),
        "nasa_m2cm3": float(re.findall(nasa_m2cm3, filecontent)[0][0]),
        "nasa_m2g": float(re.findall(nasa_m2g, filecontent)[0][0]),
    }

    return d


class PoreDiameters(BaseFeaturizer):
    def ___init___(self):
        self.labels = ["lis", "lifs", "lifsp"]

    def featurize(self, s):
        values, _ = run_zeopp(s, "-res", _parse_res_zeopp)
        return values

    def feature_labels(self):
        return self.labels

    def citations(self):
        return [
            "@article{Willems2012,"
            "doi = {10.1016/j.micromeso.2011.08.020},"
            "url = {https://doi.org/10.1016/j.micromeso.2011.08.020},"
            "year = {2012},"
            "month = feb,"
            "publisher = {Elsevier {BV}},"
            "volume = {149},"
            "number = {1},"
            "pages = {134--141},"
            "author = {Thomas F. Willems and Chris H. Rycroft and Michaeel Kazi and Juan C. Meza and Maciej Haranczyk},"
            "title = {Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials},"
            "journal = {Microporous and Mesoporous Materials}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]


class SurfaceArea(BaseFeaturizer):
    def ___init___(
        self,
        probe_radius: Union[str, float] = 0.1,
        num_samples: int = 100,
        channel_radius: Union[str, float, None] = None,
    ):
        if probe_radius != channel_radius:
            logger.warning(
                "Probe radius and channel radius are different. This is a highly unusual setting."
            )
        if isinstance(probe_radius, str):
            try:
                probe_radius = PROBE_RADII[probe_radius]
            except KeyError:
                logger.error(f"Probe radius {probe_radius} not found in PROBE_RADII")

        if channel_radius is None:
            channel_radius = probe_radius

        self.probe_radius = probe_radius
        self.num_samples = num_samples
        self.channel_radius = channel_radius

        self.labels = [
            "uc_volume",
            "density",
            "asa_a^2",
            "asa_m^2/cm^3",
            "asa_m^2/g",
            "nasa_a^2",
            "nasa_m^2/cm^3",
            "nasa_m^2/g",
        ]

    def featurize(self, s):
        command = ["-sa", f"{self.channel_radius}", f"{self.probe_radius}", f"{self.num_samples}"]
        results, _ = run_zeopp(s, command, _parse_sa_zeopp)
        return results

    def feature_labels(self):
        return self.labels

    def citations(self):
        return [
            "@article{Willems2012,"
            "doi = {10.1016/j.micromeso.2011.08.020},"
            "url = {https://doi.org/10.1016/j.micromeso.2011.08.020},"
            "year = {2012},"
            "month = feb,"
            "publisher = {Elsevier {BV}},"
            "volume = {149},"
            "number = {1},"
            "pages = {134--141},"
            "author = {Thomas F. Willems and Chris H. Rycroft and Michaeel Kazi and Juan C. Meza and Maciej Haranczyk},"
            "title = {Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials},"
            "journal = {Microporous and Mesoporous Materials}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]


class AccessibleVolume(BaseFeaturizer):
    def ___init___(
        self,
        probe_radius: Union[str, float] = 0.1,
        num_samples: int = 100,
        channel_radius: Union[str, float, None] = None,
    ):
        if probe_radius != channel_radius:
            logger.warning(
                "Probe radius and channel radius are different. This is a highly unusual setting."
            )
        if isinstance(probe_radius, str):
            try:
                probe_radius = PROBE_RADII[probe_radius]
            except KeyError:
                logger.error(f"Probe radius {probe_radius} not found in PROBE_RADII")

        if channel_radius is None:
            channel_radius = probe_radius

        self.probe_radius = probe_radius
        self.num_samples = num_samples
        self.channel_radius = channel_radius
        self.labels = [
            "uc_volume",
            "density",
            "av_a^2",
            "av_volume_fraction",
            "av_cm^3_g",
            "nav_a^3",
            "nav_volume_fraction",
            "nav_cm^3_g",
        ]

    def featurize(self, s):
        ...

    def feature_labels(self):
        ...

    def citations(self):
        return [
            "@article{Willems2012,"
            "doi = {10.1016/j.micromeso.2011.08.020},"
            "url = {https://doi.org/10.1016/j.micromeso.2011.08.020},"
            "year = {2012},"
            "month = feb,"
            "publisher = {Elsevier {BV}},"
            "volume = {149},"
            "number = {1},"
            "pages = {134--141},"
            "author = {Thomas F. Willems and Chris H. Rycroft and Michaeel Kazi and Juan C. Meza and Maciej Haranczyk},"
            "title = {Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials},"
            "journal = {Microporous and Mesoporous Materials}"
            "}"
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
