import os
import re
import subprocess
from io import StringIO
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from matminer.featurizers.base import BaseFeaturizer
from pymatgen.core import Structure

from ..utils import is_tool

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
            cwd=tempdir,
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

    return results


def _parse_sa_zeopp(filecontent):
    regex_unitcell = re.compile("Unitcell_volume: ((\d+\.\d+)|\d+)|$")
    regex_density = re.compile("Density: ((\d+\.\d+)|\d+)|$")
    asa_a2 = re.compile("ASA_A\^2: ((\d+\.\d+)|\d+)|$")
    asa_m2cm3 = re.compile("ASA_m\^2/cm\^3: ((\d+\.\d+)|\d+)|$")
    asa_m2g = re.compile("ASA_m\^2/g: ((\d+\.\d+)|\d+)|$")
    nasa_a2 = re.compile("NASA_A\^2: ((\d+\.\d+)|\d+)|$")
    nasa_m2cm3 = re.compile("NASA_m\^2/cm\^3: ((\d+\.\d+)|\d+)|$")
    nasa_m2g = re.compile("NASA_m\^2/g: ((\d+\.\d+)|\d+)|$")

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


def _parse_volpo_zeopp(filecontent):
    regex_unitcell = re.compile("Unitcell_volume: ((\d+\.\d+)|\d+)|$")
    regex_density = re.compile("Density: ((\d+\.\d+)|\d+)|$")
    av_a3 = re.compile("AV_A\^3: ((\d+\.\d+)|\d+)|$")
    av_volume_fraction = re.compile("AV_Volume_fraction: ((\d+\.\d+)|\d+)|$")
    av_cm3g = re.compile("AV_cm\^3/g: ((\d+\.\d+)|\d+)|$")
    nav_a3 = re.compile("NAV_A\^3: ((\d+\.\d+)|\d+)|$")
    nav_volume_fraction = re.compile("NAV_Volume_fraction: ((\d+\.\d+)|\d+)|$")
    nav_cm3g = re.compile("NAV_cm\^3/g: ((\d+\.\d+)|\d+)|$")

    d = {
        "unitcell_volume": float(re.findall(regex_unitcell, filecontent)[0][0]),
        "density": float(re.findall(regex_density, filecontent)[0][0]),
        "av_a3": float(re.findall(av_a3, filecontent)[0][0]),
        "av_volume_fraction": float(re.findall(av_volume_fraction, filecontent)[0][0]),
        "av_cm3g": float(re.findall(av_cm3g, filecontent)[0][0]),
        "nav_a3": float(re.findall(nav_a3, filecontent)[0][0]),
        "nav_volume_fraction": float(re.findall(nav_volume_fraction, filecontent)[0][0]),
        "nav_cm3g": float(re.findall(nav_cm3g, filecontent)[0][0]),
    }

    return d


def _parse_ray_hist_zeopp(filecontent):
    return [float(i) for i in filecontent.split("\n")[1:-1]]


def _parse_psd_zeopp(filecontent):
    return pd.read_csv(
        StringIO(filecontent),
        skiprows=11,
        names=["bin", "count", "cumulative", "derivative"],
        sep="\s+",
    )


class PoreDiameters(BaseFeaturizer):
    def __init__(self):
        self.labels = ["lis", "lifs", "lifsp"]

    def featurize(self, s):
        result = run_zeopp(s, ["-res"], _parse_res_zeopp)
        return np.array(list(result.values()))

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
    def __init__(
        self,
        probe_radius: Union[str, float] = 0.1,
        num_samples: int = 100,
        channel_radius: Union[str, float, None] = None,
    ):
        if channel_radius is not None:
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
            "asa_a2",
            "asa_m2cm3",
            "asa_m2g",
            "nasa_a2",
            "nasa_m2cm3",
            "nasa_m2g",
        ]

    def featurize(self, s: Structure) -> np.ndarray:
        command = [
            "-sa",
            f"{self.channel_radius}",
            f"{self.probe_radius}",
            f"{self.num_samples}",
        ]
        results = run_zeopp(s, command, _parse_sa_zeopp)
        return np.array(list(results.values()))

    def feature_labels(self) -> List[str]:
        return self.labels

    def citations(self) -> List[str]:
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

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]


class AccessibleVolume(BaseFeaturizer):
    def __init__(
        self,
        probe_radius: Union[str, float] = 0.1,
        num_samples: int = 100,
        channel_radius: Union[str, float, None] = None,
    ):
        if channel_radius is not None:
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
            "av_a2",
            "av_volume_fraction",
            "av_cm3g",
            "nav_a3",
            "nav_volume_fraction",
            "nav_cm3g",
        ]

    def featurize(self, s: Structure) -> np.ndarray:
        command = ["-vol", f"{self.channel_radius}", f"{self.probe_radius}", f"{self.num_samples}"]
        results = run_zeopp(s, command, _parse_volpo_zeopp)
        return np.array(list(results.values()))

    def feature_labels(self) -> List[str]:
        return self.labels

    def citations(self) -> List[str]:
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

    def implementors(self) -> List[str]:
        return ["Kevin Maik Jablonka"]


class RayTracingHistogram(BaseFeaturizer):
    def __init__(
        self,
        probe_radius: Union[str, float] = 0.0,
        num_samples: int = 50000,
        channel_radius: Union[str, float, None] = None,
    ) -> None:
        """The algorithm (implemented in zeo++) shoots random rays through the accesible volume of the cell until the ray hits atoms, and it records their lenghts to provide the corresponding histogram.
        Such ray histograms are supposed to encode the shape, topology, distribution and size of voids.

        Currently, the histogram is hard-coded to be of length 1000 (in zeo++ itself).

        Args:
            probe_radius (Union[str, float], optional): Used to estimate the accessible volume.
            Only the accessible volume is then considered for the histogram.
            Defaults to 0.0.
            num_samples (int, optional): Number of rays that are placed through sample.
                Original publication used  1,000,000 sample points for IZA zeolites and 100,000 sample points for hypothetical zeolites. Larger numbers increase the runtime Defaults to 50000.
            channel_radius (Union[str, float, None], optional):  Radius of a probe used to determine accessibility of the void space. Should typically equal the radius of the `probe_radius`. If set to `None`, we will use the `probe_radius`. Defaults to None.
        """
        if channel_radius is not None:
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

    def feature_labels(self):
        return [f"ray_hist_{i}" for i in range(1000)]

    def featurize(self, s):
        command = [
            "-ray_atom",
            f"{self.channel_radius}",
            f"{self.probe_radius}",
            f"{self.num_samples}",
        ]
        results = run_zeopp(s, command, _parse_ray_hist_zeopp)
        return np.array(results)

    def citations(self):
        return [
            "@article{Jones2013,"
            "doi = {10.1016/j.micromeso.2013.07.033},"
            "url = {https://doi.org/10.1016/j.micromeso.2013.07.033},"
            "year = {2013},"
            "month = nov,"
            "publisher = {Elsevier {BV}},"
            "volume = {181},"
            "pages = {208--216},"
            "author = {Andrew J. Jones and Christopher Ostrouchov and Maciej Haranczyk and Enrique Iglesia},"
            "title = {From rays to structures: Representation and selection of void structures in zeolites using stochastic methods},"
            "journal = {Microporous and Mesoporous Materials}"
            "}",
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
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]


class PoreSizeDistribution(BaseFeaturizer):
    def __init__(
        self,
        probe_radius: Union[str, float] = 0.0,
        num_samples: int = 5000,
        channel_radius: Union[str, float, None] = None,
        hist_type: str = "derivative",
    ) -> None:
        """
        The pore size distribution describes how much of the void space corresponds to certain pore sizes.
        We use the implementation in zeo++ to calculate the pore size distribution.

        The pore size distribution has been used by the group of Gómez-Gualdrón  as pore size standard deviation (PSSD) in, for example, 10.1021/acs.jctc.9b00940 and 10.1063/5.0048736.

        Currently, the histogram is hard-coded to be of length 1000 between 0 and 100 Angstrom (in zeo++ itself).

        Args:
            probe_radius (Union[str, float], optional): Used to estimate the accessible volume.
            Only the accessible volume is then considered for the histogram.
            Defaults to 0.0.
            num_samples (int, optional): Number of rays that are placed through sample.
                Original publication used  1,000,000 sample points for IZA zeolites and 100,000 sample points for hypothetical zeolites. Larger numbers increase the runtime Defaults to 50000.
            channel_radius (Union[str, float, None], optional):  Radius of a probe used to determine accessibility of the void space. Should typically equal the radius of the `probe_radius`. If set to `None`, we will use the `probe_radius`. Defaults to None.
            hist_type (str, optional): Type of the histogram. Available options `count`, `cumulative`, `derivative` (The derivative distribution describes the change in the cumulative distribution with respect to pore size). Defaults to "derivative".
        """
        if channel_radius is not None:
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

        self.type = hist_type.lower()
        assert self.type in [
            "count",
            "cumulative",
            "derivative",
        ], "Invalid histogram type, must be one of `count`, `cumulative`, `derivative`"

        self.probe_radius = probe_radius
        self.num_samples = num_samples
        self.channel_radius = channel_radius

    def feature_labels(self):
        return [f"psd_hist_{i}" for i in range(1000)]

    def featurize(self, s):
        command = [
            "-psd",
            f"{self.channel_radius}",
            f"{self.probe_radius}",
            f"{self.num_samples}",
        ]
        results = run_zeopp(s, command, _parse_psd_zeopp)
        return results[self.type].values

    def citations(self):
        return [
            "@article{Pinheiro2013,"
            "doi = {10.1016/j.jmgm.2013.05.007},"
            "url = {https://doi.org/10.1016/j.jmgm.2013.05.007},"
            "year = {2013},"
            "month = jul,"
            "publisher = {Elsevier {BV}},"
            "volume = {44},"
            "pages = {208--219},"
            "author = {Marielle Pinheiro and Richard L. Martin and Chris H. Rycroft and Andrew Jones and Enrique Iglesia and Maciej Haranczyk},"
            "title = {Characterization and comparison of pore landscapes in crystalline porous materials},"
            "journal = {Journal of Molecular Graphics and Modelling}"
            "}",
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
            "}",
        ]

    def implementors(self):
        return ["Kevin Maik Jablonka"]
