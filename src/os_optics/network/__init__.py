"""Linear optical network export (``opensci-linear-optical-network/v1``): a Project's optics as modes, a complex transfer
matrix, beam paths and element Jones matrices, for quantum-optics or classical simulators the customer runs.

Pure Python + numpy on top of the Engine's own tracer; the Python Engine and the browser runtime run the same code and give
the same ``contract_hash``. ``os_optics.local.export_linear_optical_network`` exports a Project,
``os_optics.local.json_api.linear_optical_network`` is its JSON entry point and ``os-optics project network`` its
command. Each document states its conventions (``conventions``); ``os-optics schema linear-optical-network`` prints
the format.
"""

from ..contracts.network import NETWORK_SCHEMA, LinearOpticalNetworkV1, NetworkExportOptionsV1
from .basis import lab_basis
from .errors import NETWORK_NOT_EXPORTABLE, NetworkErrorCode, NetworkExportError, NetworkProblem
from .export import build_linear_optical_network, contract_hash_of, mode_overlap

__all__ = [
    "NETWORK_NOT_EXPORTABLE",
    "NETWORK_SCHEMA",
    "LinearOpticalNetworkV1",
    "NetworkErrorCode",
    "NetworkExportError",
    "NetworkExportOptionsV1",
    "NetworkProblem",
    "build_linear_optical_network",
    "contract_hash_of",
    "lab_basis",
    "mode_overlap",
]
