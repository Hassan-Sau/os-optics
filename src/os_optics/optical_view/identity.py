"""Builder identity recorded in every scene (``ViewBuilderRefV1``): the view is part of the engine distribution.

``builder_package`` names the distribution that built the scene (``os-optics``), as every identity the engine writes does."""

from .._version import DISTRIBUTION_NAME, ENGINE_VERSION

BUILDER_PACKAGE = DISTRIBUTION_NAME
BUILDER_VERSION = ENGINE_VERSION
