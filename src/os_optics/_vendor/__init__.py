"""OpenSci contracts bundled with os-optics (generated; do not edit).

``shared_contracts/`` holds the contract models the local workflow reads, writes and verifies: the pure-optical Project
and its Local Validation record (``optical_design``), the public optical catalog and its release manifest
(``design_catalog``), the Optical Simulation View scene (``optical_view``), the Assembly Preview Package and its local
validation result (``assembly_preview``), the Assembly Detail Package and the local Detailed Assembly Drawing outputs
(``assembly_detail``), the protected delivery envelope and decryption grant a client opens in memory
(``protected_assembly_delivery``), the manifest of a client-generated artifact upload (``client_assembly_artifact``) and
the Optical Layout Package of an optics-only purchase with its local drawing manifest (``optical_layout``), together
with the common modules they use. ``MANIFEST.json`` lists the SHA-256 of every contract module except the two
generated package ``__init__`` files. The public optical catalog itself is versioned data kept in a local catalog store
(``os_optics.catalog``), not part of this package.

This package is an implementation detail: use ``os_optics.local``, ``os_optics.protected_delivery`` and
``os_optics.client_artifacts`` rather than importing from here.
"""
