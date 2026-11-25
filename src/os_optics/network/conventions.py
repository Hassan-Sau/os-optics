"""The conventions an ``opensci-linear-optical-network/v1`` document states, and its quantization."""

from __future__ import annotations

from ..contracts.network import NetworkConventionsV1, QuantizationV1, ReflectionModelV1

#: Decimals kept (absolute rounding) per quantity; see ``conventions().quantization``.
OPL_DIGITS = 9  # optical / geometric path length and each output beam's reference OPL, mm
OPL_OFFSET_DIGITS = 12  # a path's OPL minus its output beam's reference OPL, mm (the relative phases of T)
AMPLITUDE_DIGITS = 12  # path amplitudes and the source polarization, before phases are formed
ENTRY_DIGITS = 9  # transfer-matrix entries
FRACTION_DIGITS = 9  # power fractions, singular values, column transmission
LENGTH_DIGITS = 9  # positions and beam sizes, mm
DIRECTION_DIGITS = 12  # unit vectors
RATIO_DIGITS = 9  # refractive indices, M^2, OPD / coherence-length ratios
#: The same as resolutions.
OPL_QUANTUM = 10.0 ** -OPL_DIGITS
OPL_OFFSET_QUANTUM = 10.0 ** -OPL_OFFSET_DIGITS
AMPLITUDE_QUANTUM = 10.0 ** -AMPLITUDE_DIGITS
ENTRY_QUANTUM = 10.0 ** -ENTRY_DIGITS
ELEMENT_DIGITS = AMPLITUDE_DIGITS - 3  # element Jones matrices (a ratio of two probe states)
ELEMENT_QUANTUM = 10.0 ** -ELEMENT_DIGITS
FRACTION_QUANTUM = 10.0 ** -FRACTION_DIGITS
LENGTH_QUANTUM = 10.0 ** -LENGTH_DIGITS
DIRECTION_QUANTUM = 10.0 ** -DIRECTION_DIGITS
RATIO_QUANTUM = 10.0 ** -RATIO_DIGITS
#: Angles of incidence (deg) and Gaussian waist distances (mm) keep 6 decimals.
COARSE_DIGITS = 6
COARSE_QUANTUM = 10.0 ** -COARSE_DIGITS
#: Wavelengths, line powers and coherence lengths keep 12 significant digits.
SIGNIFICANT_DIGITS = 12
#: Relative tolerance of the classical-limit check (observed agreement ~1e-9) and its power floor (fraction of the probe).
CHECK_TOLERANCE = 1e-6
CHECK_FLOOR = 1e-9
#: T^H T = I within this, or within the quantization bound when that is larger (``transfer_matrix.lossless_tolerance``); a
#: singular value above 1 + that tolerance is gain (refused).
LOSSLESS_TOLERANCE = 1e-6
#: Coherence length of the classical-check trace: the network is the coherent, single-frequency description.
PROBE_COHERENCE_LENGTH_MM = 1e12


def conventions() -> NetworkConventionsV1:
    return NetworkConventionsV1(
        time_dependence="E(r, t) = Re[E(r) exp(-i omega t)]; a plane wave is exp[i(k.r - omega t)]",
        propagation_phase="propagation over an optical path length L multiplies the amplitude by exp(+i k0 L), k0 = 2 pi / wavelength_nm "
                          "(vacuum); L is the chief ray's optical path length (sum of n x geometric length), formed as the output "
                          "beam's reference_opl_mm (rounded to 1e-9 mm: a phase common to the beam's rows, a choice of reference "
                          "plane) plus the path's opl_offset_mm (rounded to 1e-12 mm). Relative phases of paths reaching one output "
                          "beam are exact to k0 x 5e-13 mm (~5e-9 rad at 633 nm) on top of the Engine's optical path lengths "
                          "(float64); a row's absolute phase to k0 x 5e-10 mm",
        amplitude_normalization="flux-normalized amplitudes: |T_ij|^2 is the fraction of input mode j's power (photon flux) that "
                                "leaves in output mode i",
        transfer_definition="b_out = T a_in for classical field amplitudes; for photons a_in_j^dagger -> sum_i T_ij b_out_i^dagger. Rows are "
                            "output modes, columns input modes; T is not unitary when light is lost (column_transmission < 1)",
        polarization_basis="each mode's (H, V): V = design +Z projected perpendicular to the beam direction d and normalized, H = V x d, so "
                           "H x V = d. A beam travelling horizontally has V vertical (up) and H horizontal; this holds at every port, "
                           "independent of the Engine's internal per-ray bases",
        vertical_beam_fallback="for a beam within 45 degrees of vertical ((d . Z)^2 > 1/2), V = design +Y projected perpendicular to "
                               "d instead (both references stay well conditioned); at the 45 degree switch the basis turns about the "
                               "beam, and a mode within 1e-9 of it is reported (validity warning POLARIZATION_BASIS_SWITCH)",
        input_reference_plane="the source's emission point (its position_mm): phase 0 of the source's line",
        output_reference_plane="the beam's chief-ray crossing of the output observation plane (output_modes[].position_mm)",
        gauge="a phase factor on a whole row (output reference point) or a whole column (input reference point) is a choice of reference "
              "plane: photon statistics, interference visibilities, |T_ij| and phases of closed loops arg(T_ij T_kl T_il* T_kj*) do not "
              "depend on it",
        group_delay="group_opl_mm = sum over the chief ray's segments from the emission point to the output plane of n_g x length, "
                    "n_g = n - lambda dn/dlambda of the segment's medium at wavelength_nm from the same material model the trace uses "
                    "(formula laws, the environment's air included: in closed form; tabulated: difference over +-0.01 nm; constant indices: n_g = n); group_delay_ps = "
                    "group_opl_mm / c. It is the first-order (group-velocity) delay of a wavepacket centred at wavelength_nm; group-delay "
                    "dispersion (pulse broadening) is not included. Phase uses opl_mm, envelope timing group_opl_mm",
        output_modes="every distinct beam on an output plane is one spatial mode with two polarizations; beams whose Gaussian modes "
                     "coincide (overlap >= 1 - mode_overlap_tolerance) are the same mode, beams crossing the plane in opposite directions "
                     "are different modes",
        input_modes="one spatial mode per exported source (its spectral line at wavelength_nm), with two polarizations",
        reflection_models=(
            ReflectionModelV1(interaction="mirror", model="metal reflection: (r_s, r_p) = (sqrt(R_s) e^{i pi}, sqrt(R_p)) in the (s, d_out x s) "
                              "triad, i.e. E_out = -(I - 2 n n^T) E_in for R = 1: diag(+1, -1) on (H, V) for a fold in the horizontal plane (H is p, V is s there); "
                              "a coating that states its phases gives r_s = sqrt(R_s) e^{i phi_rs}, r_p = sqrt(R_p) e^{i phi_rp} instead"),
            ReflectionModelV1(interaction="beam_splitter", model="thin symmetric splitter (non-polarizing or polarizing by its coating): "
                              "t_s = sqrt(T_s), t_p = sqrt(T_p); r_s = i sqrt(R_s), r_p = -i sqrt(R_p) in the (s, d_out x s) triad on "
                              "both sides (a physical reflection: r_p = -r_s at normal incidence, like a mirror and Fresnel); unitary "
                              "for R + T = 1 per polarization; a coating that states its phases gives r = sqrt(R) e^{i phi_r}, "
                              "t = sqrt(T) e^{i phi_t} per polarization instead (refused as NETWORK_NOT_PASSIVE when they break unitarity)"),
            ReflectionModelV1(interaction="refract", model="dielectric interface: Fresnel amplitudes (Born & Wolf signs; energy-normalized "
                              "transmission), complex under total internal reflection; a coating replaces the magnitudes sqrt(R), sqrt(T) "
                              "and keeps the Fresnel phases, or uses its stated transmission / reflection phases (not under total "
                              "internal reflection, where the bare Fresnel amplitudes apply)"),
            ReflectionModelV1(interaction="retarder", model="J = e^{-i delta/2} a a^T + e^{+i delta/2} b b^T (fast axis a), times "
                              "sqrt(transmittance)"),
            ReflectionModelV1(interaction="polarizer", model="J = sqrt(t_par) a a^T + sqrt(t_perp) b b^T (transmission axis a)"),
        ),
        not_modelled=(
            "Gouy phase", "group-delay dispersion (the delay is first order: group index at wavelength_nm)",
            "coating phases a coating does not state (catalog coatings publish powers only: the reflection models above apply)",
            "the phase of an ideal thin lens (refused)",
            "the phase of a neutral-density attenuator, transmitted or reflected (refused: its model states powers only)",
            "the optical path through a Faraday rotator (refused: its model states a rotation and a transmittance)",
            "the phase of a diffraction grating's order (refused: its model states directions and at most powers)", "scattering, diffraction, depolarization, nonlinear and active elements",
            "vacuum input ports without a source",
        ),
        quantization=(
            QuantizationV1(quantity="optical and geometric path lengths, output beams' reference optical path lengths (before "
                           "phases are formed)", resolution=OPL_QUANTUM, unit="mm"),
            QuantizationV1(quantity="path optical path length relative to its output beam's reference (before phases are formed)",
                           resolution=OPL_OFFSET_QUANTUM, unit="mm"),
            QuantizationV1(quantity="path amplitudes and source polarization (before phases are formed)", resolution=AMPLITUDE_QUANTUM,
                           unit="1"),
            QuantizationV1(quantity="transfer-matrix entries (real and imaginary parts)", resolution=ENTRY_QUANTUM, unit="1"),
            QuantizationV1(quantity="element matrices", resolution=ELEMENT_QUANTUM, unit="1"),
            QuantizationV1(quantity="power fractions, singular values, lossless tolerance, column transmission, check values",
                           resolution=FRACTION_QUANTUM, unit="1"),
            QuantizationV1(quantity="positions, beam and waist radii, path-pair optical path differences", resolution=LENGTH_QUANTUM,
                           unit="mm"),
            QuantizationV1(quantity="path-pair delays", resolution=LENGTH_QUANTUM, unit="ps"),
            QuantizationV1(quantity="path group optical path lengths and path-pair group optical path differences", resolution=OPL_QUANTUM,
                           unit="mm"),
            QuantizationV1(quantity="path and path-pair group delays", resolution=OPL_QUANTUM, unit="ps"),
            QuantizationV1(quantity="refractive indices, M^2, OPD / coherence-length ratios", resolution=RATIO_QUANTUM, unit="1"),
            QuantizationV1(quantity="unit vectors", resolution=DIRECTION_QUANTUM, unit="1"),
            QuantizationV1(quantity="element angles of incidence", resolution=COARSE_QUANTUM, unit="deg"),
            QuantizationV1(quantity="Gaussian waist distances", resolution=COARSE_QUANTUM, unit="mm"),
            QuantizationV1(quantity="wavelength_nm, line_power_W, coherence_length_mm", resolution=0.0,
                           unit=f"{SIGNIFICANT_DIGITS} significant digits"),
            QuantizationV1(quantity="phase reduction", resolution=0.0, unit="exp(i k0 L) of the quantized L, reduced modulo one wave "
                           "in exact decimal arithmetic"),
        ),
    )
