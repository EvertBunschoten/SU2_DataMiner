"""
LUT Generator for Two-Phase Thermodynamic Tables
======================================================
Generates Look-Up Tables for SU2 in Dragon format (.drg)

"""

import numpy as np
import CoolProp.CoolProp as CP
import CoolProp as CoolP
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class TwoPhase_LUT_Generator:
    """
    Generate structured Look-Up Tables for two-phase thermodynamic properties.
    Uses density-energy as controlling variables.
    """
    # TODO: initiate from SU2 DataMiner configuation
    def __init__(self, fluid_name="MM", EoS="HEOS", include_diagnostics=False,
                 viscosity_mixing_model="mcadams", conductivity_mixing_model="volume"):
        """
        Initialize the LUT generator.

        Parameters:
        -----------
        fluid_name : str
            CoolProp fluid name (e.g., "MM", "MDM", "D6")
        EoS : str
            Equation of state ("HEOS" for Helmholtz)
        include_diagnostics : bool
            If True, include T, p, VaporQuality in output (non-standard).
            If False (default), only include SU2-compatible variables.
        viscosity_mixing_model : str
            Two-phase viscosity mixing model:
            - "mcadams" (default): Reciprocal/harmonic average
            - "cicchitti": Mass-weighted average
            - "dukler": Volume-weighted average
        conductivity_mixing_model : str
            Two-phase conductivity mixing model:
            - "volume" (default): Volume-weighted
            - "mass": Mass-weighted
        """

        # TODO: retrieve fluid name, equation of state, transport variables from configuration.
        self.fluid_name = fluid_name
        self.EoS = EoS
        self.fluid = CP.AbstractState(EoS, fluid_name)
        self.include_diagnostics = include_diagnostics

        # Controlling variables TODO: retrieve from config
        self.controlling_vars = ["Density", "Energy"]

        # Table variables plus transport properties for viscous simulations
        self.table_vars = ["s", "dsdrho_e", "dsde_rho",
                           "d2sdrho2", "d2sdedrho", "d2sde2",
                           "ViscosityDyn", "Conductivity"]
        
        # TODO: Retrieve from EntropicVars
        # Optional diagnostic variables (not read by SU2 solver)
        self.diagnostic_vars = ["T", "p", "VaporQuality"]
        
        # Two-phase mixing model options
        # TODO: include in data generator
        self.viscosity_mixing_model = viscosity_mixing_model
        self.conductivity_mixing_model = conductivity_mixing_model

        # Finite difference step sizes (relative)
        self.fd_step_rho = 1e-5  # Relative step for density
        self.fd_step_e = 1e-5    # Relative step for energy

        print("\n" + "=" * 80)
        print(f"TWO-PHASE LUT GENERATOR FOR {fluid_name}")
        print("=" * 80)
        print(f"Equation of State: {EoS}")
        print(f"Controlling Variables: {self.controlling_vars}")
        print(f"Table Variables (SU2 compatible): {self.table_vars}")
        print(f"Finite difference steps: drho/rho = {self.fd_step_rho:.0e}, de/|e| = {self.fd_step_e:.0e}")
        print(f"Transport properties included: ViscosityDyn, Conductivity")
        print(f"Two-phase viscosity mixing: {self.viscosity_mixing_model}")
        print(f"Two-phase conductivity mixing: {self.conductivity_mixing_model}")
        if include_diagnostics:
            print(f"Diagnostic Variables (non-standard): {self.diagnostic_vars}")
        print("=" * 80 + "\n")

    # TODO: Include in data generator
    def _get_entropy_safe(self, rho, e):
        """
        Safely get entropy at (rho, e), returning None if evaluation fails.
        """
        try:
            self.fluid.update(CP.DmassUmass_INPUTS, rho, e)
            phase = self.fluid.phase()
            if phase == CoolP.iphase_critical_point:
                return None
            return self.fluid.smass()
        except:
            return None

    # TODO: Include in data generator
    def _compute_derivatives_fd(self, rho, e, s_center):
        """
        Compute all entropy derivatives using central finite differences.
        Works for both single-phase and two-phase regions.
        
        Returns:
        --------
        dict with keys: 'dsdrho_e', 'dsde_rho', 'd2sdrho2', 'd2sde2', 'd2sdedrho'
        Returns None if derivatives cannot be computed.
        """
        # Compute step sizes
        drho = max(rho * self.fd_step_rho, 1e-6)  # Minimum absolute step
        de = max(abs(e) * self.fd_step_e, 1.0)     # Minimum 1 J/kg step
        
        # Get entropy at stencil points for first derivatives (central difference)
        s_rho_plus = self._get_entropy_safe(rho + drho, e)
        s_rho_minus = self._get_entropy_safe(rho - drho, e)
        s_e_plus = self._get_entropy_safe(rho, e + de)
        s_e_minus = self._get_entropy_safe(rho, e - de)
        
        # Check if first derivative stencil is valid
        if any(s is None for s in [s_rho_plus, s_rho_minus, s_e_plus, s_e_minus]):
            # Try one-sided differences as fallback
            return self._compute_derivatives_fd_onesided(rho, e, s_center, drho, de)
        
        # First derivatives (central difference: O(h^2) accuracy)
        dsdrho_e = (s_rho_plus - s_rho_minus) / (2 * drho)
        dsde_rho = (s_e_plus - s_e_minus) / (2 * de)
        
        # Second derivatives
        d2sdrho2 = (s_rho_plus - 2*s_center + s_rho_minus) / (drho**2)
        d2sde2 = (s_e_plus - 2*s_center + s_e_minus) / (de**2)
        
        # Mixed derivative d2s/drhode
        # Use central difference: [s(rho+,e+) - s(rho+,e-) - s(rho-,e+) + s(rho-,e-)] / (4 * drho * de)
        s_pp = self._get_entropy_safe(rho + drho, e + de)
        s_pm = self._get_entropy_safe(rho + drho, e - de)
        s_mp = self._get_entropy_safe(rho - drho, e + de)
        s_mm = self._get_entropy_safe(rho - drho, e - de)
        
        if any(s is None for s in [s_pp, s_pm, s_mp, s_mm]):
            # Mixed derivative failed - use simpler approximation
            d2sdedrho = 0.0  # This is less critical than first derivatives
        else:
            d2sdedrho = (s_pp - s_pm - s_mp + s_mm) / (4 * drho * de)
        
        return {
            'dsdrho_e': dsdrho_e,
            'dsde_rho': dsde_rho,
            'd2sdrho2': d2sdrho2,
            'd2sde2': d2sde2,
            'd2sdedrho': d2sdedrho
        }
    # TODO: Include in data generator
    def _compute_derivatives_fd_onesided(self, rho, e, s_center, drho, de):
        """
        Fallback: compute derivatives using one-sided finite differences.
        Less accurate but more robust near boundaries.
        """
        derivs = {}
        
        # Try forward difference for ds/drho
        s_rho_plus = self._get_entropy_safe(rho + drho, e)
        s_rho_minus = self._get_entropy_safe(rho - drho, e)
        
        if s_rho_plus is not None and s_rho_minus is not None:
            derivs['dsdrho_e'] = (s_rho_plus - s_rho_minus) / (2 * drho)
            derivs['d2sdrho2'] = (s_rho_plus - 2*s_center + s_rho_minus) / (drho**2)
        elif s_rho_plus is not None:
            derivs['dsdrho_e'] = (s_rho_plus - s_center) / drho
            derivs['d2sdrho2'] = 0.0
        elif s_rho_minus is not None:
            derivs['dsdrho_e'] = (s_center - s_rho_minus) / drho
            derivs['d2sdrho2'] = 0.0
        else:
            return None  # Cannot compute
        
        # Try for ds/de
        s_e_plus = self._get_entropy_safe(rho, e + de)
        s_e_minus = self._get_entropy_safe(rho, e - de)
        
        if s_e_plus is not None and s_e_minus is not None:
            derivs['dsde_rho'] = (s_e_plus - s_e_minus) / (2 * de)
            derivs['d2sde2'] = (s_e_plus - 2*s_center + s_e_minus) / (de**2)
        elif s_e_plus is not None:
            derivs['dsde_rho'] = (s_e_plus - s_center) / de
            derivs['d2sde2'] = 0.0
        elif s_e_minus is not None:
            derivs['dsde_rho'] = (s_center - s_e_minus) / de
            derivs['d2sde2'] = 0.0
        else:
            return None
        
        # Mixed derivative - just set to zero if we're already using fallback
        derivs['d2sdedrho'] = 0.0
        
        return derivs

    # TODO: Include in data generator
    def _get_saturated_transport_properties(self, p, T):
        """
        Get transport properties at saturation conditions for both liquid and vapor phases.
        
        Parameters:
        -----------
        p : float
            Pressure [Pa]
        T : float
            Temperature [K] (saturation temperature)
            
        Returns:
        --------
        dict with keys: 'mu_l', 'mu_g', 'k_l', 'k_g', 'rho_l', 'rho_g'
        Returns None if properties cannot be computed.
        """
        try:
            # Create temporary fluid states for saturated liquid and vapor
            # Use pressure-quality inputs to get saturation properties
            
            # Saturated liquid (Q=0)
            self.fluid.update(CP.PQ_INPUTS, p, 0.0)
            mu_l = self.fluid.viscosity()
            k_l = self.fluid.conductivity()
            rho_l = self.fluid.rhomass()
            
            # Saturated vapor (Q=1)
            self.fluid.update(CP.PQ_INPUTS, p, 1.0)
            mu_g = self.fluid.viscosity()
            k_g = self.fluid.conductivity()
            rho_g = self.fluid.rhomass()
            
            return {
                'mu_l': mu_l, 'mu_g': mu_g,
                'k_l': k_l, 'k_g': k_g,
                'rho_l': rho_l, 'rho_g': rho_g
            }
        except Exception as ex:
            return None

    # TODO: Include in data generator
    def _compute_void_fraction(self, quality, rho_l, rho_g):
        """
        Compute void fraction (a) from quality (x) using the homogeneous model.
        
        a = 1 / (1 + (1-x)/x * rho_g/rho_l)
        
        Parameters:
        -----------
        quality : float
            Vapor quality (mass fraction of vapor)
        rho_l : float
            Saturated liquid density [kg/m3]
        rho_g : float
            Saturated vapor density [kg/m3]
            
        Returns:
        --------
        alpha : float
            Void fraction (volume fraction of vapor)
        """
        if quality <= 0:
            return 0.0
        elif quality >= 1:
            return 1.0
        else:
            # Homogeneous model void fraction
            return 1.0 / (1.0 + (1.0 - quality) / quality * rho_g / rho_l)

    # TODO: Include in data generator
    def _compute_twophase_viscosity(self, quality, mu_l, mu_g, rho_l=None, rho_g=None, alpha=None):
        """
        Compute two-phase mixture viscosity using selected mixing model.
        
        Parameters:
        -----------
        quality : float
            Vapor quality (mass fraction)
        mu_l : float
            Saturated liquid viscosity [Pas]
        mu_g : float
            Saturated vapor viscosity [Pas]
        rho_l, rho_g : float, optional
            Saturated densities [kg/m3] (needed for some models)
        alpha : float, optional
            Void fraction (needed for some models)
            
        Returns:
        --------
        mu_mix : float
            Two-phase mixture viscosity [Pas]
        """
        x = quality
        
        if self.viscosity_mixing_model == "mcadams":
            # McAdams et al. (1942) - Reciprocal average (recommended for most cases)
            if x <= 0:
                return mu_l
            elif x >= 1:
                return mu_g
            else:
                return 1.0 / (x / mu_g + (1.0 - x) / mu_l)
                
        elif self.viscosity_mixing_model == "cicchitti":
            # Cicchitti et al. (1960) - Mass-weighted average
            return x * mu_g + (1.0 - x) * mu_l
            
        elif self.viscosity_mixing_model == "dukler":
            # Dukler et al. (1964) - Volume-weighted (needs void fraction)
            if alpha is None:
                alpha = self._compute_void_fraction(x, rho_l, rho_g)
            return alpha * mu_g + (1.0 - alpha) * mu_l
            
        else:
            # Default to McAdams
            if x <= 0:
                return mu_l
            elif x >= 1:
                return mu_g
            else:
                return 1.0 / (x / mu_g + (1.0 - x) / mu_l)

    def _compute_twophase_conductivity(self, quality, k_l, k_g, rho_l=None, rho_g=None, alpha=None):
        """
        Compute two-phase mixture thermal conductivity using selected mixing model.
        
        Parameters:
        -----------
        quality : float
            Vapor quality (mass fraction)
        k_l : float
            Saturated liquid thermal conductivity [W/(mK)]
        k_g : float
            Saturated vapor thermal conductivity [W/(mK)]
        rho_l, rho_g : float, optional
            Saturated densities [kg/m3]
        alpha : float, optional
            Void fraction
            
        Returns:
        --------
        k_mix : float
            Two-phase mixture thermal conductivity [W/(mK)]
        """
        x = quality
        
        if alpha is None and rho_l is not None and rho_g is not None:
            alpha = self._compute_void_fraction(x, rho_l, rho_g)
        
        if self.conductivity_mixing_model == "volume":
            # Volume-weighted average (recommended)
            if alpha is None:
                # Fallback to mass-weighted if void fraction unavailable
                return x * k_g + (1.0 - x) * k_l
            return alpha * k_g + (1.0 - alpha) * k_l
            
        elif self.conductivity_mixing_model == "mass":
            # Mass-weighted average
            return x * k_g + (1.0 - x) * k_l
            
        else:
            # Default to volume-weighted
            if alpha is None:
                return x * k_g + (1.0 - x) * k_l
            return alpha * k_g + (1.0 - alpha) * k_l
    

    def SetDensityEnergyGrid(self, rho_min, rho_max, e_min, e_max,
                             N_rho=100, N_e=100):
        """
        Define the density-energy grid for the LUT.
        """
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.e_min = e_min
        self.e_max = e_max
        self.N_rho = N_rho
        self.N_e = N_e

        # Create uniform grid
        rho_range = np.linspace(rho_min, rho_max, N_rho)
        e_range = np.linspace(e_min, e_max, N_e)
        self.rho_grid, self.e_grid = np.meshgrid(rho_range, e_range)

        print(f"Grid Configuration:")
        print(f"  Density: [{rho_min:.2f}, {rho_max:.2f}] kg/m3 ({N_rho} points)")
        print(f"  Energy:  [{e_min:.0f}, {e_max:.0f}] J/kg ({N_e} points)")
        print(f"  Total grid points: {N_rho * N_e:,}")
        print()

    def ComputeTableData(self):
        """
        Compute thermodynamic properties at all grid points.
        Uses finite differences for all derivative calculations to ensure
        consistency across single-phase and two-phase regions.
        """
        print("Computing thermodynamic properties...")

        shape = self.rho_grid.shape
        n_points = shape[0] * shape[1]

        # Initialize storage arrays
        self.s_data = np.zeros(shape)
        self.T_data = np.zeros(shape)
        self.p_data = np.zeros(shape)
        self.Q_data = np.full(shape, -1.0)  # -1 = single phase

        # Entropy derivatives
        self.dsdrho_e_data = np.zeros(shape)
        self.dsde_rho_data = np.zeros(shape)
        self.d2sdrho2_data = np.zeros(shape)
        self.d2sde2_data = np.zeros(shape)
        self.d2sdedrho_data = np.zeros(shape)
        
        # Transport properties
        self.viscosity_data = np.zeros(shape)
        self.conductivity_data = np.zeros(shape)

        # Validity mask
        self.valid_mask = np.zeros(shape, dtype=bool)

        # Flatten for iteration
        rho_flat = self.rho_grid.flatten()
        e_flat = self.e_grid.flatten()

        success_count = 0
        twophase_count = 0
        fd_fallback_count = 0

        for i in tqdm(range(n_points), desc="Evaluating"):
            rho = rho_flat[i]
            e = e_flat[i]
            idx_2d = np.unravel_index(i, shape)

            try:
                # Update fluid state
                self.fluid.update(CP.DmassUmass_INPUTS, rho, e)
                phase = self.fluid.phase()

                # Skip critical point
                if phase == CoolP.iphase_critical_point:
                    continue

                # Basic properties
                s = self.fluid.smass()
                T = self.fluid.T()
                p = self.fluid.p()
                
                self.s_data[idx_2d] = s
                self.T_data[idx_2d] = T
                self.p_data[idx_2d] = p

                # Check if two-phase
                is_twophase = (phase == CoolP.iphase_twophase)
                if is_twophase:
                    self.Q_data[idx_2d] = self.fluid.Q()
                    twophase_count += 1
                
                # Compute transport properties
                if is_twophase:
                    # Two-phase: use mixing rules based on saturated properties
                    quality = self.fluid.Q()
                    sat_props = self._get_saturated_transport_properties(p, T)
                    
                    if sat_props is not None:
                        # Compute void fraction for volume-weighted models
                        alpha = self._compute_void_fraction(
                            quality, sat_props['rho_l'], sat_props['rho_g'])
                        
                        # Mixture viscosity
                        self.viscosity_data[idx_2d] = self._compute_twophase_viscosity(
                            quality, sat_props['mu_l'], sat_props['mu_g'],
                            sat_props['rho_l'], sat_props['rho_g'], alpha)
                        
                        # Mixture thermal conductivity
                        self.conductivity_data[idx_2d] = self._compute_twophase_conductivity(
                            quality, sat_props['k_l'], sat_props['k_g'],
                            sat_props['rho_l'], sat_props['rho_g'], alpha)
                    else:
                        # Fallback: try to get properties at current state (might fail)
                        try:
                            self.fluid.update(CP.DmassUmass_INPUTS, rho, e)
                            self.viscosity_data[idx_2d] = self.fluid.viscosity()
                            self.conductivity_data[idx_2d] = self.fluid.conductivity()
                        except:
                            # Last resort: interpolate based on quality
                            self.viscosity_data[idx_2d] = 1e-5  # Reasonable default
                            self.conductivity_data[idx_2d] = 0.1  # Reasonable default
                else:
                    # Single-phase: get directly from CoolProp
                    self.viscosity_data[idx_2d] = self.fluid.viscosity()
                    self.conductivity_data[idx_2d] = self.fluid.conductivity()

                # Compute derivatives
                use_fd = is_twophase  # Always use FD for two-phase
                
                if not use_fd:
                    # Try CoolProp analytical derivatives for single-phase
                    try:
                        self.dsdrho_e_data[idx_2d] = self.fluid.first_partial_deriv(
                            CP.iSmass, CP.iDmass, CP.iUmass)
                        self.dsde_rho_data[idx_2d] = self.fluid.first_partial_deriv(
                            CP.iSmass, CP.iUmass, CP.iDmass)
                        self.d2sdrho2_data[idx_2d] = self.fluid.second_partial_deriv(
                            CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
                        self.d2sde2_data[idx_2d] = self.fluid.second_partial_deriv(
                            CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
                        self.d2sdedrho_data[idx_2d] = self.fluid.second_partial_deriv(
                            CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
                    except:
                        use_fd = True  # Fall back to FD
                
                if use_fd:
                    # Compute derivatives via finite differences
                    derivs = self._compute_derivatives_fd(rho, e, s)
                    
                    if derivs is None:
                        # Cannot compute derivatives - skip this point
                        continue
                    
                    self.dsdrho_e_data[idx_2d] = derivs['dsdrho_e']
                    self.dsde_rho_data[idx_2d] = derivs['dsde_rho']
                    self.d2sdrho2_data[idx_2d] = derivs['d2sdrho2']
                    self.d2sde2_data[idx_2d] = derivs['d2sde2']
                    self.d2sdedrho_data[idx_2d] = derivs['d2sdedrho']
                    fd_fallback_count += 1

                self.valid_mask[idx_2d] = True
                success_count += 1

            except Exception as ex:
                # Point failed - leave as invalid
                pass

        print(f"\nResults:")
        print(f"  Valid points: {success_count:,} / {n_points:,} ({100 * success_count / n_points:.1f}%)")
        print(f"  Two-phase points: {twophase_count:,} ({100 * twophase_count / max(1,success_count):.1f}% of valid)")
        print(f"  Points using finite differences: {fd_fallback_count:,}")
        print()

        # Sanity check on derivatives
        self._validate_derivatives()
        
        # Validate transport properties
        self._validate_transport_properties()

        return success_count, twophase_count

    def _validate_derivatives(self):
        """
        Perform sanity checks on computed derivatives.
        """
        print("Validating derivatives...")
        
        valid_s = self.s_data[self.valid_mask]
        valid_dsde = self.dsde_rho_data[self.valid_mask]
        valid_T = self.T_data[self.valid_mask]
        
        # ds/de|rho should be approximately 1/T
        expected_dsde = 1.0 / valid_T
        rel_error = np.abs(valid_dsde - expected_dsde) / np.abs(expected_dsde + 1e-10)
        
        median_error = np.median(rel_error)
        max_error = np.max(rel_error)
        
        print(f"  ds/de vs 1/T check:")
        print(f"    Median relative error: {median_error:.2e}")
        print(f"    Max relative error: {max_error:.2e}")
        
        # Check for zeros in first derivatives (problematic)
        zero_dsdrho = np.sum(np.abs(self.dsdrho_e_data[self.valid_mask]) < 1e-20)
        zero_dsde = np.sum(np.abs(self.dsde_rho_data[self.valid_mask]) < 1e-20)
        
        if zero_dsdrho > 0 or zero_dsde > 0:
            print(f"  WARNING: Found {zero_dsdrho} zeros in ds/drho, {zero_dsde} zeros in ds/de")
        else:
            print(f"  No zero first derivatives found")
        
        print()

    def _validate_transport_properties(self):
        """
        Perform sanity checks on computed transport properties.
        """
        print("Validating transport properties...")
        
        valid_mu = self.viscosity_data[self.valid_mask]
        valid_k = self.conductivity_data[self.valid_mask]
        valid_Q = self.Q_data[self.valid_mask]
        
        # Separate single-phase and two-phase
        mask_1p = valid_Q < 0
        mask_2p = (valid_Q >= 0) & (valid_Q <= 1)
        
        # Viscosity statistics
        print(f"  Dynamic Viscosity (ViscosityDyn):")
        print(f"    Overall range: [{np.min(valid_mu):.2e}, {np.max(valid_mu):.2e}] Pas")
        if np.sum(mask_1p) > 0:
            print(f"    Single-phase: [{np.min(valid_mu[mask_1p]):.2e}, {np.max(valid_mu[mask_1p]):.2e}] Pas")
        if np.sum(mask_2p) > 0:
            print(f"    Two-phase:    [{np.min(valid_mu[mask_2p]):.2e}, {np.max(valid_mu[mask_2p]):.2e}] Pas")
        
        # Check for problematic values
        zero_mu = np.sum(valid_mu <= 0)
        if zero_mu > 0:
            print(f"    WARNING: Found {zero_mu} non-positive viscosity values!")
        
        # Thermal conductivity statistics
        print(f"  Thermal Conductivity (Conductivity):")
        print(f"    Overall range: [{np.min(valid_k):.4f}, {np.max(valid_k):.4f}] W/(mK)")
        if np.sum(mask_1p) > 0:
            print(f"    Single-phase: [{np.min(valid_k[mask_1p]):.4f}, {np.max(valid_k[mask_1p]):.4f}] W/(mK)")
        if np.sum(mask_2p) > 0:
            print(f"    Two-phase:    [{np.min(valid_k[mask_2p]):.4f}, {np.max(valid_k[mask_2p]):.4f}] W/(mK)")
        
        # Check for problematic values
        zero_k = np.sum(valid_k <= 0)
        if zero_k > 0:
            print(f"    WARNING: Found {zero_k} non-positive conductivity values!")
        else:
            print(f"    All transport properties are positive")
        
        print()

    def CreateTriangulation(self):
        """
        Create Delaunay triangulation of valid grid points.
        """
        print("Creating Delaunay triangulation...")

        # Extract valid points
        rho_valid = self.rho_grid[self.valid_mask].flatten()
        e_valid = self.e_grid[self.valid_mask].flatten()

        # Stack as (N, 2) array
        self.table_nodes = np.column_stack([rho_valid, e_valid])

        # Create Delaunay triangulation
        tri = Delaunay(self.table_nodes)
        self.table_connectivity = tri.simplices

        # Identify hull nodes
        edges = np.vstack([tri.simplices[:, [0, 1]],
                           tri.simplices[:, [1, 2]],
                           tri.simplices[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        self.table_hullnodes = np.unique(boundary_edges.flatten())

        print(f"  Triangulation nodes: {len(self.table_nodes):,}")
        print(f"  Triangles: {len(self.table_connectivity):,}")
        print(f"  Hull nodes: {len(self.table_hullnodes):,}")
        print()

    def PrepareTableData(self):
        """
        Prepare data arrays for writing to file.
        """
        print("Preparing table data for export...")

        n_nodes = len(self.table_nodes)

        # SU2-required variables (always included)
        self.table_data = {
            's': self.s_data[self.valid_mask].flatten(),
            'dsdrho_e': self.dsdrho_e_data[self.valid_mask].flatten(),
            'dsde_rho': self.dsde_rho_data[self.valid_mask].flatten(),
            'd2sdrho2': self.d2sdrho2_data[self.valid_mask].flatten(),
            'd2sdedrho': self.d2sdedrho_data[self.valid_mask].flatten(),
            'd2sde2': self.d2sde2_data[self.valid_mask].flatten(),
            'ViscosityDyn': self.viscosity_data[self.valid_mask].flatten(),
            'Conductivity': self.conductivity_data[self.valid_mask].flatten(),
        }
        
        # Diagnostic variables (only if requested)
        if self.include_diagnostics:
            self.table_data['T'] = self.T_data[self.valid_mask].flatten()
            self.table_data['p'] = self.p_data[self.valid_mask].flatten()
            self.table_data['VaporQuality'] = self.Q_data[self.valid_mask].flatten()

        print(f"  Prepared {n_nodes:,} data points")
        print(f"  Transport properties: ViscosityDyn, Conductivity")
        print()

    def WriteTableFile(self, output_filepath):
        """
        Write table to SU2 Dragon format (.drg file).
        """
        print(f"Writing Dragon LUT file: {output_filepath}")

        # Determine which variables to write
        if self.include_diagnostics:
            vars_to_write = self.table_vars + self.diagnostic_vars
        else:
            vars_to_write = self.table_vars

        with open(output_filepath, "w", encoding='ascii') as fid:
            # Header
            fid.write("Dragon library\n\n")
            fid.write("<Header>\n\n")
            fid.write("[Version]\n1.0.1\n\n")

            fid.write("[Number of points]\n")
            fid.write("%i\n" % len(self.table_nodes))
            fid.write("\n")

            fid.write("[Number of triangles]\n")
            fid.write("%i\n" % len(self.table_connectivity))
            fid.write("\n")

            fid.write("[Number of hull points]\n")
            fid.write("%i\n" % len(self.table_hullnodes))
            fid.write("\n")

            fid.write("[Number of variables]\n%i\n\n" % (len(self.controlling_vars) + len(vars_to_write)))

            fid.write("[Variable names]\n")
            all_vars = self.controlling_vars + vars_to_write
            for iVar, Var in enumerate(all_vars):
                fid.write(str(iVar + 1) + ":" + str(Var) + "\n")
            fid.write("\n")

            fid.write("</Header>\n\n")

            # Data section
            print("  Writing data points...")
            fid.write("<Data>\n")
            for iNode in range(len(self.table_nodes)):
                fid.write("\t".join("%+.14e" % cv for cv in self.table_nodes[iNode, :]))
                for var in vars_to_write:
                    fid.write("\t%+.14e" % self.table_data[var][iNode])
                fid.write("\n")
            fid.write("</Data>\n\n")

            # Connectivity section
            print("  Writing connectivity...")
            fid.write("<Connectivity>\n")
            for iCell in range(len(self.table_connectivity)):
                fid.write("\t".join("%i" % (c + 1) for c in self.table_connectivity[iCell, :]) + "\n")
            fid.write("</Connectivity>\n\n")

            # Hull section
            print("  Writing hull nodes...")
            fid.write("<Hull>\n")
            for iCell in range(len(self.table_hullnodes)):
                fid.write("%i\n" % (self.table_hullnodes[iCell] + 1))
            fid.write("</Hull>\n\n")

        print(f"  Table file written successfully!")
        print(f"  Variables: {all_vars}")
        print()

    # TODO: interactive visualization
    def VisualizeTables(self, save_dir="./"):
        """
        Create visualization plots of the table data.
        """
        print("Creating visualization plots...")

        T_vis = self.T_data[self.valid_mask].flatten()
        p_vis = self.p_data[self.valid_mask].flatten()
        Q_vis = self.Q_data[self.valid_mask].flatten()
        mu_vis = self.viscosity_data[self.valid_mask].flatten()
        k_vis = self.conductivity_data[self.valid_mask].flatten()

        fig = plt.figure(figsize=(18, 14))

        mask_1p = Q_vis < 0
        mask_2p = (Q_vis >= 0) & (Q_vis <= 1)

        # Row 1: Basic properties
        ax1 = fig.add_subplot(341, projection='3d')
        if np.sum(mask_1p) > 0:
            ax1.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        self.table_data['s'][mask_1p], c='blue', s=1, alpha=0.3, label='1p')
        if np.sum(mask_2p) > 0:
            ax1.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        self.table_data['s'][mask_2p], c='red', s=2, alpha=0.6, label='2p')
        ax1.set_xlabel('rho [kg/m3]')
        ax1.set_ylabel('e [J/kg]')
        ax1.set_zlabel('s [J/(kgK)]')
        ax1.set_title('Entropy')
        ax1.legend()

        ax2 = fig.add_subplot(342, projection='3d')
        if np.sum(mask_1p) > 0:
            ax2.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        T_vis[mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax2.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        T_vis[mask_2p], c='red', s=2, alpha=0.6)
        ax2.set_xlabel('rho [kg/m3]')
        ax2.set_ylabel('e [J/kg]')
        ax2.set_zlabel('T [K]')
        ax2.set_title('Temperature')

        ax3 = fig.add_subplot(343, projection='3d')
        if np.sum(mask_1p) > 0:
            ax3.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        p_vis[mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax3.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        p_vis[mask_2p], c='red', s=2, alpha=0.6)
        ax3.set_xlabel('rho [kg/m3]')
        ax3.set_ylabel('e [J/kg]')
        ax3.set_zlabel('p [Pa]')
        ax3.set_title('Pressure')

        # Row 1, col 4: Quality
        ax4 = fig.add_subplot(344, projection='3d')
        if np.sum(mask_2p) > 0:
            sc = ax4.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                            Q_vis[mask_2p], c=Q_vis[mask_2p], cmap='coolwarm', s=2, alpha=0.6)
            plt.colorbar(sc, ax=ax4, shrink=0.5)
        ax4.set_xlabel('rho [kg/m3]')
        ax4.set_ylabel('e [J/kg]')
        ax4.set_zlabel('Q [-]')
        ax4.set_title('Vapor Quality (2p only)')

        # Row 2: Derivatives
        ax5 = fig.add_subplot(345, projection='3d')
        if np.sum(mask_1p) > 0:
            ax5.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        self.table_data['dsdrho_e'][mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax5.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        self.table_data['dsdrho_e'][mask_2p], c='red', s=2, alpha=0.6)
        ax5.set_xlabel('rho')
        ax5.set_ylabel('e')
        ax5.set_zlabel('ds/drho_e')
        ax5.set_title('ds/drho_e')

        ax6 = fig.add_subplot(346, projection='3d')
        if np.sum(mask_1p) > 0:
            ax6.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        self.table_data['dsde_rho'][mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax6.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        self.table_data['dsde_rho'][mask_2p], c='red', s=2, alpha=0.6)
        ax6.set_xlabel('rho')
        ax6.set_ylabel('e')
        ax6.set_zlabel('ds/de_rho')
        ax6.set_title('ds/de_rho (~1/T)')

        ax7 = fig.add_subplot(347, projection='3d')
        if np.sum(mask_1p) > 0:
            ax7.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        self.table_data['d2sdrho2'][mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax7.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        self.table_data['d2sdrho2'][mask_2p], c='red', s=2, alpha=0.6)
        ax7.set_xlabel('rho')
        ax7.set_ylabel('e')
        ax7.set_zlabel('d2s/drho2')
        ax7.set_title('d2s/drho2')

        ax8 = fig.add_subplot(348, projection='3d')
        if np.sum(mask_1p) > 0:
            ax8.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        self.table_data['d2sde2'][mask_1p], c='blue', s=1, alpha=0.3)
        if np.sum(mask_2p) > 0:
            ax8.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        self.table_data['d2sde2'][mask_2p], c='red', s=2, alpha=0.6)
        ax8.set_xlabel('rho')
        ax8.set_ylabel('e')
        ax8.set_zlabel('d2s/de2')
        ax8.set_title('d2s/de2')

        # Row 3: Transport properties
        ax9 = fig.add_subplot(349, projection='3d')
        if np.sum(mask_1p) > 0:
            ax9.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        mu_vis[mask_1p]*1e6, c='blue', s=1, alpha=0.3, label='1p')
        if np.sum(mask_2p) > 0:
            ax9.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        mu_vis[mask_2p]*1e6, c='red', s=2, alpha=0.6, label='2p')
        ax9.set_xlabel('rho [kg/m3]')
        ax9.set_ylabel('e [J/kg]')
        ax9.set_zlabel('mu [uPas]')
        ax9.set_title('Dynamic Viscosity')
        ax9.legend()

        ax10 = fig.add_subplot(3, 4, 10, projection='3d')
        if np.sum(mask_1p) > 0:
            ax10.scatter(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                        k_vis[mask_1p]*1e3, c='blue', s=1, alpha=0.3, label='1p')
        if np.sum(mask_2p) > 0:
            ax10.scatter(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                        k_vis[mask_2p]*1e3, c='red', s=2, alpha=0.6, label='2p')
        ax10.set_xlabel('rho [kg/m3]')
        ax10.set_ylabel('e [J/kg]')
        ax10.set_zlabel('k [mW/(mK)]')
        ax10.set_title('Thermal Conductivity')
        ax10.legend()

        # Row 3: Transport properties vs Quality (for 2-phase region)
        ax11 = fig.add_subplot(3, 4, 11)
        if np.sum(mask_2p) > 0:
            ax11.scatter(Q_vis[mask_2p], mu_vis[mask_2p]*1e6, c='purple', s=2, alpha=0.5)
        ax11.set_xlabel('Vapor Quality [-]')
        ax11.set_ylabel('mu [uPas]')
        ax11.set_title('Viscosity vs Quality (2p)')
        ax11.grid(True, alpha=0.3)

        ax12 = fig.add_subplot(3, 4, 12)
        if np.sum(mask_2p) > 0:
            ax12.scatter(Q_vis[mask_2p], k_vis[mask_2p]*1e3, c='orange', s=2, alpha=0.5)
        ax12.set_xlabel('Vapor Quality [-]')
        ax12.set_ylabel('k [mW/(mK)]')
        ax12.set_title('Conductivity vs Quality (2p)')
        ax12.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'LUT_{self.fluid_name}_visualization.png')
        plt.savefig(save_path, dpi=150)
        print(f"  Visualization saved: {save_path}")
        plt.close()

        # Triangulation plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.triplot(self.table_nodes[:, 0], self.table_nodes[:, 1],
                   self.table_connectivity, 'k-', linewidth=0.3, alpha=0.3)

        if np.sum(mask_1p) > 0:
            ax.plot(self.table_nodes[mask_1p, 0], self.table_nodes[mask_1p, 1],
                    'b.', markersize=1, alpha=0.3, label='Single-phase')
        if np.sum(mask_2p) > 0:
            ax.plot(self.table_nodes[mask_2p, 0], self.table_nodes[mask_2p, 1],
                    'r.', markersize=2, alpha=0.6, label='Two-phase')

        ax.plot(self.table_nodes[self.table_hullnodes, 0],
                self.table_nodes[self.table_hullnodes, 1],
                'go', markersize=3, label='Hull nodes')

        ax.set_xlabel('Density [kg/m3]')
        ax.set_ylabel('Energy [J/kg]')
        ax.set_title(f'Triangulation for {self.fluid_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(save_dir, f'LUT_{self.fluid_name}_triangulation.png')
        plt.savefig(save_path, dpi=150)
        print(f"  Triangulation plot saved: {save_path}")
        plt.close()
        print()


def main():
    generator = TwoPhase_LUT_Generator(
        fluid_name="CO2",
        EoS="HEOS",
        include_diagnostics=False
    )

    generator.SetDensityEnergyGrid(
        rho_min=20,
        rho_max=400.0,
        e_min=300000,
        e_max=450000,
        N_rho=500,
        N_e=500
    )

    generator.ComputeTableData()
    generator.CreateTriangulation()
    generator.PrepareTableData()
    generator.WriteTableFile("./LUT_CO2_2phase.drg")
    generator.VisualizeTables()

    print("=" * 80)
    print("LUT GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
