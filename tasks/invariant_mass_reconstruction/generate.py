"""
Task: invariant_mass_reconstruction
Description: <FILL-THIS-IN>

Author: Ondrej Theiner
"""

from src.task import Task
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from dataclasses import dataclass
import pandas as pd


class InvariantMassReconstruction(Task):
    """
    <FILL-THIS-IN> DESCRIBE WHAT THIS TASK SIMULATES AND WHAT THE MODEL MUST DO
    """
    # ---------------------------------------------------------------------------
    # Fourvector dataclass and kinematics utilities.
    # ---------------------------------------------------------------------------
    @dataclass
    class FourVec:
        E: np.ndarray
        px: np.ndarray
        py: np.ndarray
        pz: np.ndarray

        @property
        def pT(self):
            return np.sqrt(self.px**2 + self.py**2)

        @property
        def phi(self):
            return np.arctan2(self.py, self.px)

        @property
        def eta(self):
            p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
            return 0.5 * np.log((p + self.pz) / (p - self.pz))

    # ---------------------------------------------------------------------------
    # Mass spectrum: BW (Cauchy) + exponential mixture, on a finite window.
    # ---------------------------------------------------------------------------
    @staticmethod
    def sample_true_masses(n, M, Gamma, lam, f, m_min, m_max, rng):
        """Sample n true masses from a mixture of a Breit-Wigner (Cauchy) signal,
        to produce the desired signal fraction f, and an exponential background, 
        truncated to [m_min, m_max]."""
        is_signal = rng.random(n) < f
        n_sig = int(is_signal.sum())
        n_bg = n - n_sig

        # Truncated Cauchy via inverse CDF.
        a = 0.5 + np.arctan(2 * (m_min - M) / Gamma) / np.pi
        b = 0.5 + np.arctan(2 * (m_max - M) / Gamma) / np.pi
        u = rng.uniform(a, b, size=n_sig)
        m_sig = M + 0.5 * Gamma * np.tan(np.pi * (u - 0.5))

        # Truncated exponential via inverse CDF.
        u = rng.uniform(size=n_bg)
        Z = np.exp(-lam * m_min) - np.exp(-lam * m_max)
        m_bg = -np.log(np.exp(-lam * m_min) - u * Z) / lam

        m = np.empty(n)
        m[is_signal] = m_sig
        m[~is_signal] = m_bg
        return m, is_signal
    
    # ---------------------------------------------------------------------------
    # Sample transverse momentum of parent particle
    # ---------------------------------------------------------------------------
    @staticmethod
    def sample_parent_kinematics(n, pT_scale, eta_max, rng):
        """Sample n parent (pT, eta, phi) with pT ~ Exp(1/pT_scale),
        eta uniform in [-eta_max, eta_max], phi uniform in [0, 2*pi)."""
        pT = rng.exponential(scale=pT_scale, size=n)
        eta = rng.uniform(-eta_max, eta_max, size=n)
        phi = rng.uniform(0.0, 2 * np.pi, size=n)
        return pT, eta, phi
    
    # ---------------------------------------------------------------------------
    # Decay particle isotropicaly in CM frame into two daughters, boost to lab frame.
    # ---------------------------------------------------------------------------
    def two_body_decay(self, parent_mass, parent_pT, parent_eta, parent_phi, daughter_mass, rng):
        """Decay a parent of given mass and (pT, eta, phi) isotropically in its
        rest frame into two equal-mass daughters. Returns two FourVec objects in
        the lab frame (back-to-back in the rest frame, boosted out).

        Vectorized over events.
        """
        n = parent_mass.size

        # Parent 4-vector in lab.
        px_p = parent_pT * np.cos(parent_phi)
        py_p = parent_pT * np.sin(parent_phi)
        pz_p = parent_pT * np.sinh(parent_eta)
        E_p = np.sqrt(px_p**2 + py_p**2 + pz_p**2 + parent_mass**2)

        # Isotropic decay direction in rest frame.
        cos_th = rng.uniform(-1.0, 1.0, size=n)
        sin_th = np.sqrt(1.0 - cos_th**2)
        ph = rng.uniform(0.0, 2 * np.pi, size=n)

        # |p*| from 4-momentum conservation: E_CM/2 = M/2, so p* = sqrt((M/2)^2 - m_d^2).
        p_star = np.sqrt(np.clip((parent_mass / 2.0)**2 - daughter_mass**2, 0.0, None))
        p1x_s = p_star * sin_th * np.cos(ph)
        p1y_s = p_star * sin_th * np.sin(ph)
        p1z_s = p_star * cos_th
        E_s = np.sqrt(p_star**2 + daughter_mass**2)

        # Boost to lab. Beta = parent momentum / parent energy.
        bx = px_p / E_p
        by = py_p / E_p
        bz = pz_p / E_p
        b2 = bx**2 + by**2 + bz**2
        # Numerical guard for parents nearly at rest (won't happen with reasonable pT_scale).
        b2 = np.clip(b2, 0.0, 1.0 - 1e-12)
        gamma = 1.0 / np.sqrt(1.0 - b2)

        def boost(E_s, p1, p2, p3):
            # General Lorentz boost of (E_s, p1,p2,p3) by velocity (bx,by,bz).
            bp = bx * p1 + by * p2 + bz * p3
            # factor = (gamma-1)/b2, with safe limit when b2 -> 0.
            factor = np.where(b2 > 0, (gamma - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
            E_lab = gamma * (E_s + bp)
            px_lab = p1 + factor * bp * bx + gamma * bx * E_s
            py_lab = p2 + factor * bp * by + gamma * by * E_s
            pz_lab = p3 + factor * bp * bz + gamma * bz * E_s
            return E_lab, px_lab, py_lab, pz_lab

        E1, px1, py1, pz1 = boost(E_s, p1x_s, p1y_s, p1z_s)
        # Daughter 2 is back-to-back in rest frame.
        E2, px2, py2, pz2 = boost(E_s, -p1x_s, -p1y_s, -p1z_s)

        return self.FourVec(E1, px1, py1, pz1), self.FourVec(E2, px2, py2, pz2)
    
    # ---------------------------------------------------------------------------
    # Helix propagation: track radius, track circle center, layer intersections 
    # in transverse plane + z(s). Arc length along the track to pick the correct 
    # intersection when there are two.
    # ---------------------------------------------------------------------------
    @staticmethod
    def curvature_radius(pT_GeV, B_T):
        """Helix transverse radius rho in m for given pT (GeV) and B (T)."""
        return pT_GeV / (0.3 * B_T)
    
    @staticmethod
    def track_circle_center(rho, phi0, charge):
        """Center (xc, yc) of the track circle in the transverse plane, given that
        the track passes through the origin with initial momentum direction phi0
        and signed charge in {-1, +1}. The center is in the perpendicular  direction
        to the initial momentum direction, at distance rho, on the side determined 
        by the charge.

        Convention (B along +z): a positive charge curves clockwise viewed from +z,
        so the center is at angle (phi0 - pi/2). Negative charge: (phi0 + pi/2).
        """
        perp = phi0 - np.sign(charge) * np.pi / 2.0
        xc = rho * np.cos(perp)
        yc = rho * np.sin(perp)
        return xc, yc
    
    def propagate_to_layer(self, pT, phi0, eta, charge, R_layer, B):
        """Find where a track (originating at origin with initial transverse
        direction phi0, transverse momentum pT, pseudorapidity eta, charge in
        {-1,+1}) intersects a cylindrical layer at radius R_layer, in field B.

        Returns (x, y, z). Picks the "first forward" intersection along the
        track direction. If R_layer > 2*rho the track does not reach the layer;
        returns NaNs (caller should ensure radii are within reach).

        Vectorized over events.
        """
        rho = self.curvature_radius(pT, B)
        xc, yc = self.track_circle_center(rho, phi0, charge)

        # Two circles: track (center (xc,yc), radius rho) and layer (origin, R_layer).
        # Distance between centers is rho (since track passes through origin).
        # Intersection exists iff |R_layer - rho| <= rho <= R_layer + rho,
        # i.e. R_layer <= 2*rho.
        reachable = R_layer <= 2.0 * rho

        # Standard two-circle intersection formula.
        d = rho  # distance between centers
        # a = distance from origin (center of layer circle) to the chord midpoint
        a = (R_layer**2) / (2.0 * d)
        # h = half-chord length
        h2 = R_layer**2 - a**2
        h = np.sqrt(np.clip(h2, 0.0, None))

        # Midpoint of the chord, along the line from origin to (xc,yc).
        ux = xc / d
        uy = yc / d
        mx = a * ux
        my = a * uy

        # Perpendicular offset directions.
        px = -uy
        py = ux

        # Two candidate intersections.
        x_pos = mx + h * px
        y_pos = my + h * py
        x_neg = mx - h * px
        y_neg = my - h * py

        # Pick the one reached first by following the track from origin in
        # direction phi0. Compute signed arc length along the track for each
        # candidate; choose the smallest positive one.
        s_pos = self._arc_length_from_origin(x_pos, y_pos, xc, yc, rho, charge)
        s_neg = self._arc_length_from_origin(x_neg, y_neg, xc, yc, rho, charge)
        pick_pos = (s_pos > 0) & ((s_neg <= 0) | (s_pos <= s_neg))
        x = np.where(pick_pos, x_pos, x_neg)
        y = np.where(pick_pos, y_pos, y_neg)
        s = np.where(pick_pos, s_pos, s_neg)

        # z = z_vertex + s * cot(theta) = 0 + s * sinh(eta) / ... ; for a track
        # originating at origin with pseudorapidity eta, z grows linearly with
        # transverse arc length s as z = s * sinh(eta) / 1 ... wait, careful:
        # ds_3d / ds_T = 1 / sin(theta), and dz/ds_3d = cos(theta), so
        # dz/ds_T = cos(theta)/sin(theta) = cot(theta) = sinh(eta).
        z = s * np.sinh(eta)

        x = np.where(reachable, x, np.nan)
        y = np.where(reachable, y, np.nan)
        z = np.where(reachable, z, np.nan)
        return x, y, z
    
    @staticmethod
    def _arc_length_from_origin(x, y, xc, yc, rho, charge):
        """Signed transverse arc length from origin to (x,y) along the track
        circle, going forward (positive) along the initial direction phi0.
        """
        # Angle of origin and of (x,y) as seen from circle center.
        a0 = np.arctan2(0.0 - yc, 0.0 - xc)
        a1 = np.arctan2(y - yc, x - xc)
        # Going forward means increasing angle for negative charge (CCW)
        # and decreasing angle for positive charge (CW), in our convention.
        da = a1 - a0
        # Normalize to (-pi, pi].
        da = (da + np.pi) % (2 * np.pi) - np.pi
        # Forward direction sign:
        forward_sign = -np.sign(charge)  # negative charge -> CCW -> positive da
        # We want s = forward_sign * da * rho to be positive when going forward.
        s = forward_sign * da * rho
        return s
    
    # ---------------------------------------------------------------------------
    # Detector geometry in transverse plane (input specification image).
    # ---------------------------------------------------------------------------
    @staticmethod
    def plot_detector_transverse(layer_radii, ecal_radius, layer_names):
        """Transverse (x-y) cross-section showing detector geometry only —
        no hits. Used as an input specification image so the model knows the
        layer structure and radii before analysing the data.
        """
        tracker_colors = plt.cm.tab10(np.arange(len(layer_radii)))

        fig, ax = plt.subplots(figsize=(8, 6))

        legend_handles = []
        for r, name, color in zip(layer_radii, layer_names, tracker_colors):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color=color,
                                    linewidth=1.8, linestyle='--'))
            handle = plt.Line2D([], [], color=color, linewidth=1.8, linestyle='--',
                                label=f'{name}  :  r = {r:.3f} m')
            legend_handles.append(handle)

        ax.add_patch(plt.Circle((0, 0), ecal_radius, fill=False,
                                 color='firebrick', linewidth=2.5))
        legend_handles.append(plt.Line2D([], [], color='firebrick', linewidth=2.5,
                                          label=f'ECAL  —  r = {ecal_radius:.3f} m'))

        legend_handles.append(plt.Line2D([], [], marker='+', color='black',
                                          linestyle='None', markersize=10,
                                          markeredgewidth=1.5, label='vertex / origin'))
        ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5)

        lim = ecal_radius * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Detector transverse cross-section')
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0.9, title='Detector layers', borderaxespad=0)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------------------
    # Detector longitudinal cross-section (r-z plane).
    # ---------------------------------------------------------------------------
    @staticmethod
    def plot_detector_longitudinal(layer_radii, ecal_radius, layer_names, eta_max):
        """Longitudinal (r-z) cross-section of the detector barrel.
        Each layer is drawn as a horizontal line from -z_max to +z_max,
        where z_max = r * sinh(eta_max). Eta acceptance lines are shown
        as dashed guides from the origin.
        """
        tracker_colors = plt.cm.tab10(np.arange(len(layer_radii)))
        sinh_eta = np.sinh(eta_max)

        fig, ax = plt.subplots(figsize=(11, 4))

        legend_handles = []
        for r, name, color in zip(layer_radii, layer_names, tracker_colors):
            z_max = r * sinh_eta
            ax.plot([-z_max, z_max], [r, r], color=color, linewidth=1.8, linestyle='--')
            ax.text(0, r + 0.015, f'2z = {2*z_max:.2f} m',
                    fontsize=7, color=color, va='bottom', ha='center')
            legend_handles.append(plt.Line2D([], [], color=color, linewidth=1.8,
                                             linestyle='--', label=f'{name}  :  r = {r:.3f} m'))

        z_max_ecal = ecal_radius * sinh_eta
        ax.plot([-z_max_ecal, z_max_ecal], [ecal_radius, ecal_radius],
                color='firebrick', linewidth=2.5)
        ax.text(0, ecal_radius + 0.015, f'2z = {2*z_max_ecal:.1f} m',
                fontsize=7, color='firebrick', va='bottom', ha='center')
        legend_handles.append(plt.Line2D([], [], color='firebrick', linewidth=2.5,
                                          label=f'ECAL  :  r = {ecal_radius:.3f} m'))

        # Eta acceptance lines from origin.
        r_edge = ecal_radius * 1.05
        z_edge = r_edge * sinh_eta
        for sign in (1, -1):
            ax.plot([0, sign * z_edge], [0, r_edge], 'k--', linewidth=0.8, alpha=0.4)
        ax.text(z_edge * 0.5, r_edge * 0.5, f'η = {eta_max}', fontsize=7, alpha=0.6,
                va='bottom', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.text(-z_edge * 0.5, r_edge * 0.5, f'η = −{eta_max}', fontsize=7, alpha=0.6,
                ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

        ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5)
        legend_handles.append(plt.Line2D([], [], marker='+', color='black', linestyle='None',
                                          markersize=10, markeredgewidth=1.5,
                                          label='vertex / origin'))

        ax.set_xlabel('z [m]')
        ax.set_ylabel('r [m]')
        ax.set_ylim(-0.05, ecal_radius * 1.25)
        ax.set_xlim(-(z_max_ecal + 0.5), z_max_ecal + 0.5)
        ax.set_title('Detector longitudinal cross-section (r–z plane)')
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0.9, title='Detector layers', borderaxespad=0)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------------------
    # Assemble ground-truth DataFrames.
    # ---------------------------------------------------------------------------
    def assemble_dataframes(
        self,
        masses,
        per_event_mass_tolerance,
        is_signal,
        tracker_hits_d1,
        tracker_hits_d2,
        ecal_hit_d1,
        ecal_hit_d2,
        energy_d1,
        energy_d2,
        layer_names,
    ):
        """Build three ground-truth DataFrames from vectorized event arrays.

        Parameters
        ----------
        masses        : (n,) true invariant masses
        per_event_mass_tolerance: (n,) per-event mass tolerance value, e.g. 0.1 of true mass, to be used in evaluation
        is_signal     : (n,) bool
        tracker_hits_d1 : list of (x, y, z) tuples per tracker layer, each (n,)
        tracker_hits_d2 : same for daughter 2
        ecal_hit_d1   : (x, y, z) tuple, each (n,)
        ecal_hit_d2   : same for daughter 2
        energy_d1     : (n,) ECAL energy deposits for daughter 1
        energy_d2     : (n,) ECAL energy deposits for daughter 2
        layer_names   : list of str, one per tracker layer

        Returns
        -------
        df_events  : event_id, mass, is_signal, mass_min, mass_max
        df_tracker : event_id, d1_x_{layer}, d1_y_{layer}, d1_z_{layer},
                            d2_x_{layer}, d2_y_{layer}, d2_z_{layer}, ...
        df_ecal    : event_id, d1_x, d1_y, d1_z, d1_energy,
                            d2_x, d2_y, d2_z, d2_energy
        """

        n = len(masses)
        event_ids = np.arange(n)

        df_events = pd.DataFrame({
            'event_id': event_ids,
            'mass':     masses,
            'is_signal': is_signal,
            'mass_min': masses * (1.0 - per_event_mass_tolerance),
            'mass_max': masses * (1.0 + per_event_mass_tolerance),
        })

        tracker_data = {'event_id': event_ids}
        for name, (x1, y1, z1), (x2, y2, z2) in zip(layer_names, tracker_hits_d1, tracker_hits_d2):
            tracker_data[f'd1_x_{name}'] = x1
            tracker_data[f'd1_y_{name}'] = y1
            tracker_data[f'd1_z_{name}'] = z1
            tracker_data[f'd2_x_{name}'] = x2
            tracker_data[f'd2_y_{name}'] = y2
            tracker_data[f'd2_z_{name}'] = z2
        df_tracker = pd.DataFrame(tracker_data)

        xe1, ye1, ze1 = ecal_hit_d1
        xe2, ye2, ze2 = ecal_hit_d2
        df_ecal = pd.DataFrame({
            'event_id': event_ids,
            'd1_x':     xe1,   'd1_y': ye1,   'd1_z': ze1,   'd1_energy': energy_d1,
            'd2_x':     xe2,   'd2_y': ye2,   'd2_z': ze2,   'd2_energy': energy_d2,
        })

        return df_events, df_tracker, df_ecal

    # ---------------------------------------------------------------------------
    # Main data generating method.
    # ---------------------------------------------------------------------------
    def _generate(self):
        """


        Conventions
        -----------
        - Lengths in m, momenta/energies in GeV, magnetic field in tesla.
        - Magnetic field is uniform.
        - Decay vertex is fixed at the origin.
        """
        # ======= RANDOMNESS CONFIG =======
        rng = np.random.default_rng(self.seed)

        # ====== CONFIGURATION =======
        N_EVENTS =  self.get_params()['N_EVENTS']
        SIGNAL_FRACTION = self.get_params()['SIGNAL_FRACTION']
        MEAN_LAMBDA_BKG_DECAY = self.get_params()['MEAN_LAMBDA_BKG_DECAY']
        MEAN_GAMMA_SIGNAL = self.get_params()['MEAN_GAMMA_SIGNAL']
        M_MIN = self.get_params()['M_MIN']
        M_MAX = self.get_params()['M_MAX']
        M_SIG_MIN = self.get_params()['M_SIG_MIN']
        M_SIG_MAX = self.get_params()['M_SIG_MAX']
        PT_SCALE = self.get_params()['PT_SCALE']
        ETA_MAX = self.get_params()['ETA_MAX']
        MASS_TOLERANCE = self.get_params()['MASS_TOLERANCE']
        WIDTH_TOLERANCE = self.get_params()['WIDTH_TOLERANCE']
        DAUGHTER_MASS = self.get_params()['DAUGHTER_MASS']
        B_FIELD = self.get_params()['B_FIELD']
        LAYER_RADII = self.get_params()['LAYER_RADII']
        ECAL_RADIUS = self.get_params()['ECAL_RADIUS']
        PER_EVENT_MASS_TOLERANCE = self.get_params()['PER-EVENT_MASS_TOLERANCE']

        # ======= TASK GENERATION =======
        particle_mass = rng.uniform(M_SIG_MIN, M_SIG_MAX)
        particle_width = rng.uniform(MEAN_GAMMA_SIGNAL * 0.5, MEAN_GAMMA_SIGNAL * 1.5)
        background_decay_lambda = rng.uniform(MEAN_LAMBDA_BKG_DECAY * 0.5, MEAN_LAMBDA_BKG_DECAY * 1.5)

        # Put the values to dataframe
        df_new_particle_properties = pd.DataFrame([{
                                'particle_mass':      particle_mass,
                                'particle_mass_min':  particle_mass * (1.0 - MASS_TOLERANCE),
                                'particle_mass_max':  particle_mass * (1.0 + MASS_TOLERANCE),
                                'particle_width':     particle_width,
                                'particle_width_min': particle_width * (1.0 - WIDTH_TOLERANCE),
                                'particle_width_max': particle_width * (1.0 + WIDTH_TOLERANCE),
                            }]) 

        # Generate true masses for the events, and which ones are signal vs background.
        m, is_signal = self.sample_true_masses(
            n=N_EVENTS,
            M=particle_mass,
            Gamma=particle_width,
            lam=background_decay_lambda,
            f=SIGNAL_FRACTION,
            m_min=M_MIN,
            m_max=M_MAX,
            rng=rng
        )

        # For each event, sample the parent particle kinematics (pT, eta, phi)
        pT, eta, phi = self.sample_parent_kinematics(
            n=N_EVENTS,
            pT_scale=PT_SCALE,
            eta_max=ETA_MAX,
            rng=rng
        )

        # Decay each parent into two daughters, get their 4-vectors in the lab frame. The
        # daughters are back-to-back in the rest frame, but boosted out in the lab frame.
        d1, d2 = self.two_body_decay(
            parent_mass=m,
            parent_pT=pT,
            parent_eta=eta,
            parent_phi=phi,
            daughter_mass=DAUGHTER_MASS,
            rng=rng
        )

        # Assign charges: daughters are a particle-antiparticle pair.
        q1 = rng.choice(np.array([-1, 1]), size=N_EVENTS)
        q2 = -q1

        # Propagate each daughter through tracker layers; collect (x, y, z) per layer.
        track1_hits = [self.propagate_to_layer(d1.pT, d1.phi, d1.eta, q1, R, B_FIELD) for R in LAYER_RADII]
        track2_hits = [self.propagate_to_layer(d2.pT, d2.phi, d2.eta, q2, R, B_FIELD) for R in LAYER_RADII]
        
        # Propagate each daughter to ECAL layer; collect (x, y, z) and energy. Energy is just the
        # daughter's true energy, since we assume perfect ECAL measurement.
        track1_ecal_hits = self.propagate_to_layer(d1.pT, d1.phi, d1.eta, q1, ECAL_RADIUS, B_FIELD)
        track1_energy = d1.E
        track2_ecal_hits = self.propagate_to_layer(d2.pT, d2.phi, d2.eta, q2, ECAL_RADIUS, B_FIELD)
        track2_energy = d2.E

        # Save generated data to ground truth DataFrames.
        layer_names = [f"tracker layer {i+1}" for i in range(len(LAYER_RADII))]
        df_events, df_tracker, df_ecal = self.assemble_dataframes(
            masses=m,
            per_event_mass_tolerance=PER_EVENT_MASS_TOLERANCE,
            is_signal=is_signal,
            tracker_hits_d1=track1_hits,
            tracker_hits_d2=track2_hits,
            ecal_hit_d1=track1_ecal_hits,
            ecal_hit_d2=track2_ecal_hits,
            energy_d1=track1_energy,
            energy_d2=track2_energy,
            layer_names=layer_names,
        )

        # Input CSVs are anonymised — hits from both daughters mixed, no d1/d2 labels.
        event_ids = np.arange(N_EVENTS)
        tracker_parts = []
        for name, (x1, y1, z1), (x2, y2, z2) in zip(layer_names, track1_hits, track2_hits):
            tracker_parts.append(pd.DataFrame({'event_id': event_ids, 'layer': name, 'x': x1, 'y': y1, 'z': z1}))
            tracker_parts.append(pd.DataFrame({'event_id': event_ids, 'layer': name, 'x': x2, 'y': y2, 'z': z2}))
        df_tracker_input = pd.concat(tracker_parts).sort_values('event_id').reset_index(drop=True)

        xe1, ye1, ze1 = track1_ecal_hits
        xe2, ye2, ze2 = track2_ecal_hits
        df_ecal_input = pd.concat([
            pd.DataFrame({'event_id': event_ids, 'x': xe1, 'y': ye1, 'z': ze1, 'energy': track1_energy}),
            pd.DataFrame({'event_id': event_ids, 'x': xe2, 'y': ye2, 'z': ze2, 'energy': track2_energy}),
        ]).sort_values('event_id').reset_index(drop=True)

        df_tracker_input.to_csv(self.input_dir / 'tracker_hits.csv', index=False)
        df_ecal_input.to_csv(self.input_dir / 'ecal_hits.csv', index=False)

        # Generate input image with detector geometry
        fig = self.plot_detector_transverse(LAYER_RADII, ECAL_RADIUS, layer_names)
        fig.savefig(self.input_dir / 'detector_transverse.png', dpi=150)
        plt.close(fig)

        fig = self.plot_detector_longitudinal(LAYER_RADII, ECAL_RADIUS, layer_names, ETA_MAX)
        fig.savefig(self.input_dir / 'detector_longitudinal.png', dpi=150)
        plt.close(fig)

        # Mass spectrum histogram saved to ground truth for human reference.
        bins = np.linspace(M_MIN, M_MAX, 80)
        fig, ax = plt.subplots()
        ax.hist(m,             bins=bins, label='total',            histtype='step',       color='black')
        ax.hist(m[is_signal],  bins=bins, label='signal (BW)',      histtype='stepfilled', alpha=0.4)
        ax.hist(m[~is_signal], bins=bins, label='background (exp)', histtype='stepfilled', alpha=0.4)
        ax.set_xlabel('m [GeV]')
        ax.set_ylabel('events / bin')
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.ground_truth_dir / 'mass_spectrum.png', dpi=150)
        plt.close(fig)

        # Detector geometry dataframe: one row per layer (tracker + ECAL).
        df_detector = pd.DataFrame({
            'layer':     layer_names + ['calorimeter'],
            'radius':  list(LAYER_RADII) + [ECAL_RADIUS],
        })

        # Save ground truth DataFrames to self.ground_truth dictionary for later use in
        # evaluation and metarubric generation.
        self.ground_truth['detector'] = df_detector
        self.ground_truth['events'] = df_events
        self.ground_truth['tracker'] = df_tracker
        self.ground_truth['ecal'] = df_ecal
        self.ground_truth['new_particle_properties'] = df_new_particle_properties

        # Sampled ground truth to reduce maximum number of rubrics
        df_sampled = df_events if len(df_events) <= 200 else df_events.sample(n=200, random_state=self.seed)
        self.ground_truth['events_sampled'] = df_sampled

        self.ground_truth['misc'] = pd.DataFrame({'no_of_events_in_the_list' : [len(df_sampled)]})