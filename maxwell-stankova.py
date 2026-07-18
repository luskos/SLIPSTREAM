import numpy as np
import time
from dataclasses import dataclass
from collections import deque
import sys
import os

# ============================================================
# OPTIMIZED PARAMETERS
# ============================================================

@dataclass
class WarpParams:
    v: float = 2.5
    R0: float = 1.2
    Aw: float = 0.08
    wW: float = 0.15
    Ad: float = 0.05
    wD: float = 0.15
    epsBD: float = 0.02
    wBD: float = 0.584056828
    bBI: float = 2.319271223957
    grid_size: int = 48  # Reduced for speed
    L: float = 5.0
    dt: float = 0.001

# ============================================================
# LEAP-MORT BYPASS ENGINE (OPTIMIZED)
# ============================================================

class LEAPMORTBypass:
    def __init__(self, params: WarpParams):
        self.p = params
        self.grid = np.linspace(-params.L, params.L, params.grid_size)
        self.dx = self.grid[1] - self.grid[0]
        
        # Pre-compute grid for speed
        self.x, self.y, self.z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2 + 0.001**2)
        
        self.Phi = np.zeros((params.grid_size, params.grid_size, params.grid_size))
        
        # LEAP-MORT state (fixed buffer)
        self.frequency_buffer = np.zeros((32, 32))
        self.photorefractive_memory = np.zeros((32, 32))
        self.mort_angle = 0.0
        self.rpe_phase = 0.0
        self.buffer_fill = 0.0
        self.pfb_triggered = False
        
        # Stability metrics
        self.total_energy = 0.0
        self.ANEC = 0.0
        self.min_NEC = 0.0
        self.is_stable = True
        self.horizon_detected = False
        
        # Performance
        self.frame_times = deque(maxlen=100)
        self.fps = 0.0
        
        # History
        self.history = {
            'energy': deque(maxlen=1000),
            'ANEC': deque(maxlen=1000),
            'stability': deque(maxlen=1000),
        }
        
        # Energy tracking
        self.energy_target = -0.189028  # From 777.pdf
    
    def compute_phi(self, t: float) -> np.ndarray:
        r = self.r
        
        # Wormhole field
        Phi_WH = self.p.Aw * (1 - self.p.R0/r) * np.exp(-(r - self.p.R0)**2 / self.p.wW**2)
        
        # Drive field (moving bubble)
        x_shift = self.x - self.p.v * t
        Phi_Drive = self.p.Ad * x_shift * np.exp(-x_shift**2 / self.p.wD**2)
        
        # Brans-Dicke
        Phi_BD = self.p.epsBD * np.exp(-r**2 / self.p.wBD**2)
        
        # Born-Infeld
        Phi_BI = -1/self.p.bBI * np.sqrt(1 + (r/self.p.R0)**2)
        
        return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
    
    def compute_energy_conditions(self, Phi: np.ndarray) -> dict:
        h = self.dx
        grad = np.gradient(Phi, h, edge_order=2)
        grad_norm2 = grad[0]**2 + grad[1]**2 + grad[2]**2
        
        laplacian = (np.gradient(grad[0], h, edge_order=2)[0] + 
                     np.gradient(grad[1], h, edge_order=2)[1] + 
                     np.gradient(grad[2], h, edge_order=2)[2])
        
        G_tt = np.exp(4 * Phi) * (2 * laplacian - 3 * grad_norm2)
        T_00 = G_tt / (8 * np.pi)
        
        Phi_x = grad[0]
        Phi_xx = np.gradient(Phi_x, h, edge_order=2)[0]
        G_xx = (-2 * laplacian + grad_norm2) + 2 * Phi_x**2 - 2 * Phi_xx
        T_xx = G_xx / (8 * np.pi)
        
        NEC = T_00 + T_xx
        
        return {'T00': T_00, 'Txx': T_xx, 'NEC': NEC}
    
    def leap_mort_pipeline(self, field: np.ndarray, t: float) -> np.ndarray:
        # 1. PPCI Filter - sample to 32x32
        step = self.p.grid_size // 32
        if step < 1:
            step = 1
        
        # Take a 2D slice at y=0 for the pipeline
        center = self.p.grid_size // 2
        slice_2d = field[center, :, :]
        
        # Resize to 32x32
        h, w = slice_2d.shape
        step_h = max(1, h // 32)
        step_w = max(1, w // 32)
        sample = slice_2d[::step_h, ::step_w][:32, :32]
        
        # 2. Photon Flycatcher - accumulate
        self.frequency_buffer += np.abs(sample)
        
        # 3. PFB - check if buffer is FULL
        max_possible = np.max(self.frequency_buffer) * 32 * 32
        if max_possible > 0:
            self.buffer_fill = np.sum(self.frequency_buffer) / max_possible
        else:
            self.buffer_fill = 0
        
        if self.buffer_fill > 0.95:
            # PFB TRIGGERED! ALL photons emitted with IDENTICAL PHASE!
            self.pfb_triggered = True
            
            # Phase reset: all photons have same phase
            coherent = np.ones_like(self.frequency_buffer) * np.mean(self.frequency_buffer)
            self.frequency_buffer *= 0  # Reset buffer
            
            # 4. UCM - amplify (1200x)
            coherent *= 1200.0
            
            # 5. MORT - Yin-Yang modulation
            self.mort_angle += 0.1 * t
            for i in range(32):
                for j in range(32):
                    angle = np.arctan2(j - 16, i - 16)
                    phase_shift = 0.0 if (angle - self.mort_angle) % (2*np.pi) < np.pi else np.pi
                    coherent[i, j] *= (1 + 0.5 * np.cos(2*np.pi * (i/16 + j/16 + t)))
            
            # 6. Photorefractive - diffusion (Kukhtarev model)
            laplacian = np.zeros_like(coherent)
            laplacian[1:-1, 1:-1] = (coherent[2:, 1:-1] + coherent[:-2, 1:-1] +
                                      coherent[1:-1, 2:] + coherent[1:-1, :-2] -
                                      4 * coherent[1:-1, 1:-1])
            self.photorefractive_memory += (0.1 * coherent - self.photorefractive_memory / 0.1) * 0.01
            self.photorefractive_memory += 0.01 * laplacian * 0.01
            
            # 7. RPE - Russian doll rings (frame interpolation)
            self.rpe_phase += 0.1 * t
            for n in range(1, 4):
                phase = 2 * np.pi * (np.arange(32) / 32) * n + self.rpe_phase
                coherent += 0.05 * np.sin(phase[:, None] + phase[None, :])
            
            return coherent
        else:
            self.pfb_triggered = False
            return sample
    
    def step(self, t: float) -> dict:
        start_time = time.time()
        
        # 1. Compute metric
        self.Phi = self.compute_phi(t)
        
        # 2. Energy conditions
        energy = self.compute_energy_conditions(self.Phi)
        
        # 3. LEAP-MORT pipeline
        stabilized_field = self.leap_mort_pipeline(self.Phi, t)
        
        # 4. Stability metrics
        center = self.p.grid_size // 2
        self.total_energy = np.sum(energy['T00']) * self.dx**3
        self.ANEC = np.sum(energy['NEC'][center, :, center]) * self.dx
        self.min_NEC = np.min(energy['NEC'])
        
        # 5. Horizon check
        g_tt = np.exp(4 * self.Phi) * (1 - self.p.v**2 * np.exp(-4 * self.Phi))
        self.horizon_detected = np.any(np.abs(g_tt) < 0.01)
        
        # 6. Stability criteria
        self.is_stable = (self.ANEC > 0) and (self.total_energy > -10) and not self.horizon_detected
        
        # 7. Auto-stabilization feedback
        if not self.is_stable:
            # Adjust parameters to reach target energy
            energy_error = (self.total_energy - self.energy_target) / abs(self.energy_target)
            self.p.Ad *= (1 + 0.001 * energy_error)
            self.p.Aw *= (1 + 0.001 * energy_error * 0.5)
            self.clamp_params()
        
        # Store history
        self.history['energy'].append(self.total_energy)
        self.history['ANEC'].append(self.ANEC)
        self.history['stability'].append(1.0 if self.is_stable else 0.0)
        
        # Performance
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 10:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return {
            'total_energy': self.total_energy,
            'ANEC': self.ANEC,
            'min_NEC': self.min_NEC,
            'is_stable': self.is_stable,
            'horizon_detected': self.horizon_detected,
            'fps': self.fps,
            'step_time_ms': elapsed * 1000,
            'ad': self.p.Ad,
            'aw': self.p.Aw,
            'buffer_fill': self.buffer_fill * 100,
            'pfb_triggered': self.pfb_triggered,
        }
    
    def clamp_params(self):
        self.p.Ad = np.clip(self.p.Ad, 0.01, 0.1)
        self.p.Aw = np.clip(self.p.Aw, 0.01, 0.15)
        self.p.v = np.clip(self.p.v, 1.1, 3.0)

# ============================================================
# ASCII VISUALIZATION
# ============================================================

def draw_field(field: np.ndarray, size: int = 35) -> str:
    center = field.shape[0] // 2
    slice_2d = field[center, :, :]
    
    h, w = slice_2d.shape
    step = max(1, min(h // size, w // size))
    small = slice_2d[::step, ::step]
    
    # Normalize
    small = (small - small.min()) / (small.max() - small.min() + 1e-10)
    
    chars = " .:-=+*#%@"
    lines = []
    for i in range(min(small.shape[0], size)):
        line = ""
        for j in range(min(small.shape[1], size)):
            idx = int(small[i, j] * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    
    return "\n".join(lines)

def draw_hud(diag: dict, params: WarpParams) -> str:
    bars = 40
    
    # Energy bar (range -1 to 0)
    energy_norm = (diag['total_energy'] + 1) / 1  # -1 to 0 -> 0 to 1
    energy_bar = int(np.clip(energy_norm * bars, 0, bars))
    
    # ANEC bar (range 0 to 1)
    anec_bar = int(np.clip(diag['ANEC'] * bars, 0, bars))
    
    stability = "✓ STABLE" if diag['is_stable'] else "✗ UNSTABLE"
    horizon = "✓ NONE" if not diag['horizon_detected'] else "⚠ DETECTED"
    pfb = "● ACTIVE" if diag['pfb_triggered'] else "○ WAITING"
    
    lines = [
        "╔" + "═" * 60 + "╗",
        f"║ STANKOVA SLIPSTREAM - REAL-TIME STABILIZATION ║",
        "╠" + "═" * 60 + "╣",
        f"║ VELOCITY: {params.v:.2f} c            THROAT: {params.R0:.2f} m ║",
        f"║ Ad: {params.Ad:.4f}  Aw: {params.Aw:.4f}                  ║",
        "╠" + "═" * 60 + "╣",
        f"║ TOTAL ENERGY: {diag['total_energy']:.4f}  [{'█'*energy_bar}{' '*(bars-energy_bar)}] ║",
        f"║ ANEC:         {diag['ANEC']:.4f}  [{'█'*anec_bar}{' '*(bars-anec_bar)}] ║",
        f"║ MIN NEC:      {diag['min_NEC']:.4f}                        ║",
        "╠" + "═" * 60 + "╣",
        f"║ STABILITY: {stability}    HORIZON: {horizon}          ║",
        f"║ PFB: {pfb}    FILL: {diag['buffer_fill']:.1f}%         ║",
        f"║ FPS: {diag['fps']:.1f}    STEP: {diag['step_time_ms']:.2f} ms        ║",
        "╚" + "═" * 60 + "╝",
    ]
    
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("STANKOVA SLIPSTREAM - OPTIMIZED WARP STABILIZATION")
    print("LEAP-MORT Maxwell Bypass Engine")
    print("="*70)
    print("\nPARAMETERS:")
    print(f"  Velocity: 2.5 c")
    print(f"  Throat radius: 1.2 m")
    print(f"  Deformation: 0.08")
    print(f"  Grid size: 48³")
    print("\n" + "-"*70)
    print("Press Ctrl+C to exit")
    print("-"*70 + "\n")
    
    params = WarpParams()
    bypass = LEAPMORTBypass(params)
    
    t = 0.0
    frame_count = 0
    last_print = time.time()
    print_interval = 0.15
    
    try:
        while True:
            # Run step
            diag = bypass.step(t)
            
            # Display
            if time.time() - last_print > print_interval:
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(draw_hud(diag, bypass.p))
                print()
                
                # Field slice
                print("FIELD SLICE (2D cross-section at y=0):")
                print(draw_field(bypass.Phi, size=35))
                print()
                
                # LEAP-MORT status
                mort_status = f"φ={bypass.mort_angle:.2f} rad"
                rpe_status = f"ψ={bypass.rpe_phase:.2f} rad"
                pr_status = f"mem={np.mean(np.abs(bypass.photorefractive_memory)):.3f}"
                print(f"LEAP-MORT: MORT {mort_status} | RPE {rpe_status} | PR {pr_status}")
                
                last_print = time.time()
            
            t += params.dt
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation stopped.")
    
    # Summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"Total steps: {frame_count}")
    print(f"Simulated time: {t:.1f}s")
    print(f"Average FPS: {bypass.fps:.1f}")
    print(f"Final energy: {diag['total_energy']:.4f} (target: {bypass.energy_target:.4f})")
    print(f"Final ANEC: {diag['ANEC']:.4f} (target: >0)")
    print(f"Stability: {'✅ STABLE' if diag['is_stable'] else '❌ UNSTABLE'}")
    print(f"Horizon: {'✅ NONE' if not diag['horizon_detected'] else '❌ DETECTED'}")
    print("="*70)

if __name__ == "__main__":
    main()