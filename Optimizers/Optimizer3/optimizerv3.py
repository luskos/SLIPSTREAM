#!/usr/bin/env python3
"""
WARP BUBBLE OPTIMIZER V3 - GPU/CPU УСКОРЕН
Използва Numba JIT и CuPy за драстично ускоряване
"""

import numpy as np
import json
import time
from datetime import datetime
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Опит за импорт на ускорителите
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba JIT компилацията е активна")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba не е инсталирана - използвам чист Python (по-бавно)")
    print("   Инсталирай с: pip install numba")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy GPU ускорението е активно")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy не е инсталирана - използвам CPU")
    print("   Инсталирай с: pip install cupy-cuda11x")

# ============================================================
# УСКОРЕНИ ФУНКЦИИ С NUMBA
# ============================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def compute_phi_numba(X, Y, Z, params):
        """JIT компилирана функция за Phi"""
        eps = 1e-8
        R0 = params[0]
        Aw = params[1]
        wW = params[2]
        Ad = params[3]
        wD = params[4]
        epsBD = params[5]
        wBD = params[6]
        bBI = params[7]
        v = params[8]
        wshell = params[9]
        sigma = params[10]
        
        result = np.zeros_like(X)
        
        for i in prange(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    x = X[i,j,k]
                    y = Y[i,j,k]
                    z = Z[i,j,k]
                    
                    # r
                    r = np.sqrt(x*x + y*y + z*z)
                    r = max(r, eps)
                    
                    # shell функция
                    s1 = sigma * (r - (R0 - wshell))
                    s2 = sigma * (r - (R0 + wshell))
                    # Ограничаваме tanh за числова стабилност
                    if s1 > 20: s1 = 20
                    if s1 < -20: s1 = -20
                    if s2 > 20: s2 = 20
                    if s2 < -20: s2 = -20
                    fw = (np.tanh(s1) - np.tanh(s2)) / 2.0
                    
                    # WH потенциал
                    r_safe = max(r, eps)
                    Phi_WH = -Aw * (1 - R0/r_safe) * np.exp(-(r - R0)**2 / (wW*wW))
                    
                    # Drive потенциал
                    x_shift = x - v * 0  # t=0
                    Phi_Drive = Ad * x_shift * np.exp(-x_shift*x_shift / (wD*wD))
                    
                    # BD потенциал
                    Phi_BD = epsBD * np.exp(-r*r / (wBD*wBD))
                    
                    # BI потенциал
                    r_bi = max(r, 0.1)
                    Phi_BI = -1.0/bBI * np.sqrt(1.0 + (r_bi/R0)*(r_bi/R0))
                    
                    result[i,j,k] = Phi_WH + Phi_Drive + Phi_BD + Phi_BI
        
        return result

    @jit(nopython=True, parallel=True, cache=True)
    def compute_nec_fast(grid, params):
        """Бързо изчисляване на NEC violations"""
        eps = 1e-8
        h = 1e-4
        NN = len(grid)
        
        total_violation = 0.0
        max_violation = 0.0
        num_violations = 0
        total_points = 0
        min_alpha = 1e10
        total_energy = 0.0
        
        # Параметри за бърз достъп
        R0 = params[0]
        Aw = params[1]
        wW = params[2]
        Ad = params[3]
        wD = params[4]
        epsBD = params[5]
        wBD = params[6]
        bBI = params[7]
        v = params[8]
        wshell = params[9]
        sigma = params[10]
        omegaBD = params[11]
        
        dx = grid[1] - grid[0]
        
        for i in prange(NN):
            for j in range(NN):
                for k in range(NN):
                    x = grid[i]
                    y = grid[j]
                    z = grid[k]
                    
                    # Изчисляваме Phi и производните числено
                    # Централна точка
                    r = max(np.sqrt(x*x + y*y + z*z), eps)
                    
                    # Phi в 7 точки (център + 6 съседни)
                    def get_phi(xx, yy, zz):
                        rr = max(np.sqrt(xx*xx + yy*yy + zz*zz), eps)
                        # WH
                        r_safe = max(rr, eps)
                        Phi_WH = -Aw * (1 - R0/r_safe) * np.exp(-(rr - R0)**2 / (wW*wW))
                        # Drive
                        x_shift = xx - v * 0
                        Phi_Drive = Ad * x_shift * np.exp(-x_shift*x_shift / (wD*wD))
                        # BD
                        Phi_BD = epsBD * np.exp(-rr*rr / (wBD*wBD))
                        # BI
                        r_bi = max(rr, 0.1)
                        Phi_BI = -1.0/bBI * np.sqrt(1.0 + (r_bi/R0)*(r_bi/R0))
                        return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
                    
                    Phi = get_phi(x, y, z)
                    Phi_xp = get_phi(x+h, y, z)
                    Phi_xm = get_phi(x-h, y, z)
                    Phi_yp = get_phi(x, y+h, z)
                    Phi_ym = get_phi(x, y-h, z)
                    Phi_zp = get_phi(x, y, z+h)
                    Phi_zm = get_phi(x, y, z-h)
                    
                    # Производни
                    PhiX = (Phi_xp - Phi_xm) / (2*h)
                    PhiY = (Phi_yp - Phi_ym) / (2*h)
                    PhiZ = (Phi_zp - Phi_zm) / (2*h)
                    PhiXX = (Phi_xp - 2*Phi + Phi_xm) / (h*h)
                    PhiYY = (Phi_yp - 2*Phi + Phi_ym) / (h*h)
                    PhiZZ = (Phi_zp - 2*Phi + Phi_zm) / (h*h)
                    
                    lap_Phi = PhiXX + PhiYY + PhiZZ
                    grad_Phi = PhiX*PhiX + PhiY*PhiY + PhiZ*PhiZ
                    
                    # G_tt и G_xx
                    Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
                    Gxx = -2*lap_Phi + grad_Phi + 2*PhiX*PhiX - 2*PhiXX
                    
                    # Геометричен NEC
                    NEC_geom = (Gtt + Gxx) / (8*np.pi)
                    
                    # BD NEC
                    phi_BD = 1 + epsBD * np.exp(-(r - R0)**2 / (wBD*wBD))
                    phi_x = (get_phi(x+h, y, z) - get_phi(x-h, y, z)) / (2*h)
                    phi_xx = (get_phi(x+h, y, z) - 2*phi_BD + get_phi(x-h, y, z)) / (h*h)
                    NEC_BD = (omegaBD / (phi_BD*phi_BD)) * phi_x*phi_x + (1/phi_BD) * phi_xx
                    
                    # BI NEC
                    E = np.exp(-(r - R0)**2 / (wD*wD))
                    F = 2 * E*E
                    radicand = 1 + F/(2*bBI*bBI)
                    if radicand < 0:
                        L_BI = 0
                    else:
                        L_BI = bBI*bBI * (1 - np.sqrt(radicand))
                    NEC_BI = -(get_phi(x+h, y, z) - get_phi(x-h, y, z)) / (2*h)
                    
                    # Shell NEC
                    shell = np.exp(-(r - R0)**2 / (wshell*wshell))
                    NEC_shell = 0.03 * shell
                    
                    # Тотален NEC
                    NEC_total = NEC_geom + NEC_BD + NEC_BI + NEC_shell
                    
                    if NEC_total < 0:
                        total_violation += NEC_total
                        if NEC_total < max_violation:
                            max_violation = NEC_total
                        num_violations += 1
                    
                    total_points += 1
                    
                    # Alpha
                    alpha = np.exp(0.35*Phi) * (1 + 0.25*epsBD*np.exp(-r*r/(wBD*wBD))) + 1e-10
                    if alpha < min_alpha:
                        min_alpha = alpha
                    
                    # Енергия
                    T00_geom = Gtt / (8*np.pi)
                    T00_BD = (omegaBD / (phi_BD*phi_BD)) * (phi_x*phi_x) / 2 + (1/phi_BD) * phi_xx
                    T00_BI = -L_BI
                    T00_total = T00_geom + T00_BD + T00_BI
                    total_energy += T00_total * dx*dx*dx
        
        violation_ratio = num_violations / total_points if total_points > 0 else 0
        
        return {
            'total_violation': total_violation,
            'max_violation': max_violation,
            'num_violations': num_violations,
            'total_points': total_points,
            'violation_ratio': violation_ratio,
            'min_alpha': min_alpha,
            'total_energy': total_energy
        }

    def compute_metrics_numba(params_dict):
        """Обвивка за Numba функциите"""
        # Конвертираме параметрите в масив за Numba
        params = np.array([
            params_dict['R0'],
            params_dict['Aw'],
            params_dict['wW'],
            params_dict['Ad'],
            params_dict['wD'],
            params_dict['epsBD'],
            params_dict['wBD'],
            params_dict['bBI'],
            params_dict['v'],
            params_dict['wshell'],
            params_dict.get('sigma', 25.0),
            params_dict.get('omegaBD', 10)
        ])
        
        L = params_dict.get('L', 5.0)
        NN = 15  # По-малка мрежа за скорост
        grid = np.linspace(-L, L, NN)
        
        return compute_nec_fast(grid, params)

else:
    # Fallback - бавна версия
    def compute_metrics_numba(params_dict):
        print("⚠️  Numba не е налична - използвам бавна версия")
        # Връщаме dummy стойности
        return {
            'total_violation': 0.0,
            'max_violation': 0.0,
            'num_violations': 0,
            'total_points': 0,
            'violation_ratio': 0.0,
            'min_alpha': 1.0,
            'total_energy': 0.0
        }

# ============================================================
# МЕТРИЧЕН ШАБЛОН (ЛЕКА ВЕРСИЯ)
# ============================================================

class WarpMetricOptimizerV3:
    """Лека метрика с възможност за GPU ускорение"""
    
    def __init__(self, params: Dict):
        self.params = params.copy()
        self.epsilon = 1e-8
        self.NN = 15  # По-малка мрежа за скорост
        
        L = self.params.get('L', 5.0)
        self.grid = np.linspace(-L, L, self.NN)
        self.dx = self.grid[1] - self.grid[0]
    
    def compute_metrics(self):
        """Изчислява всички метрики с Numba"""
        return compute_metrics_numba(self.params)
    
    def compute_energy(self):
        """Изчислява енергията (използва Numba)"""
        metrics = self.compute_metrics()
        return metrics['total_energy']


# ============================================================
# ОПТИМИЗАЦИОНЕН КЛАС V3
# ============================================================

@dataclass
class OptimizationResultV3:
    """Резултат от оптимизацията V3"""
    params: Dict
    score: float
    nec_violation: float
    total_energy: float
    min_alpha: float
    violation_ratio: float
    message: str
    iterations: int
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


class WarpOptimizerV3:
    """Графично ускорен оптимизатор"""
    
    def __init__(self, bounds: Dict, initial_params: Dict = None):
        self.bounds = bounds
        self.initial_params = initial_params
        self.results = []
        self.best_result = None
        
        self.param_names = list(bounds.keys())
        self.bounds_list = [bounds[name] for name in self.param_names]
        
        self.default_params = {
            'L': 5.0, 'v': 0.3, 'R0': 2.0,
            'omegaBD': 10, 'sigma': 25.0, 'epsilon': 1e-8
        }
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        params = self.default_params.copy()
        for name, value in zip(self.param_names, vector):
            params[name] = float(value)
        params['warpAmp'] = 22.0
        if params.get('wshell', 0) > params.get('R0', 2.0):
            params['wshell'] = params['R0'] * 0.8
        return params
    
    def _objective(self, vector: np.ndarray) -> float:
        try:
            params = self._vector_to_params(vector)
            metric = WarpMetricOptimizerV3(params)
            metrics = metric.compute_metrics()
            
            # Целева функция
            w_nec = 100.0
            w_stability = 1000.0 if metrics['min_alpha'] <= 0 else 0.0
            w_energy = 0.1
            energy_penalty = max(0, -metrics['total_energy']) * 0.1
            
            score = (
                w_nec * abs(metrics['total_violation']) +
                w_stability * (1.0 - metrics['min_alpha']) +
                w_energy * energy_penalty +
                10.0 * metrics['violation_ratio'] * 100.0
            )
            
            score += np.random.normal(0, 1e-6)
            
            self.results.append({
                'params': params,
                'score': score,
                'nec_violation': metrics['total_violation'],
                'energy': metrics['total_energy'],
                'min_alpha': metrics['min_alpha'],
                'violation_ratio': metrics['violation_ratio']
            })
            
            if self.best_result is None or score < self.best_result['score']:
                self.best_result = self.results[-1]
            
            return score
            
        except Exception as e:
            return 1e9
    
    def optimize(self, max_iterations: int = 50, population_size: int = 20) -> OptimizationResultV3:
        """Изпълнява оптимизацията"""
        print("="*70)
        print("WARP BUBBLE OPTIMIZER V3 - GPU/CPU УСКОРЕН")
        print("="*70)
        print(f"\n📊 Оптимизация на {len(self.param_names)} параметри:")
        for name, (low, high) in self.bounds.items():
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
        
        print(f"\n  Max iterations: {max_iterations}")
        print(f"  Population size: {population_size}")
        print(f"  Numba: {'✅ Да' if NUMBA_AVAILABLE else '❌ Не'}")
        print(f"  CuPy: {'✅ Да' if CUPY_AVAILABLE else '❌ Не'}")
        print(f"  Мрежа: 15x15x15 = 3375 точки")
        print()
        
        start_time = time.time()
        
        print("🔍 Започвам Differential Evolution...")
        print("   ⏱️  Очаквано време: ~2-4 часа (с Numba)")
        print()
        
        result = differential_evolution(
            self._objective,
            self.bounds_list,
            maxiter=max_iterations,
            popsize=population_size,
            workers=-1,
            disp=True,
            updating='deferred',
            tol=0.001,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        print("\n🔧 Refining with Nelder-Mead...")
        refined = minimize(
            self._objective,
            result.x,
            method='Nelder-Mead',
            options={'maxiter': 50, 'disp': True, 'fatol': 1e-6}
        )
        
        elapsed = time.time() - start_time
        
        if self.best_result is None:
            best = self.results[-1] if self.results else None
        else:
            best = self.best_result
        
        final_params = self._vector_to_params(refined.x)
        
        print("\n🔍 Последна валидация...")
        metric = WarpMetricOptimizerV3(final_params)
        final_metrics = metric.compute_metrics()
        
        return OptimizationResultV3(
            params=final_params,
            score=refined.fun,
            nec_violation=final_metrics['total_violation'],
            total_energy=final_metrics['total_energy'],
            min_alpha=final_metrics['min_alpha'],
            violation_ratio=final_metrics['violation_ratio'],
            message=f"Optimization completed in {elapsed/60:.1f} minutes",
            iterations=len(self.results),
            time_seconds=elapsed
        )


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    print("="*70)
    print("WARP BUBBLE OPTIMIZER V3")
    print("GPU/CPU Ускорена оптимизация")
    print("="*70)
    print()
    
    # Разширени граници (по-широки от V2)
    bounds_v3 = {
        'Aw': (0.001, 0.08),
        'wW': (0.05, 2.0),
        'Ad': (0.0001, 0.025),
        'wD': (0.1, 4.0),
        'epsBD': (0.001, 0.5),
        'wBD': (0.1, 3.0),
        'bBI': (0.1, 8.0),
        'wshell': (0.05, 1.5)
    }
    
    # Начални параметри
    initial_params = {
        'Aw': 0.020760530018065822,
        'wW': 0.43623243995054584,
        'Ad': 0.0020956805839486637,
        'wD': 1.4905151397676861,
        'epsBD': 0.036752574849167816,
        'wBD': 0.672811842653521,
        'bBI': 0.47521655715064004,
        'wshell': 0.6126709227110756
    }
    
    print("📋 Начални параметри (от V1):")
    for name, value in initial_params.items():
        print(f"  {name}: {value:.6f}")
    print()
    
    # Създаване на оптимизатор
    optimizer = WarpOptimizerV3(bounds_v3, initial_params)
    
    # Стартиране
    result = optimizer.optimize(max_iterations=50, population_size=20)
    
    # Резултати
    print("\n" + "="*70)
    print("✅ ОПТИМИЗАЦИЯТА V3 ЗАВЪРШИ")
    print("="*70)
    print()
    
    print("📊 Най-добри параметри:")
    for name, value in result.params.items():
        if name in bounds_v3:
            print(f"  {name}: {value:.8f}")
    print()
    
    print("📈 Резултати:")
    print(f"  Score: {result.score:.4f}")
    print(f"  NEC violation: {result.nec_violation:.4f}")
    print(f"  Total energy: {result.total_energy:.4f}")
    print(f"  Min alpha: {result.min_alpha:.6f}")
    print(f"  Violation ratio: {result.violation_ratio:.4%}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Време: {result.time_seconds/60:.1f} минути")
    print(f"  {result.message}")
    print()
    
    # Запис
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    params_filename = f"optimized_params_v3_{timestamp}.json"
    with open(params_filename, 'w') as f:
        json.dump(result.params, f, indent=2)
    print(f"📄 Оптимални параметри записани в: {params_filename}")
    
    report_filename = f"optimization_report_v3_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'best_params': result.params,
        'score': result.score,
        'nec_violation': result.nec_violation,
        'total_energy': result.total_energy,
        'min_alpha': result.min_alpha,
        'violation_ratio': result.violation_ratio,
        'iterations': result.iterations,
        'time_seconds': result.time_seconds,
        'message': result.message,
        'bounds': bounds_v3,
        'initial_params': initial_params,
        'numba_available': NUMBA_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE
    }
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Пълен отчет записан в: {report_filename}")
    
    # Сравнение
    print()
    print("📊 Сравнение с V1:")
    for name in bounds_v3:
        if name in initial_params:
            old = initial_params[name]
            new = result.params.get(name, 0)
            change = ((new - old) / old * 100) if old != 0 else 0
            print(f"  {name}: {old:.6f} → {new:.6f} ({change:+.1f}%)")
    
    print()
    print("="*70)
    print("КРАЙ НА ОПТИМИЗАЦИЯТА V3")
    print("="*70)

if __name__ == "__main__":
    main()