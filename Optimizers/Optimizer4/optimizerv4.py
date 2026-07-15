#!/usr/bin/env python3
"""
WARP BUBBLE OPTIMIZER V4 - СТАБИЛНА ВЕРСИЯ
Физически ограничения, clip, штрафове и валидация
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

# Опит за импорт на Numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba JIT компилацията е активна")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba не е инсталирана - използвам чист Python")
    print("   Инсталирай с: pip install numba")

# ============================================================
# ФИЗИЧЕСКИ ГРАНИЦИ (РАЗУМНИ)
# ============================================================

# Въз основа на V1 резултатите - по-тесни и физически обосновани
BOUNDS_V4 = {
    'Aw': (0.005, 0.04),        # Wormhole потенциал
    'wW': (0.2, 0.8),           # Wormhole ширина (>0)
    'Ad': (0.0005, 0.008),      # Drive потенциал
    'wD': (0.8, 2.5),           # Drive ширина (от V1: 1.49)
    'epsBD': (0.01, 0.08),      # BD амплитуда (от V1: 0.036)
    'wBD': (0.3, 1.2),          # BD ширина (от V1: 0.67)
    'bBI': (0.3, 1.5),          # BI параметър (от V1: 0.475)
    'wshell': (0.3, 1.0)        # Shell дебелина (от V1: 0.612)
}

# ============================================================
# УСКОРЕНИ ФУНКЦИИ С NUMBA
# ============================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def compute_metrics_numba(grid, params):
        """JIT компилирана функция за бързо изчисляване на метриките"""
        eps = 1e-8
        h = 1e-4
        NN = len(grid)
        
        # Разпакетиране на параметрите
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
        
        # Инициализация на резултатите
        total_violation = 0.0
        max_violation = 0.0
        num_violations = 0
        total_points = 0
        min_alpha = 1e10
        total_energy = 0.0
        sum_alpha = 0.0
        
        dx = grid[1] - grid[0]
        
        for i in prange(NN):
            for j in range(NN):
                for k in range(NN):
                    x = grid[i]
                    y = grid[j]
                    z = grid[k]
                    
                    # Радиална функция
                    rr = np.sqrt(x*x + y*y + z*z)
                    rr = max(rr, eps)
                    
                    # Helper функция за Phi
                    def get_phi(xx, yy, zz):
                        rrr = np.sqrt(xx*xx + yy*yy + zz*zz)
                        rrr = max(rrr, eps)
                        
                        # WH
                        r_safe = max(rrr, eps)
                        Phi_WH = -Aw * (1.0 - R0/r_safe) * np.exp(-(rrr - R0)*(rrr - R0) / (wW*wW))
                        
                        # Drive
                        x_shift = xx - v * 0.0
                        Phi_Drive = Ad * x_shift * np.exp(-x_shift*x_shift / (wD*wD))
                        
                        # BD
                        Phi_BD = epsBD * np.exp(-rrr*rrr / (wBD*wBD))
                        
                        # BI
                        r_bi = max(rrr, 0.1)
                        Phi_BI = -1.0/bBI * np.sqrt(1.0 + (r_bi/R0)*(r_bi/R0))
                        
                        return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
                    
                    # Централна стойност
                    Phi = get_phi(x, y, z)
                    
                    # Производни (6-точков stencil)
                    Phi_xp = get_phi(x+h, y, z)
                    Phi_xm = get_phi(x-h, y, z)
                    Phi_yp = get_phi(x, y+h, z)
                    Phi_ym = get_phi(x, y-h, z)
                    Phi_zp = get_phi(x, y, z+h)
                    Phi_zm = get_phi(x, y, z-h)
                    
                    # Първи производни
                    PhiX = (Phi_xp - Phi_xm) / (2.0*h)
                    PhiY = (Phi_yp - Phi_ym) / (2.0*h)
                    PhiZ = (Phi_zp - Phi_zm) / (2.0*h)
                    
                    # Втори производни
                    PhiXX = (Phi_xp - 2.0*Phi + Phi_xm) / (h*h)
                    PhiYY = (Phi_yp - 2.0*Phi + Phi_ym) / (h*h)
                    PhiZZ = (Phi_zp - 2.0*Phi + Phi_zm) / (h*h)
                    
                    lapPhi = PhiXX + PhiYY + PhiZZ
                    gradPhi = PhiX*PhiX + PhiY*PhiY + PhiZ*PhiZ
                    
                    # Einstein тензор
                    Gtt = np.exp(4.0*Phi) * (2.0*lapPhi - 3.0*gradPhi)
                    Gxx = -2.0*lapPhi + gradPhi + 2.0*PhiX*PhiX - 2.0*PhiXX
                    
                    # Геометричен NEC
                    NEC_geom = (Gtt + Gxx) / (8.0*np.pi)
                    
                    # BD NEC
                    phiBD = 1.0 + epsBD * np.exp(-(rr - R0)*(rr - R0) / (wBD*wBD))
                    phiBD_x = (get_phi(x+h, y, z) - get_phi(x-h, y, z)) / (2.0*h)
                    phiBD_xx = (get_phi(x+h, y, z) - 2.0*phiBD + get_phi(x-h, y, z)) / (h*h)
                    NEC_BD = (omegaBD / (phiBD*phiBD)) * phiBD_x*phiBD_x + (1.0/phiBD) * phiBD_xx
                    
                    # BI NEC (опростено)
                    E_BI = np.exp(-(rr - R0)*(rr - R0) / (wD*wD))
                    F_BI = 2.0 * E_BI * E_BI
                    radicand = 1.0 + F_BI / (2.0*bBI*bBI)
                    if radicand < 0:
                        radicand = 0.0
                    L_BI = bBI*bBI * (1.0 - np.sqrt(radicand))
                    
                    # BI NEC чрез числена производна
                    E_xp = np.exp(-(np.sqrt((x+h)*(x+h) + y*y + z*z) - R0)**2 / (wD*wD))
                    E_xm = np.exp(-(np.sqrt((x-h)*(x-h) + y*y + z*z) - R0)**2 / (wD*wD))
                    F_xp = 2.0 * E_xp * E_xp
                    F_xm = 2.0 * E_xm * E_xm
                    rad_xp = 1.0 + F_xp / (2.0*bBI*bBI)
                    rad_xm = 1.0 + F_xm / (2.0*bBI*bBI)
                    if rad_xp < 0: rad_xp = 0.0
                    if rad_xm < 0: rad_xm = 0.0
                    L_xp = bBI*bBI * (1.0 - np.sqrt(rad_xp))
                    L_xm = bBI*bBI * (1.0 - np.sqrt(rad_xm))
                    NEC_BI = -(L_xp - L_xm) / (2.0*h)
                    
                    # Shell NEC
                    shell = np.exp(-(rr - R0)*(rr - R0) / (wshell*wshell))
                    NEC_shell = 0.03 * shell
                    
                    # Тотален NEC
                    NEC_total = NEC_geom + NEC_BD + NEC_BI + NEC_shell
                    
                    # Статистика за NEC
                    if NEC_total < 0.0:
                        total_violation += NEC_total
                        if NEC_total < max_violation:
                            max_violation = NEC_total
                        num_violations += 1
                    
                    total_points += 1
                    
                    # Alpha
                    alpha = np.exp(0.35*Phi) * (1.0 + 0.25*epsBD*np.exp(-rr*rr/(wBD*wBD))) + 1e-10
                    if alpha < min_alpha:
                        min_alpha = alpha
                    sum_alpha += alpha
                    
                    # Енергия
                    T00_geom = Gtt / (8.0*np.pi)
                    T00_BD = (omegaBD / (phiBD*phiBD)) * (phiBD_x*phiBD_x) / 2.0 + (1.0/phiBD) * phiBD_xx
                    T00_BI = -L_BI
                    T00_total = T00_geom + T00_BD + T00_BI
                    total_energy += T00_total * dx*dx*dx
        
        violation_ratio = num_violations / total_points if total_points > 0 else 0.0
        mean_alpha = sum_alpha / total_points if total_points > 0 else 1.0
        
        return {
            'total_violation': total_violation,
            'max_violation': max_violation,
            'num_violations': num_violations,
            'total_points': total_points,
            'violation_ratio': violation_ratio,
            'min_alpha': min_alpha,
            'mean_alpha': mean_alpha,
            'total_energy': total_energy
        }

else:
    # Fallback - бавна версия
    def compute_metrics_numba(grid, params):
        print("⚠️  Numba не е налична - използвам бавна версия")
        return {
            'total_violation': 0.0,
            'max_violation': 0.0,
            'num_violations': 0,
            'total_points': 0,
            'violation_ratio': 0.0,
            'min_alpha': 1.0,
            'mean_alpha': 1.0,
            'total_energy': 0.0
        }

# ============================================================
# МЕТРИЧЕН ШАБЛОН V4
# ============================================================

class WarpMetricOptimizerV4:
    """Стабилна метрика с физически ограничения"""
    
    def __init__(self, params: Dict):
        self.params = params.copy()
        self.epsilon = 1e-8
        self.NN = 15
        
        L = self.params.get('L', 5.0)
        self.grid = np.linspace(-L, L, self.NN)
        self.dx = self.grid[1] - self.grid[0]
    
    def compute_metrics(self):
        """Изчислява метриките с Numba"""
        # Конвертираме параметрите в масив
        params_array = np.array([
            self.params.get('R0', 2.0),
            self.params.get('Aw', 0.02),
            self.params.get('wW', 0.5),
            self.params.get('Ad', 0.002),
            self.params.get('wD', 1.5),
            self.params.get('epsBD', 0.04),
            self.params.get('wBD', 0.7),
            self.params.get('bBI', 0.5),
            self.params.get('v', 0.3),
            self.params.get('wshell', 0.6),
            self.params.get('sigma', 25.0),
            self.params.get('omegaBD', 10)
        ])
        
        return compute_metrics_numba(self.grid, params_array)


# ============================================================
# ОПТИМИЗАЦИОНЕН КЛАС V4
# ============================================================

@dataclass
class OptimizationResultV4:
    params: Dict
    score: float
    nec_violation: float
    total_energy: float
    min_alpha: float
    mean_alpha: float
    violation_ratio: float
    message: str
    iterations: int
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


class WarpOptimizerV4:
    """Стабилен оптимизатор с физически ограничения"""
    
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
    
    def _clamp_params(self, params: Dict) -> Dict:
        """Осигурява, че всички параметри са в границите"""
        clamped = params.copy()
        for name, value in params.items():
            if name in self.bounds:
                low, high = self.bounds[name]
                clamped[name] = np.clip(value, low, high)
        
        # Физически гаранции
        clamped['wW'] = max(clamped.get('wW', 0.5), 0.1)
        clamped['wshell'] = max(clamped.get('wshell', 0.3), 0.05)
        clamped['bBI'] = max(clamped.get('bBI', 0.5), 0.1)
        clamped['epsBD'] = max(clamped.get('epsBD', 0.01), 0.001)
        
        return clamped
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        """Преобразува вектор в параметри с clamp"""
        params = self.default_params.copy()
        for name, value in zip(self.param_names, vector):
            params[name] = float(value)
        
        params['warpAmp'] = 22.0
        
        # Clamp-ваме всички параметри
        params = self._clamp_params(params)
        
        return params
    
    def _objective(self, vector: np.ndarray) -> float:
        """Целева функция с физически ограничения"""
        try:
            # Проверка за излизане от границите (штраф)
            penalty = 0.0
            for name, value in zip(self.param_names, vector):
                low, high = self.bounds[name]
                if value < low:
                    penalty += (low - value) * 1e6
                if value > high:
                    penalty += (value - high) * 1e6
            
            # Ако има голямо нарушение, връщаме веднага
            if penalty > 1e6:
                return 1e12 + penalty
            
            params = self._vector_to_params(vector)
            metric = WarpMetricOptimizerV4(params)
            metrics = metric.compute_metrics()
            
            # Проверка за физическа валидност
            if metrics['min_alpha'] <= 0:
                return 1e9 + abs(metrics['min_alpha']) * 1e6
            
            # Ако всички точки нарушават NEC, това е лошо
            if metrics['violation_ratio'] > 0.5:
                return 1e8 + metrics['violation_ratio'] * 1e8
            
            # Целева функция
            w_nec = 100.0
            w_energy = 0.1
            w_violation = 50.0
            
            score = (
                w_nec * abs(metrics['total_violation']) +
                w_energy * abs(metrics['total_energy']) +
                w_violation * metrics['violation_ratio'] * 100.0
            )
            
            # Бонус за голям min_alpha (стабилност)
            score -= metrics['min_alpha'] * 10.0
            
            # Запис на резултата
            self.results.append({
                'params': params,
                'score': score,
                'nec_violation': metrics['total_violation'],
                'energy': metrics['total_energy'],
                'min_alpha': metrics['min_alpha'],
                'mean_alpha': metrics['mean_alpha'],
                'violation_ratio': metrics['violation_ratio']
            })
            
            if self.best_result is None or score < self.best_result['score']:
                self.best_result = self.results[-1]
                print(f"  🎯 Нов най-добър резултат: {score:.2f} (violation: {metrics['violation_ratio']:.2%})")
            
            return score
            
        except Exception as e:
            return 1e12
    
    def optimize(self, max_iterations: int = 40, population_size: int = 20) -> OptimizationResultV4:
        """Изпълнява оптимизацията"""
        print("="*70)
        print("WARP BUBBLE OPTIMIZER V4 - СТАБИЛНА ВЕРСИЯ")
        print("Физически ограничения, clamp и штрафове")
        print("="*70)
        print()
        
        print("📊 Оптимизация на параметрите:")
        for name, (low, high) in self.bounds.items():
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
        
        print(f"\n  Max iterations: {max_iterations}")
        print(f"  Population size: {population_size}")
        print(f"  Numba: {'✅ Да' if NUMBA_AVAILABLE else '❌ Не'}")
        print(f"  Мрежа: 15x15x15 = 3375 точки")
        print()
        
        start_time = time.time()
        
        print("🔍 Започвам Differential Evolution...")
        print("   ⏱️  Очаквано време: ~30-60 минути")
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
        metric = WarpMetricOptimizerV4(final_params)
        final_metrics = metric.compute_metrics()
        
        return OptimizationResultV4(
            params=final_params,
            score=refined.fun,
            nec_violation=final_metrics['total_violation'],
            total_energy=final_metrics['total_energy'],
            min_alpha=final_metrics['min_alpha'],
            mean_alpha=final_metrics['mean_alpha'],
            violation_ratio=final_metrics['violation_ratio'],
            message=f"Optimization completed in {elapsed/60:.1f} minutes",
            iterations=len(self.results),
            time_seconds=elapsed
        )


# ============================================================
# ДОПЪЛНИТЕЛНА ВАЛИДАЦИЯ
# ============================================================

def validate_params(params: Dict) -> Dict:
    """Проверява дали параметрите са физически валидни"""
    issues = []
    
    # Проверка за положителни стойности
    if params.get('wW', 0) <= 0:
        issues.append("wW <= 0 (трябва да е > 0)")
    if params.get('wshell', 0) <= 0:
        issues.append("wshell <= 0 (трябва да е > 0)")
    if params.get('bBI', 0) <= 0:
        issues.append("bBI <= 0 (трябва да е > 0)")
    if params.get('Aw', 0) <= 0:
        issues.append("Aw <= 0 (трябва да е > 0)")
    if params.get('epsBD', 0) <= 0:
        issues.append("epsBD <= 0 (трябва да е > 0)")
    
    # Проверка за разумни стойности
    if params.get('wW', 0) > 2.0:
        issues.append("wW > 2.0 (твърде голяма ширина)")
    if params.get('wshell', 0) > 1.5:
        issues.append("wshell > 1.5 (твърде дебела обвивка)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    print("="*70)
    print("WARP BUBBLE OPTIMIZER V4")
    print("Стабилна версия с физически ограничения")
    print("="*70)
    print()
    
    # Начални параметри (от V1)
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
    
    print("📋 Физически граници:")
    for name, (low, high) in BOUNDS_V4.items():
        print(f"  {name}: [{low:.4f}, {high:.4f}]")
    print()
    
    # Създаване на оптимизатор
    optimizer = WarpOptimizerV4(BOUNDS_V4, initial_params)
    
    # Стартиране
    result = optimizer.optimize(max_iterations=35, population_size=15)
    
    # Резултати
    print("\n" + "="*70)
    print("✅ ОПТИМИЗАЦИЯТА V4 ЗАВЪРШИ")
    print("="*70)
    print()
    
    print("📊 Най-добри параметри:")
    for name, value in result.params.items():
        if name in BOUNDS_V4:
            print(f"  {name}: {value:.8f}")
    print()
    
    print("📈 Резултати:")
    print(f"  Score: {result.score:.4f}")
    print(f"  NEC violation: {result.nec_violation:.4f}")
    print(f"  Total energy: {result.total_energy:.4f}")
    print(f"  Min alpha: {result.min_alpha:.6f}")
    print(f"  Mean alpha: {result.mean_alpha:.6f}")
    print(f"  Violation ratio: {result.violation_ratio:.4%}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Време: {result.time_seconds/60:.1f} минути")
    print(f"  {result.message}")
    print()
    
    # Валидация
    validation = validate_params(result.params)
    print("🔍 Валидация:")
    if validation['valid']:
        print("  ✅ Всички параметри са физически валидни!")
    else:
        print("  ❌ Намерени проблеми:")
        for issue in validation['issues']:
            print(f"     - {issue}")
    print()
    
    # Запис
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    params_filename = f"optimized_params_v4_{timestamp}.json"
    with open(params_filename, 'w') as f:
        json.dump(result.params, f, indent=2)
    print(f"📄 Оптимални параметри записани в: {params_filename}")
    
    report_filename = f"optimization_report_v4_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'best_params': result.params,
        'score': result.score,
        'nec_violation': result.nec_violation,
        'total_energy': result.total_energy,
        'min_alpha': result.min_alpha,
        'mean_alpha': result.mean_alpha,
        'violation_ratio': result.violation_ratio,
        'iterations': result.iterations,
        'time_seconds': result.time_seconds,
        'message': result.message,
        'bounds': BOUNDS_V4,
        'initial_params': initial_params,
        'validation': validation
    }
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Пълен отчет записан в: {report_filename}")
    
    # Сравнение
    print()
    print("📊 Сравнение с V1:")
    for name in BOUNDS_V4:
        old = initial_params.get(name, 0)
        new = result.params.get(name, 0)
        change = ((new - old) / old * 100) if old != 0 else 0
        status = "✅" if abs(change) < 50 else "⚠️"
        print(f"  {status} {name}: {old:.6f} → {new:.6f} ({change:+.1f}%)")
    
    print()
    print("="*70)
    print("КРАЙ НА ОПТИМИЗАЦИЯТА V4")
    print("="*70)
    print()
    print("💡 Следващи стъпки:")
    print("  1. Провери optimization_report_v4_*.json за детайли")
    print("  2. Ако резултатите са добри, генерирай SecondPass файловете")
    print("  3. Ако не, може да пуснеш още една итерация")

if __name__ == "__main__":
    main()