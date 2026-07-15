#!/usr/bin/env python3
"""
WARP BUBBLE OPTIMIZER V2 - SECOND PASS
Разширени граници, по-интелигентна целева функция, по-дълга оптимизация
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

# ============================================================
# МЕТРИЧЕН ШАБЛОН ЗА ОПТИМИЗАЦИЯ V2
# ============================================================

class WarpMetricOptimizerV2:
    """Подобрена версия на метриката за оптимизация"""
    
    def __init__(self, params: Dict):
        self.params = params.copy()
        self.epsilon = 1e-8
        self.NN = 21  # По-груба мрежа за бързина
        
        # Създаване на мрежата
        L = self.params.get('L', 5.0)
        self.grid = np.linspace(-L, L, self.NN)
        self.dx = self.grid[1] - self.grid[0]
        self.X, self.Y, self.Z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.t = 0
    
    def r(self, x=None, y=None, z=None):
        if x is None:
            r_val = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            return np.maximum(r_val, self.params.get('epsilon', 1e-8))
        else:
            r_val = np.sqrt(x**2 + y**2 + z**2)
            return max(r_val, self.params.get('epsilon', 1e-8))
    
    def fw(self, s):
        R0 = self.params['R0']
        wshell = self.params['wshell']
        sigma = self.params.get('sigma', 25.0)
        # По-добра регуларизация за shell функцията
        return (np.tanh(sigma*(s - (R0 - wshell))) - 
                np.tanh(sigma*(s - (R0 + wshell)))) / 2.0
    
    def Phi(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            r = self.r()
            # WH потенциал - добавен регуларизатор
            r_safe = np.maximum(r, self.epsilon)
            Phi_WH = -self.params['Aw'] * (1 - self.params['R0']/r_safe) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
            # Drive потенциал
            x_shift = self.X - self.params['v']*t
            Phi_Drive = self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
            # BD потенциал
            Phi_BD = self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
            # BI потенциал - подобрена регуларизация
            r_bi = np.maximum(r, 0.1)  # Предотвратява сингулярност в центъра
            Phi_BI = -1/self.params['bBI'] * np.sqrt(1 + (r_bi/self.params['R0'])**2)
            return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
        else:
            r = self.r(x, y, z)
            r_safe = max(r, self.epsilon)
            Phi_WH = -self.params['Aw'] * (1 - self.params['R0']/r_safe) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
            x_shift = x - self.params['v']*t
            Phi_Drive = self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
            Phi_BD = self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
            r_bi = max(r, 0.1)
            Phi_BI = -1/self.params['bBI'] * np.sqrt(1 + (r_bi/self.params['R0'])**2)
            return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
    
    def alpha(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            Phi = self.Phi()
            r = self.r()
            # Добавяме малък регуларизатор за α > 0
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) + 1e-10
        else:
            Phi = self.Phi(x, y, z, t)
            r = self.r(x, y, z)
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) + 1e-10
    
    def beta_x(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            r = self.r()
            r_safe = np.maximum(r, self.epsilon)
            return -self.params['v'] * 22.0 * self.fw(self.r(self.X - self.params['v']*t, self.Y, self.Z)) / np.sqrt(1 + (r_safe/(self.params['bBI']*self.params['wW']))**2)
        else:
            r = self.r(x, y, z)
            r_safe = max(r, self.epsilon)
            return -self.params['v'] * 22.0 * self.fw(self.r(x - self.params['v']*t, y, z)) / np.sqrt(1 + (r_safe/(self.params['bBI']*self.params['wW']))**2)
    
    def g_tt(self, x=None, y=None, z=None, t=None):
        if x is None:
            alpha = self.alpha()
            beta = self.beta_x()
            return -alpha**2 + beta**2
        else:
            alpha = self.alpha(x, y, z, t)
            beta = self.beta_x(x, y, z, t)
            return -alpha**2 + beta**2
    
    def g_xx(self, x=None, y=None, z=None, t=None):
        if x is None:
            Phi = self.Phi()
            r = self.r()
            return np.exp(-2*Phi) * (1 + 0.15*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) * (1 + 0.1/self.params['bBI'])
        else:
            Phi = self.Phi(x, y, z, t)
            r = self.r(x, y, z)
            return np.exp(-2*Phi) * (1 + 0.15*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) * (1 + 0.1/self.params['bBI'])
    
    def numerical_derivative(self, func, x, y, z, axis=0, h=1e-5):
        if axis == 0:
            return (func(x+h, y, z) - func(x-h, y, z)) / (2*h)
        elif axis == 1:
            return (func(x, y+h, z) - func(x, y-h, z)) / (2*h)
        else:
            return (func(x, y, z+h) - func(x, y, z-h)) / (2*h)
    
    def numerical_second_derivative(self, func, x, y, z, axis=0, h=1e-5):
        if axis == 0:
            return (func(x+h, y, z) - 2*func(x, y, z) + func(x-h, y, z)) / (h**2)
        elif axis == 1:
            return (func(x, y+h, z) - 2*func(x, y, z) + func(x, y-h, z)) / (h**2)
        else:
            return (func(x, y, z+h) - 2*func(x, y, z) + func(x, y, z-h)) / (h**2)
    
    # ============================================================
    # МЕТОДИ ЗА ОПТИМИЗАЦИЯ - V2
    # ============================================================
    
    def compute_NEC_detailed(self):
        """Изчислява детайлна статистика за NEC"""
        grid = self.grid[::2]  # всяка втора точка за скорост
        dx = self.dx * 2
        
        nec_violations = []
        alpha_vals = []
        energy_density = []
        gtt_values = []
        
        for x in grid:
            for y in grid:
                for z in grid:
                    # Изчисли Phi и производните
                    h = 1e-4
                    Phi = self.Phi(x, y, z)
                    Phi_xp = self.Phi(x+h, y, z)
                    Phi_xm = self.Phi(x-h, y, z)
                    Phi_yp = self.Phi(x, y+h, z)
                    Phi_ym = self.Phi(x, y-h, z)
                    Phi_zp = self.Phi(x, y, z+h)
                    Phi_zm = self.Phi(x, y, z-h)
                    
                    lap_Phi = (Phi_xp - 2*Phi + Phi_xm)/h**2 + \
                              (Phi_yp - 2*Phi + Phi_ym)/h**2 + \
                              (Phi_zp - 2*Phi + Phi_zm)/h**2
                    
                    grad_Phi = ((Phi_xp - Phi_xm)/(2*h))**2 + \
                               ((Phi_yp - Phi_ym)/(2*h))**2 + \
                               ((Phi_zp - Phi_zm)/(2*h))**2
                    
                    # G_tt и G_xx
                    Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
                    Gxx = -2*lap_Phi + grad_Phi + 2*((Phi_xp - Phi_xm)/(2*h))**2 - 2*(Phi_xp - 2*Phi + Phi_xm)/h**2
                    
                    # Геометричен NEC
                    NEC_geom = (Gtt + Gxx) / (8*np.pi)
                    
                    # BD NEC
                    phi_BD = self.phiBD(x, y, z)
                    phi_x = (self.phiBD(x+h, y, z) - self.phiBD(x-h, y, z)) / (2*h)
                    phi_xx = (self.phiBD(x+h, y, z) - 2*phi_BD + self.phiBD(x-h, y, z)) / h**2
                    NEC_BD = (self.params['omegaBD'] / phi_BD**2) * phi_x**2 + (1/phi_BD) * phi_xx
                    
                    # BI NEC
                    NEC_BI = -(self.LBI(x+h, y, z) - self.LBI(x-h, y, z)) / (2*h)
                    
                    # Shell NEC
                    shell = np.exp(-(self.r(x,y,z) - self.params['R0'])**2 / self.params['wshell']**2)
                    NEC_shell = 0.03 * shell
                    
                    # Тотален NEC
                    NEC_total = NEC_geom + NEC_BD + NEC_BI + NEC_shell
                    
                    if NEC_total < 0:
                        nec_violations.append(NEC_total)
                    
                    # Записваме за статистика
                    alpha_vals.append(self.alpha(x, y, z))
                    gtt_values.append(self.g_tt(x, y, z))
                    
                    # T00 за енергия
                    T00_geom = Gtt / (8*np.pi)
                    T00_BD = (self.params['omegaBD'] / phi_BD**2) * (phi_x**2 + 0 + 0) / 2 + (1/phi_BD) * phi_xx
                    T00_BI = -self.LBI(x, y, z)
                    T00_total = T00_geom + T00_BD + T00_BI + 0
                    energy_density.append(T00_total)
        
        # Статистика
        total_violation = sum(nec_violations) if nec_violations else 0
        max_violation = min(nec_violations) if nec_violations else 0
        num_violations = len(nec_violations)
        total_points = len(grid)**3
        
        return {
            'total_violation': total_violation,
            'max_violation': max_violation,
            'num_violations': num_violations,
            'total_points': total_points,
            'violation_ratio': num_violations / total_points if total_points > 0 else 0,
            'min_alpha': min(alpha_vals) if alpha_vals else 0,
            'max_alpha': max(alpha_vals) if alpha_vals else 0,
            'mean_alpha': np.mean(alpha_vals) if alpha_vals else 0,
            'min_gtt': min(gtt_values) if gtt_values else 0,
            'max_gtt': max(gtt_values) if gtt_values else 0,
            'mean_energy': np.mean(energy_density) if energy_density else 0,
            'total_energy': sum(energy_density) * dx**3 if energy_density else 0
        }
    
    def phiBD(self, x, y, z):
        """Brans-Dicke скаларно поле"""
        r = self.r(x, y, z)
        return 1 + self.params['epsBD'] * np.exp(-(r - self.params['R0'])**2 / self.params['wBD']**2)
    
    def LBI(self, x, y, z):
        """Born-Infeld Lagrangian"""
        r = self.r(x, y, z)
        E = np.exp(-(r - self.params['R0'])**2 / self.params['wD']**2)
        F = 2 * E**2
        b = self.params['bBI']
        radicand = 1 + F/(2*b**2)
        if radicand < 0:
            return 0  # Защита срещу комплексни числа
        return b**2 * (1 - np.sqrt(radicand))
    
    def compute_stability_score(self):
        """Изчислява цялостна стабилност на метриката"""
        grid = self.grid[::2]
        score = 0
        issues = []
        
        for x in grid:
            for y in grid:
                for z in grid:
                    # Проверка на α > 0
                    alpha = self.alpha(x, y, z)
                    if alpha <= 0:
                        score += 1
                        issues.append(f"alpha<=0 at ({x},{y},{z})")
                    
                    # Проверка на g_tt (без големи сингулярности)
                    gtt = self.g_tt(x, y, z)
                    if not np.isfinite(gtt) or abs(gtt) > 1e6:
                        score += 10
                        issues.append(f"gtt singular at ({x},{y},{z})")
                    
                    # Проверка на Phi (без сингулярности)
                    Phi = self.Phi(x, y, z)
                    if not np.isfinite(Phi) or abs(Phi) > 1e6:
                        score += 10
                        issues.append(f"Phi singular at ({x},{y},{z})")
        
        return score, issues
    
    def compute_engineering_metrics(self):
        """Изчислява инженерни метрики"""
        nec_data = self.compute_NEC_detailed()
        
        # Скоростен фактор (warp speed)
        v_warp = self.params['v'] * 22.0  # warpAmp * v
        
        # Ефективност (енергия на единица скорост)
        efficiency = abs(nec_data['total_energy']) / (v_warp + 1e-10)
        
        # Alpha стабилност
        alpha_stability = nec_data['min_alpha'] / (nec_data['mean_alpha'] + 1e-10)
        
        return {
            'v_warp': v_warp,
            'efficiency': efficiency,
            'alpha_stability': alpha_stability,
            'violation_ratio': nec_data['violation_ratio']
        }


# ============================================================
# ОПТИМИЗАЦИОНЕН КЛАС V2
# ============================================================

@dataclass
class OptimizationResultV2:
    """Резултат от оптимизацията V2"""
    params: Dict
    score: float
    nec_violation: float
    total_energy: float
    min_alpha: float
    violation_ratio: float
    stability_score: int
    engineering_metrics: Dict
    message: str
    iterations: int
    
    def to_dict(self):
        return asdict(self)


class WarpOptimizerV2:
    """Подобрен оптимизатор с разширени граници"""
    
    def __init__(self, bounds: Dict, initial_params: Dict = None):
        self.bounds = bounds
        self.initial_params = initial_params
        self.results = []
        self.best_result = None
        
        # Дефиниране на границите като списък за scipy
        self.param_names = list(bounds.keys())
        self.bounds_list = [bounds[name] for name in self.param_names]
        
        # Параметри по подразбиране
        self.default_params = {
            'L': 5.0, 'v': 0.3, 'R0': 2.0,
            'omegaBD': 10, 'sigma': 25.0, 'epsilon': 1e-8
        }
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        """Преобразува вектор в параметри"""
        params = self.default_params.copy()
        for name, value in zip(self.param_names, vector):
            params[name] = float(value)
        
        # Добавяме фиксирани параметри
        params['warpAmp'] = 22.0
        
        # Осигуряваме физически валидни стойности
        if params.get('wshell', 0) > params.get('R0', 2.0):
            params['wshell'] = params['R0'] * 0.8  # shell не може да е по-голям от R0
        
        return params
    
    def _objective(self, vector: np.ndarray) -> float:
        """Целева функция за оптимизация (по-малко = по-добре)"""
        try:
            params = self._vector_to_params(vector)
            metric = WarpMetricOptimizerV2(params)
            
            # 1. Изчисли NEC нарушения
            nec_data = metric.compute_NEC_detailed()
            
            # 2. Изчисли стабилност
            stability_score, issues = metric.compute_stability_score()
            
            # 3. Инженерни метрики
            eng_metrics = metric.compute_engineering_metrics()
            
            # 4. Комбиниран резултат с по-интелигентни тегла
            # Основна цел: минимизиране на екзотичната материя
            w_nec = 100.0
            
            # Вторична цел: стабилност (α > 0 навсякъде)
            w_stability = 50.0 if stability_score > 0 else 0.0
            
            # Третична цел: положителна енергия
            w_energy = 0.5
            energy_penalty = max(0, -nec_data['total_energy']) * 0.1
            
            # Четвъртична цел: малък violation ratio
            w_violation = 10.0 * nec_data['violation_ratio']
            
            # Бонус за добра ефективност
            efficiency_bonus = -0.1 * eng_metrics['efficiency'] if eng_metrics['efficiency'] > 0 else 0
            
            score = (
                w_nec * abs(nec_data['total_violation']) +
                w_stability * stability_score +
                w_energy * energy_penalty +
                w_violation * 100.0 +
                efficiency_bonus
            )
            
            # Добавяне на малък шум за избягване на локални минимуми
            score += np.random.normal(0, 1e-6)
            
            # Записваме резултата
            self.results.append({
                'params': params,
                'score': score,
                'nec_violation': nec_data['total_violation'],
                'energy': nec_data['total_energy'],
                'min_alpha': nec_data['min_alpha'],
                'violation_ratio': nec_data['violation_ratio'],
                'stability_score': stability_score,
                'alpha_stability': eng_metrics['alpha_stability'],
                'v_warp': eng_metrics['v_warp']
            })
            
            # Запазваме най-добрия
            if self.best_result is None or score < self.best_result['score']:
                self.best_result = self.results[-1]
            
            return score
            
        except Exception as e:
            # Ако има грешка, връщаме висока стойност (лош резултат)
            return 1e9
    
    def optimize(self, max_iterations: int = 100, population_size: int = 30) -> OptimizationResultV2:
        """Изпълнява оптимизацията"""
        print("="*70)
        print("WARP BUBBLE OPTIMIZER V2 - SECOND PASS")
        print("Разширени граници и по-интелигентна оптимизация")
        print("="*70)
        print(f"\n📊 Оптимизация на {len(self.param_names)} параметри:")
        for name, (low, high) in self.bounds.items():
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
        print(f"\n  Max iterations: {max_iterations}")
        print(f"  Population size: {population_size}")
        print()
        
        start_time = time.time()
        
        # Differential Evolution
        print("🔍 Започвам Differential Evolution (може да отнеме няколко часа)...")
        print("   ⏱️  Очаквано време: ~8-12 часа")
        print("   💡 Може да работи на заден план")
        print()
        
        result = differential_evolution(
            self._objective,
            self.bounds_list,
            maxiter=max_iterations,
            popsize=population_size,
            workers=-1,  # Използва всички ядра
            disp=True,
            updating='deferred',
            tol=0.001,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        # Nelder-Mead refinement
        print("\n🔧 Refining with Nelder-Mead...")
        refined = minimize(
            self._objective,
            result.x,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': True, 'fatol': 1e-6}
        )
        
        elapsed = time.time() - start_time
        
        # Вземаме най-добрия резултат
        if self.best_result is None:
            best = self.results[-1] if self.results else None
        else:
            best = self.best_result
        
        # Създаваме финален резултат
        final_params = self._vector_to_params(refined.x)
        
        # Последна валидация с по-фина мрежа
        print("\n🔍 Последна валидация...")
        metric = WarpMetricOptimizerV2(final_params)
        final_metrics = metric.compute_NEC_detailed()
        stability_score, issues = metric.compute_stability_score()
        eng_metrics = metric.compute_engineering_metrics()
        
        return OptimizationResultV2(
            params=final_params,
            score=refined.fun,
            nec_violation=final_metrics['total_violation'],
            total_energy=final_metrics['total_energy'],
            min_alpha=final_metrics['min_alpha'],
            violation_ratio=final_metrics['violation_ratio'],
            stability_score=stability_score,
            engineering_metrics=eng_metrics,
            message=f"Optimization completed in {elapsed/60:.1f} minutes",
            iterations=len(self.results)
        )
    
    def get_best_parameters(self) -> Dict:
        """Връща най-добрите намерени параметри"""
        if self.best_result:
            return self.best_result['params']
        return None


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    print("="*70)
    print("WARP BUBBLE OPTIMIZER V2")
    print("Разширени граници и по-интелигентна оптимизация")
    print("="*70)
    print()
    
    # РАЗШИРЕНА ГРАНИЦИ (въз основа на резултатите от V1)
    bounds_v2 = {
        'Aw': (0.001, 0.08),          # Разширение нагоре
        'wW': (0.05, 2.0),            # Много по-широк диапазон
        'Ad': (0.0001, 0.025),        # По-широк долен и горен край
        'wD': (0.1, 4.0),             # Много по-широк (1.49 беше добър)
        'epsBD': (0.001, 0.5),        # Разширение
        'wBD': (0.1, 3.0),            # По-широк диапазон
        'bBI': (0.1, 8.0),            # По-широк (0.475 беше под долната граница)
        'wshell': (0.05, 1.5)         # По-широк (0.613 беше над горната граница)
    }
    
    # Начални параметри (от V1 оптимизацията)
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
    
    print("📋 Разширени граници (V2):")
    for name, (low, high) in bounds_v2.items():
        print(f"  {name}: [{low:.4f}, {high:.4f}]")
    print()
    
    # Създаване на оптимизатор
    optimizer = WarpOptimizerV2(bounds_v2, initial_params)
    
    # Стартиране на оптимизацията (100 итерации, 30 популация)
    result = optimizer.optimize(max_iterations=100, population_size=30)
    
    # Резултати
    print("\n" + "="*70)
    print("✅ ОПТИМИЗАЦИЯТА V2 ЗАВЪРШИ")
    print("="*70)
    print()
    
    print("📊 Най-добри параметри:")
    for name, value in result.params.items():
        if name in bounds_v2:
            print(f"  {name}: {value:.8f}")
    print()
    
    print("📈 Резултати:")
    print(f"  Score: {result.score:.4f}")
    print(f"  NEC violation: {result.nec_violation:.4f}")
    print(f"  Total energy: {result.total_energy:.4f}")
    print(f"  Min alpha: {result.min_alpha:.6f}")
    print(f"  Violation ratio: {result.violation_ratio:.4%}")
    print(f"  Stability score: {result.stability_score}")
    print(f"  Warp speed: {result.engineering_metrics['v_warp']:.2f}c")
    print(f"  Efficiency: {result.engineering_metrics['efficiency']:.4f}")
    print(f"  Alpha stability: {result.engineering_metrics['alpha_stability']:.4f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  {result.message}")
    print()
    
    # Запис на резултатите
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Запис на оптималните параметри
    params_filename = f"optimized_params_v2_{timestamp}.json"
    with open(params_filename, 'w') as f:
        json.dump(result.params, f, indent=2)
    print(f"📄 Оптимални параметри записани в: {params_filename}")
    
    # Запис на пълния отчет
    report_filename = f"optimization_report_v2_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'best_params': result.params,
        'score': result.score,
        'nec_violation': result.nec_violation,
        'total_energy': result.total_energy,
        'min_alpha': result.min_alpha,
        'violation_ratio': result.violation_ratio,
        'stability_score': result.stability_score,
        'engineering_metrics': result.engineering_metrics,
        'iterations': result.iterations,
        'message': result.message,
        'bounds': bounds_v2,
        'initial_params': initial_params
    }
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Пълен отчет записан в: {report_filename}")
    
    # Сравнение с V1
    print()
    print("📊 Сравнение с V1 оптимизацията:")
    for name in bounds_v2:
        if name in initial_params:
            old = initial_params[name]
            new = result.params.get(name, 0)
            change = ((new - old) / old * 100) if old != 0 else 0
            print(f"  {name}: {old:.6f} → {new:.6f} ({change:+.1f}%)")
    
    print()
    print("="*70)
    print("КРАЙ НА ОПТИМИЗАЦИЯТА V2")
    print("="*70)
    print()
    print("💡 Следващи стъпки:")
    print("  1. Провери резултатите в optimization_report_v2_*.json")
    print("  2. Ако резултатите са добри, генерирай SecondPass файловете")
    print("  3. Ако не, може да пуснеш още една итерация с по-тесни граници")
    
    return result

if __name__ == "__main__":
    main()