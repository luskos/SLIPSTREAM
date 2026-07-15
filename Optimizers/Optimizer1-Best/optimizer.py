import numpy as np
import json
import time
from datetime import datetime
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# ============================================================
# МЕТРИЧЕН ШАБЛОН ЗА ОПТИМИЗАЦИЯ
# ============================================================

class WarpMetricOptimizer:
    """Лека версия на метриката за бърза оптимизация"""
    
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
        return (np.tanh(sigma*(s - (R0 - wshell))) - 
                np.tanh(sigma*(s - (R0 + wshell)))) / 2.0
    
    def Phi(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            r = self.r()
            # WH потенциал
            Phi_WH = -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
            # Drive потенциал
            x_shift = self.X - self.params['v']*t
            Phi_Drive = self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
            # BD потенциал
            Phi_BD = self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
            # BI потенциал
            Phi_BI = -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
            return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
        else:
            r = self.r(x, y, z)
            Phi_WH = -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
            x_shift = x - self.params['v']*t
            Phi_Drive = self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
            Phi_BD = self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
            Phi_BI = -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
            return Phi_WH + Phi_Drive + Phi_BD + Phi_BI
    
    def alpha(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            Phi = self.Phi()
            r = self.r()
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2))
        else:
            Phi = self.Phi(x, y, z, t)
            r = self.r(x, y, z)
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2))
    
    def beta_x(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            r = self.r()
            return -self.params['v'] * 22.0 * self.fw(self.r(self.X - self.params['v']*t, self.Y, self.Z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
        else:
            r = self.r(x, y, z)
            return -self.params['v'] * 22.0 * self.fw(self.r(x - self.params['v']*t, y, z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
    
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
    # МЕТОДИ ЗА ОПТИМИЗАЦИЯ
    # ============================================================
    
    def compute_NEC(self):
        """Изчислява NEC на редуцирана мрежа"""
        grid = self.grid[::2]  # всяка втора точка за скорост
        dx = self.dx * 2
        
        nec_violations = []
        alpha_vals = []
        
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
                    
                    # BD NEC (опростено)
                    phi_BD = 1 + self.params['epsBD'] * np.exp(-(self.r(x,y,z) - self.params['R0'])**2 / self.params['wBD']**2)
                    phi_x = (self.phiBD(x+h, y, z) - self.phiBD(x-h, y, z)) / (2*h)
                    phi_xx = (self.phiBD(x+h, y, z) - 2*phi_BD + self.phiBD(x-h, y, z)) / h**2
                    NEC_BD = (self.params['omegaBD'] / phi_BD**2) * phi_x**2 + (1/phi_BD) * phi_xx
                    
                    # BI NEC
                    E = np.exp(-(self.r(x,y,z) - self.params['R0'])**2 / self.params['wD']**2)
                    L_BI = self.params['bBI']**2 * (1 - np.sqrt(1 + 2*E**2/(2*self.params['bBI']**2)))
                    NEC_BI = -(self.LBI(x+h, y, z) - self.LBI(x-h, y, z)) / (2*h)
                    
                    # Shell NEC
                    shell = np.exp(-(self.r(x,y,z) - self.params['R0'])**2 / self.params['wshell']**2)
                    NEC_shell = 0.03 * shell
                    
                    # Тотален NEC
                    NEC_total = NEC_geom + NEC_BD + NEC_BI + NEC_shell
                    
                    if NEC_total < 0:
                        nec_violations.append(NEC_total)
                    
                    # Записваме alpha за проверка на стабилността
                    alpha_vals.append(self.alpha(x, y, z))
        
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
            'min_alpha': min(alpha_vals) if alpha_vals else 0
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
        return b**2 * (1 - np.sqrt(1 + F/(2*b**2)))
    
    def compute_energy(self):
        """Изчислява общата енергия"""
        grid = self.grid[::2]
        dx = self.dx * 2
        
        energy = 0
        for x in grid:
            for y in grid:
                for z in grid:
                    # Опростена T00
                    Phi = self.Phi(x, y, z)
                    h = 1e-4
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
                    
                    Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
                    T00_geom = Gtt / (8*np.pi)
                    
                    # BD T00
                    phi_BD = self.phiBD(x, y, z)
                    phi_x = (self.phiBD(x+h, y, z) - self.phiBD(x-h, y, z)) / (2*h)
                    phi_y = (self.phiBD(x, y+h, z) - self.phiBD(x, y-h, z)) / (2*h)
                    phi_z = (self.phiBD(x, y, z+h) - self.phiBD(x, y, z-h)) / (2*h)
                    phi_xx = (self.phiBD(x+h, y, z) - 2*phi_BD + self.phiBD(x-h, y, z)) / h**2
                    phi_yy = (self.phiBD(x, y+h, z) - 2*phi_BD + self.phiBD(x, y-h, z)) / h**2
                    phi_zz = (self.phiBD(x, y, z+h) - 2*phi_BD + self.phiBD(x, y, z-h)) / h**2
                    lap_phi = phi_xx + phi_yy + phi_zz
                    T00_BD = (self.params['omegaBD'] / phi_BD**2) * (phi_x**2 + phi_y**2 + phi_z**2) / 2 + (1/phi_BD) * lap_phi
                    
                    # BI T00
                    T00_BI = -self.LBI(x, y, z)
                    
                    # Shell T00
                    T00_shell = 0
                    
                    T00_total = T00_geom + T00_BD + T00_BI + T00_shell
                    energy += T00_total * dx**3
        
        return energy
    
    def check_stability(self):
        """Проверка на стабилността"""
        # Проверка на α > 0
        grid = self.grid[::2]
        alpha_vals = []
        for x in grid:
            for y in grid:
                for z in grid:
                    alpha = self.alpha(x, y, z)
                    alpha_vals.append(alpha)
        
        min_alpha = min(alpha_vals) if alpha_vals else 0
        alpha_stable = min_alpha > 0
        
        # Проверка за сингулярности
        try:
            for x in grid:
                for y in grid:
                    for z in grid:
                        Phi = self.Phi(x, y, z)
                        if not np.isfinite(Phi):
                            return False, "Singularity in Phi"
                        gtt = self.g_tt(x, y, z)
                        if not np.isfinite(gtt):
                            return False, "Singularity in g_tt"
        except:
            return False, "Numerical error"
        
        return alpha_stable, "OK" if alpha_stable else "alpha <= 0 detected"


# ============================================================
# ОПТИМИЗАЦИОНЕН КЛАС
# ============================================================

@dataclass
class OptimizationResult:
    """Резултат от оптимизацията"""
    params: Dict
    score: float
    nec_violation: float
    total_energy: float
    min_alpha: float
    violation_ratio: float
    message: str
    iterations: int
    
    def to_dict(self):
        return asdict(self)


class WarpOptimizer:
    """Оптимизатор за warp параметри"""
    
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
        
        return params
    
    def _objective(self, vector: np.ndarray) -> float:
        """Целева функция за оптимизация (по-малко = по-добре)"""
        try:
            params = self._vector_to_params(vector)
            metric = WarpMetricOptimizer(params)
            
            # 1. Изчисли NEC нарушения
            nec_data = metric.compute_NEC()
            
            # 2. Изчисли енергия
            energy = metric.compute_energy()
            
            # 3. Проверка на стабилност
            stable, msg = metric.check_stability()
            
            # 4. Комбиниран резултат
            # Тегла
            w_nec = 10.0      # Тежест на NEC нарушенията
            w_energy = 0.1    # Тежест на енергията
            w_stability = 100.0  # Тежест на стабилността (голяма = задължителна)
            
            score = (
                w_nec * abs(nec_data['total_violation']) +
                w_energy * abs(energy) +
                w_stability * (0 if stable else 1.0)
            )
            
            # Бонус за малко нарушения
            score += 10.0 * nec_data['violation_ratio']
            
            # Записваме резултата
            self.results.append({
                'params': params,
                'score': score,
                'nec_violation': nec_data['total_violation'],
                'energy': energy,
                'min_alpha': nec_data['min_alpha'],
                'violation_ratio': nec_data['violation_ratio'],
                'stable': stable
            })
            
            # Запазваме най-добрия
            if self.best_result is None or score < self.best_result['score']:
                self.best_result = self.results[-1]
            
            return score
            
        except Exception as e:
            # Ако има грешка, връщаме висока стойност (лош резултат)
            return 1e9
    
    def optimize(self, max_iterations: int = 100, population_size: int = 20) -> OptimizationResult:
        """Изпълнява оптимизацията"""
        print("="*70)
        print("WARP BUBBLE OPTIMIZER - SECOND PASS")
        print("="*70)
        print(f"\n📊 Оптимизация на {len(self.param_names)} параметри:")
        for name, (low, high) in self.bounds.items():
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
        print(f"\n  Max iterations: {max_iterations}")
        print(f"  Population size: {population_size}")
        print()
        
        start_time = time.time()
        
        # Differential Evolution
        print("🔍 Започвам Differential Evolution...")
        result = differential_evolution(
            self._objective,
            self.bounds_list,
            maxiter=max_iterations,
            popsize=population_size,
            workers=-1,  # Използва всички ядра
            disp=True,
            updating='deferred'
        )
        
        # Nelder-Mead refinement
        print("\n🔧 Refining with Nelder-Mead...")
        refined = minimize(
            self._objective,
            result.x,
            method='Nelder-Mead',
            options={'maxiter': 50, 'disp': True}
        )
        
        elapsed = time.time() - start_time
        
        # Вземаме най-добрия резултат
        if self.best_result is None:
            # Ако нямаме резултати, използваме последния
            best = self.results[-1] if self.results else None
        else:
            best = self.best_result
        
        # Създаваме финален резултат
        final_params = self._vector_to_params(refined.x)
        
        return OptimizationResult(
            params=final_params,
            score=refined.fun,
            nec_violation=best['nec_violation'] if best else 0,
            total_energy=best['energy'] if best else 0,
            min_alpha=best['min_alpha'] if best else 0,
            violation_ratio=best['violation_ratio'] if best else 0,
            message=f"Optimization completed in {elapsed:.1f}s",
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
    print("WARP BUBBLE OPTIMIZER - SECOND PASS")
    print("="*70)
    print()
    
    # Дефиниране на границите
    bounds = {
        'Aw': (0.001, 0.05),
        'wW': (0.1, 1.0),
        'Ad': (0.0005, 0.01),
        'wD': (0.2, 1.2),
        'epsBD': (0.001, 0.2),
        'wBD': (0.2, 1.0),
        'bBI': (0.5, 5.0),
        'wshell': (0.05, 0.5)
    }
    
    # Начални параметри (от FirstPass)
    initial_params = {
        'Aw': 0.027708718669180482,
        'wW': 0.531340720195939,
        'Ad': 0.005760176127668741,
        'wD': 0.2,
        'epsBD': 0.18742018325637566,
        'wBD': 0.5840568282558871,
        'bBI': 2.31927122395741,
        'wshell': 0.27365589631357656
    }
    
    print("📋 Начални параметри (FirstPass):")
    for name, value in initial_params.items():
        print(f"  {name}: {value:.6f}")
    print()
    
    # Създаване на оптимизатор
    optimizer = WarpOptimizer(bounds, initial_params)
    
    # Стартиране на оптимизацията
    result = optimizer.optimize(max_iterations=50, population_size=15)
    
    # Резултати
    print("\n" + "="*70)
    print("✅ ОПТИМИЗАЦИЯТА ЗАВЪРШИ")
    print("="*70)
    print()
    
    print("📊 Най-добри параметри:")
    for name, value in result.params.items():
        if name in bounds:
            print(f"  {name}: {value:.8f}")
    print()
    
    print("📈 Резултати:")
    print(f"  Score: {result.score:.4f}")
    print(f"  NEC violation: {result.nec_violation:.4f}")
    print(f"  Total energy: {result.total_energy:.4f}")
    print(f"  Min alpha: {result.min_alpha:.6f}")
    print(f"  Violation ratio: {result.violation_ratio:.4%}")
    print(f"  Iterations: {result.iterations}")
    print(f"  {result.message}")
    print()
    
    # Запис на резултатите
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Запис на оптималните параметри
    params_filename = f"optimized_params_{timestamp}.json"
    with open(params_filename, 'w') as f:
        json.dump(result.params, f, indent=2)
    print(f"📄 Оптимални параметри записани в: {params_filename}")
    
    # Запис на пълния отчет
    report_filename = f"optimization_report_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'best_params': result.params,
        'score': result.score,
        'nec_violation': result.nec_violation,
        'total_energy': result.total_energy,
        'min_alpha': result.min_alpha,
        'violation_ratio': result.violation_ratio,
        'iterations': result.iterations,
        'message': result.message,
        'bounds': bounds,
        'initial_params': initial_params
    }
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Пълен отчет записан в: {report_filename}")
    
    # Сравнение с началните параметри
    print()
    print("📊 Сравнение с FirstPass:")
    for name in bounds:
        old = initial_params.get(name, 0)
        new = result.params.get(name, 0)
        change = ((new - old) / old * 100) if old != 0 else 0
        print(f"  {name}: {old:.6f} → {new:.6f} ({change:+.1f}%)")
    
    print()
    print("="*70)
    print("КРАЙ")
    print("="*70)
    
    return result

if __name__ == "__main__":
    main()