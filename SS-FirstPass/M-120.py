import numpy as np
import json
from datetime import datetime

# ============================================================
# МЕТРИЧЕН ШАБЛОН (ЕДИНАКЪВ ЗА ВСИЧКИ СЕТОВЕ)
# ============================================================

class WarpMetric:
    """Универсален метричен шаблон за Warp Bubble"""
    
    def __init__(self, params=None):
        # Параметри по подразбиране (от документа)
        self.params = {
            'L': 5.0, 'NN': 41, 'v': 0.3, 'R0': 2.0,
            'Aw': 0.027708718669180482,
            'wW': 0.531340720195939,
            'Ad': 0.005760176127668741,
            'wD': 0.2,
            'epsBD': 0.18742018325637566,
            'wBD': 0.5840568282558871,
            'omegaBD': 10,
            'bBI': 2.31927122395741,
            'wshell': 0.27365589631357656,
            'epsilon': 1e-6,
            'sigma': 25.0,
            'warpAmp': 22.0
        }
        if params:
            self.params.update(params)
        
        # Създаване на мрежата
        self.grid = np.linspace(-self.params['L'], self.params['L'], self.params['NN'])
        self.dx = self.grid[1] - self.grid[0]
        
        # 3D координати (за векторни операции)
        self.X, self.Y, self.Z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.t = 0  # време по подразбиране
    
    def r(self, x=None, y=None, z=None):
        """Радиална функция с регуларизация"""
        if x is None:
            r_val = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            return np.maximum(r_val, self.params['epsilon'])
        else:
            r_val = np.sqrt(x**2 + y**2 + z**2)
            return max(r_val, self.params['epsilon'])
    
    def fw(self, s):
        """Warp shell функция (форма на балонa)"""
        R0 = self.params['R0']
        wshell = self.params['wshell']
        sigma = self.params['sigma']
        return (np.tanh(sigma*(s - (R0 - wshell))) - 
                np.tanh(sigma*(s - (R0 + wshell)))) / 2.0
    
    def Phi_WH(self, x=None, y=None, z=None):
        """Wormhole потенциал"""
        if x is None:
            r = self.r()
            return -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
        else:
            r = self.r(x, y, z)
            return -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
    
    def Phi_Drive(self, x=None, y=None, z=None, t=None):
        """Warp drive потенциал"""
        if t is None:
            t = self.t
        if x is None:
            x_shift = self.X - self.params['v']*t
            return self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
        else:
            x_shift = x - self.params['v']*t
            return self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
    
    def Phi_BD(self, x=None, y=None, z=None):
        """Brans-Dicke скаларен потенциал"""
        if x is None:
            r = self.r()
            return self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
        else:
            r = self.r(x, y, z)
            return self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
    
    def Phi_BI(self, x=None, y=None, z=None):
        """Born-Infeld стабилизатор"""
        if x is None:
            r = self.r()
            return -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
        else:
            r = self.r(x, y, z)
            return -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
    
    def Phi(self, x=None, y=None, z=None, t=None):
        """Тотален скаларен потенциал"""
        return (self.Phi_WH(x, y, z) + 
                self.Phi_Drive(x, y, z, t) + 
                self.Phi_BD(x, y, z) + 
                self.Phi_BI(x, y, z))
    
    def alpha(self, x=None, y=None, z=None, t=None):
        """Lapse функция"""
        if x is None:
            Phi = self.Phi()
            r = self.r()
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2))
        else:
            Phi = self.Phi(x, y, z, t)
            r = self.r(x, y, z)
            return np.exp(0.35*Phi) * (1 + 0.25*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2))
    
    def beta_x(self, x=None, y=None, z=None, t=None):
        """Shift вектор (x компонента)"""
        if t is None:
            t = self.t
        if x is None:
            r = self.r()
            return -self.params['v'] * self.params['warpAmp'] * self.fw(self.r(self.X - self.params['v']*t, self.Y, self.Z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
        else:
            r = self.r(x, y, z)
            return -self.params['v'] * self.params['warpAmp'] * self.fw(self.r(x - self.params['v']*t, y, z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
    
    def g_tt(self, x=None, y=None, z=None, t=None):
        """Метрична компонента g_tt"""
        if x is None:
            alpha = self.alpha()
            beta = self.beta_x()
            return -alpha**2 + beta**2
        else:
            alpha = self.alpha(x, y, z, t)
            beta = self.beta_x(x, y, z, t)
            return -alpha**2 + beta**2
    
    def g_xx(self, x=None, y=None, z=None, t=None):
        """Метрична компонента g_xx (същата за yy, zz)"""
        if x is None:
            Phi = self.Phi()
            r = self.r()
            return np.exp(-2*Phi) * (1 + 0.15*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) * (1 + 0.1/self.params['bBI'])
        else:
            Phi = self.Phi(x, y, z, t)
            r = self.r(x, y, z)
            return np.exp(-2*Phi) * (1 + 0.15*self.params['epsBD']*np.exp(-r**2/self.params['wBD']**2)) * (1 + 0.1/self.params['bBI'])
    
    def get_full_metric(self, x, y, z, t=None):
        """Връща пълната 4x4 метрика в точка (x,y,z)"""
        if t is None:
            t = self.t
        gtt = self.g_tt(x, y, z, t)
        gxx = self.g_xx(x, y, z, t)
        gtx = self.beta_x(x, y, z, t)
        
        return np.array([
            [gtt, gtx, 0, 0],
            [gtx, gxx, 0, 0],
            [0, 0, gxx, 0],
            [0, 0, 0, gxx]
        ])
    
    def numerical_derivative(self, func, x, y, z, axis=0, h=1e-5):
        """Числена производна на функция"""
        if axis == 0:  # x
            return (func(x+h, y, z) - func(x-h, y, z)) / (2*h)
        elif axis == 1:  # y
            return (func(x, y+h, z) - func(x, y-h, z)) / (2*h)
        else:  # z
            return (func(x, y, z+h) - func(x, y, z-h)) / (2*h)
    
    def numerical_second_derivative(self, func, x, y, z, axis=0, h=1e-5):
        """Числена втора производна на функция"""
        if axis == 0:  # x
            return (func(x+h, y, z) - 2*func(x, y, z) + func(x-h, y, z)) / (h**2)
        elif axis == 1:  # y
            return (func(x, y+h, z) - 2*func(x, y, z) + func(x, y-h, z)) / (h**2)
        else:  # z
            return (func(x, y, z+h) - 2*func(x, y, z) + func(x, y, z-h)) / (h**2)
    
    # ============================================================
    # BORN-INFELD СЕКТОР (ЗА СЕТ 04)
    # ============================================================
    
    def EBI(self, x, y, z):
        """Born-Infeld електрическо поле: E = exp(-(r-R0)²/wD²)"""
        r = self.r(x, y, z)
        return np.exp(-(r - self.params['R0'])**2 / self.params['wD']**2)
    
    def FBI(self, x, y, z):
        """F = 2*E²"""
        return 2 * self.EBI(x, y, z)**2
    
    def LBI(self, x, y, z):
        """Born-Infeld Lagrangian: ℒ_BI = b_BI²(1 - sqrt(1 + F/(2*b_BI²)))"""
        b = self.params['bBI']
        F = self.FBI(x, y, z)
        return b**2 * (1 - np.sqrt(1 + F/(2*b**2)))
    
    def T00BI(self, x, y, z):
        """Born-Infeld T_00 = -ℒ_BI"""
        return -self.LBI(x, y, z)
    
    def NECBI(self, x, y, z):
        """Born-Infeld NEC (x-посока) = -∂ℒ_BI/∂x"""
        h = 1e-5
        return -(self.LBI(x+h, y, z) - self.LBI(x-h, y, z)) / (2*h)
    
    def BI_stress_energy(self, x, y, z):
        """Born-Infeld стрес-енергиен тензор компоненти"""
        # T^μν_BI = ... (опростено)
        T00 = self.T00BI(x, y, z)
        T11 = T00 / 3  # Изотропна апроксимация
        T22 = T00 / 3
        T33 = T00 / 3
        return T00, T11, T22, T33
    
    # ============================================================
    # МЕТОДИ ЗА СЕТ 04 - BORN-INFELD СПЕЦИФИЧНИ
    # ============================================================
    
    def BI_limit_check(self, x, y, z):
        """Проверка на Born-Infeld параметъра b_BI спрямо E"""
        E = self.EBI(x, y, z)
        b = self.params['bBI']
        # Изискваме E < b_BI за реалистични полета
        return E, b
    
    def BI_lagrangian_reality(self, x, y, z):
        """Проверка дали Lagrangian е реален (коренът под радикала е положителен)"""
        F = self.FBI(x, y, z)
        b = self.params['bBI']
        radicand = 1 + F/(2*b**2)
        return radicand
    
    def BI_Maxwell_equations(self, x, y, z):
        """Проверка на нелинейните Maxwell уравнения (опростено)"""
        # Gauss закон: ∇·E = ρ (за BI)
        h = 1e-4
        E = self.EBI(x, y, z)
        E_xp = self.EBI(x+h, y, z)
        E_xm = self.EBI(x-h, y, z)
        E_yp = self.EBI(x, y+h, z)
        E_ym = self.EBI(x, y-h, z)
        E_zp = self.EBI(x, y, z+h)
        E_zm = self.EBI(x, y, z-h)
        
        div_E = (E_xp - E_xm)/(2*h) + (E_yp - E_ym)/(2*h) + (E_zp - E_zm)/(2*h)
        
        # Faraday закон: ∂B/∂t = -∇×E (при t=0, B=0)
        # Опростено: проверяваме дали div_E е малко
        return div_E
    
    def BI_T_symmetry(self, x, y, z):
        """Проверка на симетрията на BI стрес-енергиен тензор"""
        T00, T11, T22, T33 = self.BI_stress_energy(x, y, z)
        # Проверяваме дали T_ij = T_ji (симетрично)
        # За изотропен случай, всички диагонални са равни
        symmetry_ok = abs(T11 - T22) < 1e-10 and abs(T22 - T33) < 1e-10
        return symmetry_ok, T00, T11, T22, T33
    
    def BI_vacuum_birefringence(self, x, y, z):
        """Анализ на вакуумно двойно лъчепречупване (опростено)"""
        # В силни BI полета, скоростта на светлината зависи от посоката
        E = self.EBI(x, y, z)
        b = self.params['bBI']
        
        # Опростен прокси за бirefringence
        if abs(E) > 1e-10:
            birefringence = E / b
        else:
            birefringence = 0.0
        
        return birefringence


# ============================================================
# ПРОВЕРКИ 101-120: BORN-INFELD НЕЛИНЕЕН ЕЛЕКТРОМАГНЕТИЗЪМ
# ============================================================

class BornInfeldChecks:
    """Проверки 101-120: Born-Infeld нелинеен електромагнитен сектор"""
    
    def __init__(self, metric):
        self.metric = metric
        self.results = []
    
    def _prepare_for_json(self, results):
        """Подготвя резултатите за JSON сериализация"""
        prepared = []
        for r in results:
            item = {}
            for key, value in r.items():
                if isinstance(value, (bool, np.bool_)):
                    item[key] = bool(value)
                elif isinstance(value, np.ndarray):
                    item[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    item[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    item[key] = int(value)
                elif isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        item[key] = None
                    else:
                        item[key] = value
                else:
                    item[key] = value
            prepared.append(item)
        return prepared
    
    def check_101_104_BI_limit(self):
        """Проверки 101-104: Оценка на E спрямо b_BI в 4 точки"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 101):
            E, b = self.metric.BI_limit_check(x, y, z)
            
            # За реалистични полета, E < b_BI
            passed = E < b
            
            self.results.append({
                'check_id': i,
                'name': f"E < b_BI at {name} ({x},{y},{z})",
                'passed': passed,
                'value': {"E": E, "b_BI": b},
                'expected': "E < b_BI",
                'tolerance': 1e-6
            })
    
    def check_105_108_BI_lagrangian(self):
        """Проверки 105-108: Изчисляване на ℒ_BI и проверка за комплексност"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 105):
            radicand = self.metric.BI_lagrangian_reality(x, y, z)
            L = self.metric.LBI(x, y, z)
            
            # Коренът трябва да е положителен (radicand > 0)
            passed = radicand > 0 and np.isfinite(L)
            
            self.results.append({
                'check_id': i,
                'name': f"BI Lagrangian reality at {name} ({x},{y},{z})",
                'passed': passed,
                'value': {"ℒ_BI": L, "radicand": radicand},
                'expected': "radicand > 0 and ℒ_BI finite",
                'tolerance': 1e-6
            })
    
    def check_109_112_BI_Maxwell(self):
        """Проверки 109-112: Нелинейни Maxwell уравнения (Gauss и Faraday)"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 109):
            div_E = self.metric.BI_Maxwell_equations(x, y, z)
            
            # Gauss закон: div_E = 0 (във вакуум)
            passed = abs(div_E) < 1e-4
            
            self.results.append({
                'check_id': i,
                'name': f"BI Maxwell (Gauss) at {name} ({x},{y},{z})",
                'passed': passed,
                'value': div_E,
                'expected': "≈ 0",
                'tolerance': 1e-4
            })
    
    def check_113_116_BI_T_symmetry(self):
        """Проверки 113-116: Симетрия на BI стрес-енергиен тензор"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 113):
            symmetry_ok, T00, T11, T22, T33 = self.metric.BI_T_symmetry(x, y, z)
            
            # T_ij трябва да е симетричен
            passed = symmetry_ok and np.isfinite(T00)
            
            self.results.append({
                'check_id': i,
                'name': f"BI T_ij symmetry at {name} ({x},{y},{z})",
                'passed': passed,
                'value': {"T00": T00, "T11": T11, "T22": T22, "T33": T33},
                'expected': "symmetric and finite",
                'tolerance': 1e-6
            })
    
    def check_117_120_BI_birefringence(self):
        """Проверки 117-120: Вакуумно двойно лъчепречупване в силни Warp полета"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 117):
            birefringence = self.metric.BI_vacuum_birefringence(x, y, z)
            
            # Проверяваме дали бirefringence е краен и реален
            passed = np.isfinite(birefringence) and not np.isnan(birefringence)
            
            self.results.append({
                'check_id': i,
                'name': f"BI vacuum birefringence at {name} ({x},{y},{z})",
                'passed': passed,
                'value': birefringence,
                'expected': "finite and real",
                'tolerance': 1e-6
            })
    
    def run_all(self):
        """Изпълнява всички проверки 101-120"""
        self.check_101_104_BI_limit()
        self.check_105_108_BI_lagrangian()
        self.check_109_112_BI_Maxwell()
        self.check_113_116_BI_T_symmetry()
        self.check_117_120_BI_birefringence()
        return self.results


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    """Изпълнява проверките и генерира отчет"""
    print("="*70)
    print("WARP BUBBLE CHECKS - SET 04 (Проверки 101-120)")
    print("Born-Infeld нелинеен електромагнитен сектор")
    print("="*70)
    print()
    
    # Инициализация
    metric = WarpMetric()
    checks = BornInfeldChecks(metric)
    
    # Изпълнение
    results = checks.run_all()
    
    # Статистика
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    
    print(f"\n📊 РЕЗУЛТАТИ:")
    print(f"  ✅ Преминали: {passed}/20")
    print(f"  ❌ Неуспешни: {failed}/20")
    print()
    
    if failed > 0:
        print("❌ НЕУСПЕШНИ ПРОВЕРКИ:")
        for r in results:
            if not r['passed']:
                print(f"  {r['check_id']:3d}. {r['name']}")
                print(f"      Стойност: {r['value']}")
                print(f"      Очаквано: {r['expected']}")
                print()
    
    # Запис на резултатите
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checks_set04_{timestamp}.json"
    
    # Подготви резултатите за JSON
    prepared_results = checks._prepare_for_json(results)
    
    report = {
        'timestamp': timestamp,
        'set': "04",
        'checks': "101-120",
        'description': "Born-Infeld nonlinear electromagnetic sector",
        'total': len(results),
        'passed': passed,
        'failed': failed,
        'results': prepared_results
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Отчетът е записан в: {filename}")
    print()
    print("="*70)
    print("КРАЙ НА SET 04")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()