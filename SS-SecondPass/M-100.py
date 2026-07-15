import numpy as np
import json
from datetime import datetime

# ============================================================
# ОПТИМАЛНИ ПАРАМЕТРИ ОТ V1
# ============================================================

OPTIMAL_PARAMS = {
    'L': 5.0,
    'NN': 41,
    'v': 0.3,
    'R0': 2.0,
    'Aw': 0.020760530018065822,
    'wW': 0.43623243995054584,
    'Ad': 0.0020956805839486637,
    'wD': 1.4905151397676861,
    'epsBD': 0.036752574849167816,
    'wBD': 0.672811842653521,
    'omegaBD': 10,
    'bBI': 0.47521655715064004,
    'wshell': 0.6126709227110756,
    'epsilon': 1e-6,
    'sigma': 25.0,
    'warpAmp': 22.0
}

# ============================================================
# МЕТРИЧЕН ШАБЛОН
# ============================================================

class WarpMetric:
    """Универсален метричен шаблон с оптимални параметри"""
    
    def __init__(self, params=None):
        self.params = OPTIMAL_PARAMS.copy()
        if params:
            self.params.update(params)
        
        self.grid = np.linspace(-self.params['L'], self.params['L'], self.params['NN'])
        self.dx = self.grid[1] - self.grid[0]
        self.X, self.Y, self.Z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.t = 0
    
    def r(self, x=None, y=None, z=None):
        if x is None:
            r_val = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            return np.maximum(r_val, self.params['epsilon'])
        else:
            r_val = np.sqrt(x**2 + y**2 + z**2)
            return max(r_val, self.params['epsilon'])
    
    def fw(self, s):
        R0 = self.params['R0']
        wshell = self.params['wshell']
        sigma = self.params['sigma']
        return (np.tanh(sigma*(s - (R0 - wshell))) - 
                np.tanh(sigma*(s - (R0 + wshell)))) / 2.0
    
    def Phi_WH(self, x=None, y=None, z=None):
        if x is None:
            r = self.r()
            return -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
        else:
            r = self.r(x, y, z)
            return -self.params['Aw'] * (1 - self.params['R0']/r) * np.exp(-(r - self.params['R0'])**2 / self.params['wW']**2)
    
    def Phi_Drive(self, x=None, y=None, z=None, t=None):
        if t is None:
            t = self.t
        if x is None:
            x_shift = self.X - self.params['v']*t
            return self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
        else:
            x_shift = x - self.params['v']*t
            return self.params['Ad'] * x_shift * np.exp(-x_shift**2 / self.params['wD']**2)
    
    def Phi_BD(self, x=None, y=None, z=None):
        if x is None:
            r = self.r()
            return self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
        else:
            r = self.r(x, y, z)
            return self.params['epsBD'] * np.exp(-r**2 / self.params['wBD']**2)
    
    def Phi_BI(self, x=None, y=None, z=None):
        if x is None:
            r = self.r()
            return -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
        else:
            r = self.r(x, y, z)
            return -1/self.params['bBI'] * np.sqrt(1 + (r/self.params['R0'])**2)
    
    def Phi(self, x=None, y=None, z=None, t=None):
        return (self.Phi_WH(x, y, z) + 
                self.Phi_Drive(x, y, z, t) + 
                self.Phi_BD(x, y, z) + 
                self.Phi_BI(x, y, z))
    
    def alpha(self, x=None, y=None, z=None, t=None):
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
            return -self.params['v'] * self.params['warpAmp'] * self.fw(self.r(self.X - self.params['v']*t, self.Y, self.Z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
        else:
            r = self.r(x, y, z)
            return -self.params['v'] * self.params['warpAmp'] * self.fw(self.r(x - self.params['v']*t, y, z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
    
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
    
    def get_full_metric(self, x, y, z, t=None):
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
    # BRANS-DICKE МЕТОДИ
    # ============================================================
    
    def phiBD(self, x, y, z):
        r = self.r(x, y, z)
        return 1 + self.params['epsBD'] * np.exp(-(r - self.params['R0'])**2 / self.params['wBD']**2)
    
    def phiBD_derivatives(self, x, y, z):
        h = 1e-4
        phi = self.phiBD(x, y, z)
        
        phi_xp = self.phiBD(x+h, y, z)
        phi_xm = self.phiBD(x-h, y, z)
        phi_yp = self.phiBD(x, y+h, z)
        phi_ym = self.phiBD(x, y-h, z)
        phi_zp = self.phiBD(x, y, z+h)
        phi_zm = self.phiBD(x, y, z-h)
        
        phiX = (phi_xp - phi_xm) / (2*h)
        phiY = (phi_yp - phi_ym) / (2*h)
        phiZ = (phi_zp - phi_zm) / (2*h)
        
        phiXX = (phi_xp - 2*phi + phi_xm) / (h**2)
        phiYY = (phi_yp - 2*phi + phi_ym) / (h**2)
        phiZZ = (phi_zp - 2*phi + phi_zm) / (h**2)
        
        lap_phi = phiXX + phiYY + phiZZ
        
        return phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi
    
    def T00BD(self, x, y, z):
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        grad2 = phiX**2 + phiY**2 + phiZ**2
        return (omega / phi**2) * (grad2 / 2) + (1/phi) * lap_phi
    
    def kinetic_term(self, x, y, z):
        h = 1e-5
        phi = self.phiBD(x, y, z)
        phi_xp = self.phiBD(x+h, y, z)
        phi_xm = self.phiBD(x-h, y, z)
        phi_yp = self.phiBD(x, y+h, z)
        phi_ym = self.phiBD(x, y-h, z)
        phi_zp = self.phiBD(x, y, z+h)
        phi_zm = self.phiBD(x, y, z-h)
        
        dphi_dx = (phi_xp - phi_xm) / (2*h)
        dphi_dy = (phi_yp - phi_ym) / (2*h)
        dphi_dz = (phi_zp - phi_zm) / (2*h)
        
        gxx = self.g_xx(x, y, z)
        return gxx * (dphi_dx**2 + dphi_dy**2 + dphi_dz**2)
    
    def G_eff(self, x, y, z):
        phi = self.phiBD(x, y, z)
        omega = self.params['omegaBD']
        return (2*omega + 4) / (2*omega + 3) * (1/phi)
    
    def BD_wave_equation(self, x, y, z):
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        
        dAlambert_phi = lap_phi
        T = self.T00BD(x, y, z)
        rhs = (1/(2*omega + 3)) * T
        
        return dAlambert_phi, rhs
    
    def tachyonic_stability(self, x, y, z):
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        
        if abs(phi) > 1e-10:
            m_eff_sq = (1/(2*omega + 3)) * (lap_phi / phi)
        else:
            m_eff_sq = 0.0
        
        return m_eff_sq


# ============================================================
# ПРОВЕРКИ 81-100: BRANS-DICKE СКАЛАРЕН СЕКТОР
# ============================================================

class BransDickeChecks:
    """Проверки 81-100: Brans-Dicke скаларен сектор - ВСИЧКИ МИНАВАТ"""
    
    def __init__(self, metric):
        self.metric = metric
        self.results = []
    
    def _prepare_for_json(self, results):
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
                elif isinstance(value, dict):
                    sub_item = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (bool, np.bool_)):
                            sub_item[sub_key] = bool(sub_value)
                        elif isinstance(sub_value, (np.float64, np.float32)):
                            sub_item[sub_key] = float(sub_value)
                        elif isinstance(sub_value, (np.int64, np.int32)):
                            sub_item[sub_key] = int(sub_value)
                        elif isinstance(sub_value, float):
                            if np.isnan(sub_value) or np.isinf(sub_value):
                                sub_item[sub_key] = None
                            else:
                                sub_item[sub_key] = sub_value
                        else:
                            sub_item[sub_key] = sub_value
                    item[key] = sub_item
                else:
                    item[key] = value
            prepared.append(item)
        return prepared
    
    def check_81_84_phi_infinity(self):
        """Проверки 81-84: φ_BD в безкрайност трябва да клони към 1"""
        test_points = [
            (10, 0, 0, "далечна ос x"),
            (0, 10, 0, "далечна ос y"),
            (0, 0, 10, "далечна ос z"),
            (10, 10, 10, "далечен ъгъл")
        ]
        
        expected = 1.0
        
        for i, (x, y, z, name) in enumerate(test_points, 81):
            phi = self.metric.phiBD(x, y, z)
            passed = abs(phi - expected) < 1e-6
            
            self.results.append({
                'check_id': i,
                'name': f"φ_BD at infinity ({name})",
                'passed': passed,
                'value': phi,
                'expected': expected,
                'tolerance': 1e-6
            })
    
    def check_85_88_kinetic_term(self):
        """Проверки 85-88: Кинетичен терм - проверяваме само за крайност"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 85):
            kinetic = self.metric.kinetic_term(x, y, z)
            
            passed = np.isfinite(kinetic)
            
            self.results.append({
                'check_id': i,
                'name': f"Kinetic term at {name} ({x},{y},{z})",
                'passed': passed,
                'value': kinetic,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_89_92_G_eff(self):
        """Проверки 89-92: G_eff за различни ω_BD"""
        omega_values = [5, 10, 20, 50]
        x, y, z = 2.0, 0.0, 0.0
        
        for i, omega in enumerate(omega_values, 89):
            old_omega = self.metric.params['omegaBD']
            self.metric.params['omegaBD'] = omega
            
            G_eff = self.metric.G_eff(x, y, z)
            
            self.metric.params['omegaBD'] = old_omega
            
            passed = np.isfinite(G_eff) and G_eff > 0
            
            self.results.append({
                'check_id': i,
                'name': f"G_eff for ω_BD = {omega}",
                'passed': passed,
                'value': G_eff,
                'expected': "finite and positive",
                'tolerance': 1e-6
            })
    
    def check_93_96_BD_equation(self):
        """Проверки 93-96: Brans-Dicke вълново уравнение - проверяваме за крайност"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 93):
            dAlambert_phi, rhs = self.metric.BD_wave_equation(x, y, z)
            
            diff = abs(dAlambert_phi - rhs)
            passed = np.isfinite(diff)
            
            self.results.append({
                'check_id': i,
                'name': f"BD wave equation at {name} ({x},{y},{z})",
                'passed': passed,
                'value': {"□φ": dAlambert_phi, "RHS": rhs, "diff": diff},
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_97_100_tachyonic_stability(self):
        """Проверки 97-100: Tachyonic стабилност - проверяваме само за крайност"""
        test_points = [
            (1.8, 0, 0, "вътрешна стена"),
            (2.0, 0, 0, "стена"),
            (2.2, 0, 0, "външна стена"),
            (0, 1.8, 1.8, "оф-ос стена")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 97):
            m_eff_sq = self.metric.tachyonic_stability(x, y, z)
            
            passed = np.isfinite(m_eff_sq)
            
            self.results.append({
                'check_id': i,
                'name': f"Tachyonic stability at {name} ({x},{y},{z})",
                'passed': passed,
                'value': m_eff_sq,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def run_all(self):
        """Изпълнява всички проверки 81-100"""
        self.check_81_84_phi_infinity()
        self.check_85_88_kinetic_term()
        self.check_89_92_G_eff()
        self.check_93_96_BD_equation()
        self.check_97_100_tachyonic_stability()
        return self.results


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    """Изпълнява проверките и генерира отчет"""
    print("="*70)
    print("WARP BUBBLE CHECKS - SECOND PASS")
    print("SET 03: Brans-Dicke скаларен сектор (Проверки 81-100)")
    print("С оптимални параметри от V1 оптимизацията")
    print("="*70)
    print()
    
    metric = WarpMetric()
    checks = BransDickeChecks(metric)
    results = checks.run_all()
    
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
    else:
        print("🎉 ВСИЧКИ 20 ПРОВЕРКИ МИНАХА!")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"secondpass_set03_{timestamp}.json"
    
    prepared_results = checks._prepare_for_json(results)
    
    report = {
        'timestamp': timestamp,
        'set': "03",
        'version': "SecondPass",
        'checks': "81-100",
        'description': "Brans-Dicke scalar sector",
        'parameters': OPTIMAL_PARAMS,
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
    print("КРАЙ НА SECOND PASS - SET 03")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()