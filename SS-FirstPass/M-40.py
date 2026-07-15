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
            # Връща 3D масив
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
            x_shift = self.X - self.params['v']*t
            return -self.params['v'] * self.params['warpAmp'] * self.fw(self.r(self.X - self.params['v']*t, self.Y, self.Z)) / np.sqrt(1 + (r/(self.params['bBI']*self.params['wW']))**2)
        else:
            r = self.r(x, y, z)
            x_shift = x - self.params['v']*t
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
# ПРОВЕРКИ 1-40: ГЕОМЕТРИЧНИ И МЕТРИЧНИ ТЕСТОВЕ
# ============================================================

class GeometricChecks:
    """Проверки 1-40: Геометрични и метрични тестове"""
    
    def __init__(self, metric):
        self.metric = metric
        self.results = []
    
    def _prepare_for_json(self, results):
        """Подготвя резултатите за JSON сериализация"""
        prepared = []
        for r in results:
            item = {}
            for key, value in r.items():
                # Проверка за всички видове bool
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
    
    def check_1_4_signature(self):
        """Проверки 1-4: Сигнатура на метриката в 4 точки"""
        test_points = [
            (0, 0, 0, "център"),
            (2, 0, 0, "стена"),
            (4, 0, 0, "външен ръб"),
            (4.9, 4.9, 4.9, "ъгъл")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 1):
            g = self.metric.get_full_metric(x, y, z)
            # Използвай eigvals вместо eigvalsh (работи с несиметрични матрици)
            eigenvals = np.linalg.eigvals(g)
            # Закръгли до най-близкото цяло число
            signature = np.round(np.sign(eigenvals.real))
            
            # Очакваме (-, +, +, +)
            expected = np.array([-1, 1, 1, 1])
            passed = np.array_equal(signature, expected)
            
            self.results.append({
                'check_id': i,
                'name': f"Signature at {name} ({x},{y},{z})",
                'passed': passed,
                'value': signature.tolist(),
                'expected': expected.tolist(),
                'tolerance': 1e-6
            })
    
    def check_5_8_determinant(self):
        """Проверки 5-8: Детерминанта на метриката"""
        test_points = [
            (0, 0, 0, "център"),
            (2, 0, 0, "стена"),
            (4, 0, 0, "външен ръб"),
            (4.9, 4.9, 4.9, "ъгъл")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 5):
            g = self.metric.get_full_metric(x, y, z)
            det = np.linalg.det(g)
            
            # Детерминантата не трябва да е 0 или безкрайност
            passed = not (np.isclose(det, 0, atol=1e-10) or np.isinf(det))
            
            self.results.append({
                'check_id': i,
                'name': f"Determinant at {name} ({x},{y},{z})",
                'passed': passed,
                'value': det,
                'expected': "finite and non-zero",
                'tolerance': 1e-10
            })
    
    def check_9_12_inverse_metric(self):
        """Проверки 9-12: Обратна метрика"""
        test_points = [
            (0, 0, 0, "център"),
            (2, 0, 0, "стена"),
            (4, 0, 0, "външен ръб"),
            (4.9, 4.9, 4.9, "ъгъл")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 9):
            g = self.metric.get_full_metric(x, y, z)
            try:
                g_inv = np.linalg.inv(g)
                # Проверка: g * g_inv = I
                identity = np.eye(4)
                product = g @ g_inv
                passed = np.allclose(product, identity, atol=1e-8)
                value = "invertible"
            except np.linalg.LinAlgError:
                passed = False
                value = "singular"
            
            self.results.append({
                'check_id': i,
                'name': f"Inverse metric at {name} ({x},{y},{z})",
                'passed': passed,
                'value': value,
                'expected': "invertible",
                'tolerance': 1e-8
            })
    
    def check_13_16_four_velocity(self):
        """Проверки 13-16: Четири-скорост на комоving observer"""
        test_points = [
            (0, 0, 0, "център"),
            (2, 0, 0, "стена"),
            (4, 0, 0, "външен ръб"),
            (4.9, 4.9, 4.9, "ъгъл")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 13):
            alpha = self.metric.alpha(x, y, z)
            beta = self.metric.beta_x(x, y, z)
            
            # Правилната формула за Eulerian observer:
            # u^μ = (1/α, -β^i/α, 0, 0)
            u = np.array([1/alpha, -beta/alpha, 0, 0])
            g = self.metric.get_full_metric(x, y, z)
            
            # u_μ = g_{μν} u^ν
            u_mu = g @ u
            norm = np.dot(u, u_mu)
            
            # Трябва да е -1 (проверка с толеранс)
            passed = np.isclose(norm, -1, rtol=1e-6, atol=1e-6)
            
            self.results.append({
                'check_id': i,
                'name': f"Four-velocity norm at {name} ({x},{y},{z})",
                'passed': passed,
                'value': norm,
                'expected': -1,
                'tolerance': 1e-6
            })
    
    def check_17_24_christoffel(self):
        """Проверки 17-24: Christoffel символи"""
        points = [(0,0,0), (2,0,0), (4,0,0)]
        
        for i, (x, y, z) in enumerate(points, 17):
            g = self.metric.get_full_metric(x, y, z)
            g_inv = np.linalg.inv(g)
            
            h = 1e-5
            # ∂_x g_tt
            gtt_x = (self.metric.g_tt(x+h, y, z) - self.metric.g_tt(x-h, y, z)) / (2*h)
            # ∂_x g_tx
            gtx_x = (self.metric.beta_x(x+h, y, z) - self.metric.beta_x(x-h, y, z)) / (2*h)
            # ∂_t g_xx (при t=0)
            dt = 1e-5
            gxx_t = (self.metric.g_xx(x, y, z, dt) - self.metric.g_xx(x, y, z, -dt)) / (2*dt)
            
            # Γ^t_{tx} = 1/2 g^{tt}(∂_t g_{tx} + ∂_x g_{tt} - ∂_t g_{tx})
            Gamma_t_tx = 0.5 * g_inv[0,0] * gtt_x
            
            passed = np.isfinite(Gamma_t_tx)
            
            self.results.append({
                'check_id': 17 + i,
                'name': f"Christoffel Γ^t_tx at ({x},{y},{z})",
                'passed': passed,
                'value': Gamma_t_tx,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_25_28_ricci(self):
        """Проверки 25-28: Ricci тензор"""
        points = [(0,0,0), (2,0,0), (4,0,0), (4.9,4.9,4.9)]
        
        for i, (x, y, z) in enumerate(points, 25):
            h = 1e-4
            gtt = self.metric.g_tt(x, y, z)
            gtt_xp = self.metric.g_tt(x+h, y, z)
            gtt_xm = self.metric.g_tt(x-h, y, z)
            gtt_yp = self.metric.g_tt(x, y+h, z)
            gtt_ym = self.metric.g_tt(x, y-h, z)
            gtt_zp = self.metric.g_tt(x, y, z+h)
            gtt_zm = self.metric.g_tt(x, y, z-h)
            
            lap_gtt = (gtt_xp - 2*gtt + gtt_xm)/h**2 + \
                      (gtt_yp - 2*gtt + gtt_ym)/h**2 + \
                      (gtt_zp - 2*gtt + gtt_zm)/h**2
            
            R_tt = -0.5 * lap_gtt
            
            passed = np.isfinite(R_tt)
            
            self.results.append({
                'check_id': i,
                'name': f"Ricci R_tt at ({x},{y},{z})",
                'passed': passed,
                'value': R_tt,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_29_32_covariant_constancy(self):
        """Проверки 29-32: Ковариантна константност на метриката ∇_ρ g_μν = 0"""
        directions = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]
        x, y, z = 2, 0, 0
        
        for i, (dx, dy, dz) in enumerate(directions, 29):
            h = 1e-5
            g_center = self.metric.get_full_metric(x, y, z)
            g_plus = self.metric.get_full_metric(x + h*dx, y + h*dy, z + h*dz)
            g_minus = self.metric.get_full_metric(x - h*dx, y - h*dy, z - h*dz)
            
            derivative = (g_plus - g_minus) / (2*h)
            
            max_deriv = np.max(np.abs(derivative))
            max_metric = np.max(np.abs(g_center))
            
            if max_metric > 1e-10:
                relative_change = max_deriv / max_metric
            else:
                relative_change = max_deriv
            
            # По-строг толеранс за covariant constancy
            passed = relative_change < 0.001
            
            self.results.append({
                'check_id': i,
                'name': f"Covariant constancy in direction ({dx},{dy},{dz})",
                'passed': passed,
                'value': relative_change,
                'expected': 0,
                'tolerance': 0.001
            })
    
    def check_33_36_ricci_tensor(self):
        """Проверки 33-36: Ricci тензор компоненти"""
        points = [(0,0,0), (2,0,0), (4,0,0), (4.9,4.9,4.9)]
        
        for i, (x, y, z) in enumerate(points, 33):
            h = 1e-4
            gxx = self.metric.g_xx(x, y, z)
            gxx_xp = self.metric.g_xx(x+h, y, z)
            gxx_xm = self.metric.g_xx(x-h, y, z)
            
            R_xx = -(gxx_xp - 2*gxx + gxx_xm)/h**2
            
            passed = np.isfinite(R_xx)
            
            self.results.append({
                'check_id': i,
                'name': f"Ricci R_xx at ({x},{y},{z})",
                'passed': passed,
                'value': R_xx,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_37_40_ricci_scalar(self):
        """Проверки 37-40: Ricci скалар в 4 ключови региона"""
        regions = [
            (0, 0, 0, "център"),
            (2, 0, 0, "стена"),
            (4.5, 0, 0, "външен ръб"),
            (100, 0, 0, "безкрайност (асимптотично)")
        ]
        
        for i, (x, y, z, name) in enumerate(regions, 37):
            Phi = self.metric.Phi(x, y, z)
            
            h = 1e-4
            Phi_xp = self.metric.Phi(x+h, y, z)
            Phi_xm = self.metric.Phi(x-h, y, z)
            Phi_yp = self.metric.Phi(x, y+h, z)
            Phi_ym = self.metric.Phi(x, y-h, z)
            Phi_zp = self.metric.Phi(x, y, z+h)
            Phi_zm = self.metric.Phi(x, y, z-h)
            
            lap_Phi = (Phi_xp - 2*Phi + Phi_xm)/h**2 + \
                      (Phi_yp - 2*Phi + Phi_ym)/h**2 + \
                      (Phi_zp - 2*Phi + Phi_zm)/h**2
            
            grad_Phi = ((Phi_xp - Phi_xm)/(2*h))**2 + \
                       ((Phi_yp - Phi_ym)/(2*h))**2 + \
                       ((Phi_zp - Phi_zm)/(2*h))**2
            
            R = 2*lap_Phi - 2*grad_Phi
            
            if name == "безкрайност (асимптотично)":
                passed = abs(R) < 0.001
            else:
                passed = np.isfinite(R)
            
            self.results.append({
                'check_id': i,
                'name': f"Ricci scalar at {name} ({x},{y},{z})",
                'passed': passed,
                'value': R,
                'expected': 0 if name == "безкрайност (асимптотично)" else "finite",
                'tolerance': 0.001
            })
    
    def run_all(self):
        """Изпълнява всички проверки 1-40"""
        self.check_1_4_signature()
        self.check_5_8_determinant()
        self.check_9_12_inverse_metric()
        self.check_13_16_four_velocity()
        self.check_17_24_christoffel()
        self.check_25_28_ricci()
        self.check_29_32_covariant_constancy()
        self.check_33_36_ricci_tensor()
        self.check_37_40_ricci_scalar()
        return self.results


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def diagnose_point():
    """Диагностика на проблемната точка (2,0,0)"""
    print("="*70)
    print("🔍 ДИАГНОСТИКА В ТОЧКА (2,0,0)")
    print("="*70)
    
    metric = WarpMetric()
    x, y, z = 2.0, 0.0, 0.0
    
    # Изчисли всички компоненти
    Phi = metric.Phi(x, y, z)
    alpha = metric.alpha(x, y, z)
    beta = metric.beta_x(x, y, z)
    gtt = metric.g_tt(x, y, z)
    gxx = metric.g_xx(x, y, z)
    
    print(f"\n  📐 Скаларни величини:")
    print(f"     Phi = {Phi:.6f}")
    print(f"     alpha = {alpha:.6f}")
    print(f"     beta_x = {beta:.6f}")
    print(f"     g_tt = {gtt:.6f}")
    print(f"     g_xx = {gxx:.6f}")
    
    # Проверка на сигнатурата
    g = metric.get_full_metric(x, y, z)
    print(f"\n  📊 Пълна метрика:")
    print(f"     {g}")
    
    eigenvals = np.linalg.eigvals(g)
    print(f"\n  📈 Собствени стойности:")
    print(f"     {eigenvals}")
    print(f"     Сигнатура: {np.sign(eigenvals)}")
    
    # Проверка на four-velocity
    u = np.array([1/alpha, -beta/alpha, 0, 0])
    u_mu = g @ u
    norm = np.dot(u, u_mu)
    print(f"\n  🚀 Four-velocity:")
    print(f"     u^μ = {u}")
    print(f"     u_μ = {u_mu}")
    print(f"     u^μ u_μ = {norm:.6f}")
    
    print("\n" + "="*70 + "\n")

def main():
    """Изпълнява проверките и генерира отчет"""
    
    # Първо диагностика
    diagnose_point()
    
    print("="*70)
    print("WARP BUBBLE CHECKS - SET 01 (Проверки 1-40)")
    print("Геометрични и метрични тестове")
    print("="*70)
    print()
    
    # Инициализация
    metric = WarpMetric()
    checks = GeometricChecks(metric)
    
    # Изпълнение
    results = checks.run_all()
    
    # Статистика
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    
    print(f"\n📊 РЕЗУЛТАТИ:")
    print(f"  ✅ Преминали: {passed}/40")
    print(f"  ❌ Неуспешни: {failed}/40")
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
    filename = f"checks_set01_{timestamp}.json"
    
    # Подготви резултатите за JSON
    prepared_results = checks._prepare_for_json(results)
    
    report = {
        'timestamp': timestamp,
        'set': "01",
        'checks': "1-40",
        'description': "Geometric and metric tests",
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
    print("КРАЙ НА SET 01")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()