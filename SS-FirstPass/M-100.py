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
    # МЕТОДИ ЗА ЕНЕРГИЙНИЯ ТЕНЗОР (ЗА СЕТ 02)
    # ============================================================
    
    def PhiX(self, x, y, z, t=None):
        """∂Φ/∂x"""
        if t is None:
            t = self.t
        h = 1e-5
        return (self.Phi(x+h, y, z, t) - self.Phi(x-h, y, z, t)) / (2*h)
    
    def PhiY(self, x, y, z, t=None):
        """∂Φ/∂y"""
        if t is None:
            t = self.t
        h = 1e-5
        return (self.Phi(x, y+h, z, t) - self.Phi(x, y-h, z, t)) / (2*h)
    
    def PhiZ(self, x, y, z, t=None):
        """∂Φ/∂z"""
        if t is None:
            t = self.t
        h = 1e-5
        return (self.Phi(x, y, z+h, t) - self.Phi(x, y, z-h, t)) / (2*h)
    
    def PhiXX(self, x, y, z, t=None):
        """∂²Φ/∂x²"""
        if t is None:
            t = self.t
        h = 1e-4
        return (self.Phi(x+h, y, z, t) - 2*self.Phi(x, y, z, t) + self.Phi(x-h, y, z, t)) / (h**2)
    
    def LapPhi(self, x, y, z, t=None):
        """∇²Φ = ∂²Φ/∂x² + ∂²Φ/∂y² + ∂²Φ/∂z²"""
        if t is None:
            t = self.t
        h = 1e-4
        Phi = self.Phi(x, y, z, t)
        Phi_xp = self.Phi(x+h, y, z, t)
        Phi_xm = self.Phi(x-h, y, z, t)
        Phi_yp = self.Phi(x, y+h, z, t)
        Phi_ym = self.Phi(x, y-h, z, t)
        Phi_zp = self.Phi(x, y, z+h, t)
        Phi_zm = self.Phi(x, y, z-h, t)
        
        return (Phi_xp - 2*Phi + Phi_xm)/h**2 + \
               (Phi_yp - 2*Phi + Phi_ym)/h**2 + \
               (Phi_zp - 2*Phi + Phi_zm)/h**2
    
    def NormGradPhi(self, x, y, z, t=None):
        """|∇Φ|² = (∂Φ/∂x)² + (∂Φ/∂y)² + (∂Φ/∂z)²"""
        if t is None:
            t = self.t
        return self.PhiX(x, y, z, t)**2 + self.PhiY(x, y, z, t)**2 + self.PhiZ(x, y, z, t)**2
    
    def Gtt(self, x, y, z, t=None):
        """Einstein G_tt = e^{4Φ}(2∇²Φ - 3|∇Φ|²)"""
        if t is None:
            t = self.t
        Phi = self.Phi(x, y, z, t)
        return np.exp(4*Phi) * (2*self.LapPhi(x, y, z, t) - 3*self.NormGradPhi(x, y, z, t))
    
    def Gxx(self, x, y, z, t=None):
        """Einstein G_xx = -2∇²Φ + |∇Φ|² + 2(∂Φ/∂x)² - 2∂²Φ/∂x²"""
        if t is None:
            t = self.t
        return (-2*self.LapPhi(x, y, z, t) + 
                self.NormGradPhi(x, y, z, t) + 
                2*self.PhiX(x, y, z, t)**2 - 
                2*self.PhiXX(x, y, z, t))
    
    # ============================================================
    # BRANS-DICKE СЕКТОР (ЗА СЕТ 03)
    # ============================================================
    
    def phiBD(self, x, y, z):
        """Brans-Dicke скаларно поле: φ_BD = 1 + ε_BD * exp(-(r-R0)²/wBD²)"""
        r = self.r(x, y, z)
        return 1 + self.params['epsBD'] * np.exp(-(r - self.params['R0'])**2 / self.params['wBD']**2)
    
    def phiBD_derivatives(self, x, y, z):
        """Първи и втори производни на φ_BD"""
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
        """Brans-Dicke T_00 = (ω_BD/φ²)*(|∇φ|²/2) + (1/φ)*∇²φ"""
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        grad2 = phiX**2 + phiY**2 + phiZ**2
        return (omega / phi**2) * (grad2 / 2) + (1/phi) * lap_phi
    
    def NECBD(self, x, y, z):
        """Brans-Dicke NEC (x-посока) = (ω_BD/φ²)*(∂φ/∂x)² + (1/φ)*∂²φ/∂x²"""
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        return (omega / phi**2) * phiX**2 + (1/phi) * phiXX
    
    def T00Total(self, x, y, z, t=None):
        """Тотален T_00 = T_00^Geom + T_00^BD + T_00^BI + T_00^Shell"""
        if t is None:
            t = self.t
        # Опростена версия за сет 03
        return self.T00BD(x, y, z)
    
    # ============================================================
    # МЕТОДИ ЗА СЕТ 03 - BRANS-DICKE СПЕЦИФИЧНИ
    # ============================================================
    
    def phiBD_infinity(self):
        """Стойност на φ_BD в безкрайност"""
        return 1.0  # φ_BD → 1 при r → ∞
    
    def kinetic_term(self, x, y, z):
        """Кинетичен терм: (∂_μ Φ)(∂^μ Φ)"""
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
        
        # Използваме метриката за повдигане на индекса
        gtt = self.g_tt(x, y, z)
        gxx = self.g_xx(x, y, z)
        
        # (∂_μ Φ)(∂^μ Φ) = -g_tt * (∂_t Φ)^2 + g_xx * (∂_x Φ)^2 + ...
        # За t=0, ∂_t Φ = 0
        return gxx * (dphi_dx**2 + dphi_dy**2 + dphi_dz**2)
    
    def G_eff(self, x, y, z):
        """Ефективна гравитационна константа G_eff = (2ω_BD + 4)/(2ω_BD + 3) * 1/φ_BD"""
        phi = self.phiBD(x, y, z)
        omega = self.params['omegaBD']
        return (2*omega + 4) / (2*omega + 3) * (1/phi)
    
    def BD_wave_equation(self, x, y, z):
        """Brans-Dicke вълново уравнение: □φ = (1/(2ω_BD+3)) * T"""
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        
        # □φ = -∂_t²φ + ∇²φ (за t=0, ∂_t²φ = 0)
        dAlambert_phi = lap_phi
        
        # T = следа на стрес-енергиен тензор (опростено)
        T = self.T00BD(x, y, z)
        
        # Уравнение: □φ = (1/(2ω_BD+3)) * T
        rhs = (1/(2*omega + 3)) * T
        
        return dAlambert_phi, rhs
    
    def tachyonic_stability(self, x, y, z):
        """Tachyonic стабилност: провери дали ефективната маса е реална"""
        phi, phiX, phiY, phiZ, phiXX, phiYY, phiZZ, lap_phi = self.phiBD_derivatives(x, y, z)
        omega = self.params['omegaBD']
        
        # Ефективна маса: m_eff² = (1/(2ω_BD+3)) * d²V/dφ²
        # Опростено: m_eff² ≈ (1/(2ω_BD+3)) * lap_phi / phi
        if abs(phi) > 1e-10:
            m_eff_sq = (1/(2*omega + 3)) * (lap_phi / phi)
        else:
            m_eff_sq = 0.0
        
        return m_eff_sq


# ============================================================
# ПРОВЕРКИ 81-100: BRANS-DICKE СКАЛАРЕН СЕКТОР
# ============================================================

class BransDickeChecks:
    """Проверки 81-100: Brans-Dicke скаларен сектор"""
    
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
    
    def check_81_84_phi_infinity(self):
        """Проверки 81-84: Стойност на φ_BD в безкрайност (трябва да клони към 1)"""
        test_points = [
            (10, 0, 0, "далечна ос x"),
            (0, 10, 0, "далечна ос y"),
            (0, 0, 10, "далечна ос z"),
            (10, 10, 10, "далечен ъгъл")
        ]
        
        expected = 1.0  # φ_BD → 1 при r → ∞
        
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
        """Проверки 85-88: Кинетичен терм (∂_μ Φ)(∂^μ Φ) в 4 точки около балона"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 85):
            kinetic = self.metric.kinetic_term(x, y, z)
            
            # Проверяваме дали кинетичният терм е реален и краен
            passed = np.isfinite(kinetic) and not np.isnan(kinetic)
            
            self.results.append({
                'check_id': i,
                'name': f"Kinetic term at {name} ({x},{y},{z})",
                'passed': passed,
                'value': kinetic,
                'expected': "finite and real",
                'tolerance': 1e-6
            })
    
    def check_89_92_G_eff(self):
        """Проверки 89-92: Влияние на ω_BD върху ефективната гравитационна константа G_eff"""
        # Тестваме различни стойности на omegaBD
        omega_values = [5, 10, 20, 50]
        x, y, z = 2.0, 0.0, 0.0  # тестова точка на стената
        
        for i, omega in enumerate(omega_values, 89):
            # Запазваме старата стойност
            old_omega = self.metric.params['omegaBD']
            self.metric.params['omegaBD'] = omega
            
            G_eff = self.metric.G_eff(x, y, z)
            
            # Възстановяваме
            self.metric.params['omegaBD'] = old_omega
            
            # Проверяваме дали G_eff е крайна и положителна
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
        """Проверки 93-96: Brans-Dicke уравнения на движение (вълново уравнение)"""
        test_points = [
            (0, 0, 0, "център"),
            (1.5, 0, 0, "вътрешен ръб"),
            (2.0, 0, 0, "стена"),
            (2.5, 0, 0, "външен ръб")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 93):
            dAlambert_phi, rhs = self.metric.BD_wave_equation(x, y, z)
            
            # Проверяваме дали уравнението е изпълнено: □φ = (1/(2ω+3)) * T
            # С толеранс заради числени грешки
            diff = abs(dAlambert_phi - rhs)
            passed = diff < 1e-4
            
            self.results.append({
                'check_id': i,
                'name': f"BD wave equation at {name} ({x},{y},{z})",
                'passed': passed,
                'value': {"□φ": dAlambert_phi, "RHS": rhs, "diff": diff},
                'expected': "□φ ≈ RHS",
                'tolerance': 1e-4
            })
    
    def check_97_100_tachyonic_stability(self):
        """Проверки 97-100: Tachyonic стабилност на BD полето в стената на балона"""
        test_points = [
            (1.8, 0, 0, "вътрешна стена"),
            (2.0, 0, 0, "стена"),
            (2.2, 0, 0, "външна стена"),
            (0, 1.8, 1.8, "оф-ос стена")
        ]
        
        for i, (x, y, z, name) in enumerate(test_points, 97):
            m_eff_sq = self.metric.tachyonic_stability(x, y, z)
            
            # За стабилност, m_eff² ≥ 0 (без tachyonic режими)
            passed = m_eff_sq >= -1e-6
            
            self.results.append({
                'check_id': i,
                'name': f"Tachyonic stability at {name} ({x},{y},{z})",
                'passed': passed,
                'value': m_eff_sq,
                'expected': "≥ 0",
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
    print("WARP BUBBLE CHECKS - SET 03 (Проверки 81-100)")
    print("Brans-Dicke скаларен сектор")
    print("="*70)
    print()
    
    # Инициализация
    metric = WarpMetric()
    checks = BransDickeChecks(metric)
    
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
    filename = f"checks_set03_{timestamp}.json"
    
    # Подготви резултатите за JSON
    prepared_results = checks._prepare_for_json(results)
    
    report = {
        'timestamp': timestamp,
        'set': "03",
        'checks': "81-100",
        'description': "Brans-Dicke scalar sector",
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
    print("КРАЙ НА SET 03")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()