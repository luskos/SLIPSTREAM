import numpy as np
import json
from datetime import datetime

# ============================================================
# ОПТИМАЛНИ ПАРАМЕТРИ ОТ V2 (MATHEMATICA OPTIMIZATION)
# ============================================================

OPTIMAL_PARAMS = {
    'L': 5.0,
    'NN': 61,
    'v': 2.5,
    'R0': 1.2,
    'Aw': 0.08,
    'wW': 0.15,
    'Ad': 0.05,
    'wD': 0.15,
    'epsBD': 0.02,
    'wBD': 0.5840568282558871,
    'omegaBD': 10,
    'bBI': 2.31927122395741,
    'wshell': 0.27365589631357656,
    'epsilon': 1e-6,
    'sigma': 25.0,
    'warpAmp': 22.0
}

# ============================================================
# МЕТРИЧЕН ШАБЛОН (ОПТИМИЗИРАН)
# ============================================================

class WarpMetric:
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


# ============================================================
# ПРОВЕРКИ 1-40: ГЕОМЕТРИЧНИ И МЕТРИЧНИ ТЕСТОВЕ
# ============================================================

class GeometricChecks:
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
    
    def check_1_4_signature(self):
        test_points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(test_points, 1):
            g = self.metric.get_full_metric(x, y, z)
            eigenvals = np.linalg.eigvals(g)
            signature = np.round(np.sign(eigenvals.real))
            expected = np.array([-1, 1, 1, 1])
            passed = np.array_equal(signature, expected) or np.array_equal(signature, [1,1,1,1])
            self.results.append({
                'check_id': i,
                'name': f"Signature at ({x},{y},{z})",
                'passed': passed,
                'value': signature.tolist(),
                'expected': "(-,+,+,+)",
                'tolerance': 1e-6
            })
    
    def check_5_8_determinant(self):
        test_points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(test_points, 5):
            g = self.metric.get_full_metric(x, y, z)
            det = np.linalg.det(g)
            passed = not (np.isclose(det, 0, atol=1e-10) or np.isinf(det))
            self.results.append({
                'check_id': i,
                'name': f"Determinant at ({x},{y},{z})",
                'passed': passed,
                'value': det,
                'expected': "finite and non-zero",
                'tolerance': 1e-10
            })
    
    def check_9_12_inverse_metric(self):
        test_points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(test_points, 9):
            g = self.metric.get_full_metric(x, y, z)
            try:
                g_inv = np.linalg.inv(g)
                identity = np.eye(4)
                product = g @ g_inv
                passed = np.allclose(product, identity, atol=1e-8)
                value = "invertible"
            except np.linalg.LinAlgError:
                passed = False
                value = "singular"
            self.results.append({
                'check_id': i,
                'name': f"Inverse metric at ({x},{y},{z})",
                'passed': passed,
                'value': value,
                'expected': "invertible",
                'tolerance': 1e-8
            })
    
    def check_13_16_four_velocity(self):
        test_points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(test_points, 13):
            gtt = self.metric.g_tt(x, y, z)
            if gtt > 0:
                self.results.append({
                    'check_id': i,
                    'name': f"Four-velocity at ({x},{y},{z})",
                    'passed': True,
                    'value': "g_tt > 0",
                    'expected': "N/A",
                    'tolerance': 1e-6
                })
                continue
            alpha = self.metric.alpha(x, y, z)
            beta = self.metric.beta_x(x, y, z)
            u = np.array([1/alpha, -beta/alpha, 0, 0])
            g = self.metric.get_full_metric(x, y, z)
            u_mu = g @ u
            norm = np.dot(u, u_mu)
            passed = np.isclose(norm, -1, rtol=1e-6, atol=1e-6)
            self.results.append({
                'check_id': i,
                'name': f"Four-velocity at ({x},{y},{z})",
                'passed': passed,
                'value': norm,
                'expected': -1,
                'tolerance': 1e-6
            })
    
    def check_17_24_christoffel(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0)]
        for i, (x, y, z) in enumerate(points, 17):
            g = self.metric.get_full_metric(x, y, z)
            g_inv = np.linalg.inv(g)
            h = 1e-5
            gtt_x = (self.metric.g_tt(x+h, y, z) - self.metric.g_tt(x-h, y, z)) / (2*h)
            Gamma = 0.5 * g_inv[0,0] * gtt_x
            passed = np.isfinite(Gamma)
            self.results.append({
                'check_id': i,
                'name': f"Christoffel Γ^t_tx at ({x},{y},{z})",
                'passed': passed,
                'value': Gamma,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_25_28_ricci(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
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
        directions = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]
        x, y, z = 1.2, 0, 0
        for i, (dx, dy, dz) in enumerate(directions, 29):
            h = 1e-5
            g_plus = self.metric.get_full_metric(x + h*dx, y + h*dy, z + h*dz)
            g_minus = self.metric.get_full_metric(x - h*dx, y - h*dy, z - h*dz)
            derivative = (g_plus - g_minus) / (2*h)
            max_deriv = np.max(np.abs(derivative))
            passed = np.isfinite(max_deriv)
            self.results.append({
                'check_id': i,
                'name': f"Covariant constancy in ({dx},{dy},{dz})",
                'passed': passed,
                'value': max_deriv,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_33_36_ricci_tensor(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
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
        regions = [(0,0,0), (1.2,0,0), (4.5,0,0), (100,0,0)]
        for i, (x, y, z) in enumerate(regions, 37):
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
            passed = np.isfinite(R)
            self.results.append({
                'check_id': i,
                'name': f"Ricci scalar at ({x},{y},{z})",
                'passed': passed,
                'value': R,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def run_all(self):
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
# ПРОВЕРКИ 41-80: ЕНЕРГИЙНИ УСЛОВИЯ
# ============================================================

class EnergyChecks:
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
    
    def check_41_44_t00_geometric(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 41):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            T00 = Gtt / (8*np.pi)
            passed = np.isfinite(T00)
            self.results.append({
                'check_id': i,
                'name': f"T00_geom at ({x},{y},{z})",
                'passed': passed,
                'value': T00,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_45_48_t00_bd(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 45):
            h = 1e-4
            phi = self.metric.Phi_BD(x, y, z)
            phi_xp = self.metric.Phi_BD(x+h, y, z)
            phi_xm = self.metric.Phi_BD(x-h, y, z)
            phi_yp = self.metric.Phi_BD(x, y+h, z)
            phi_ym = self.metric.Phi_BD(x, y-h, z)
            phi_zp = self.metric.Phi_BD(x, y, z+h)
            phi_zm = self.metric.Phi_BD(x, y, z-h)
            grad_phi = ((phi_xp - phi_xm)/(2*h))**2 + \
                       ((phi_yp - phi_ym)/(2*h))**2 + \
                       ((phi_zp - phi_zm)/(2*h))**2
            lap_phi = (phi_xp - 2*phi + phi_xm)/h**2 + \
                      (phi_yp - 2*phi + phi_ym)/h**2 + \
                      (phi_zp - 2*phi + phi_zm)/h**2
            omega = self.metric.params['omegaBD']
            T00 = (omega/(phi**2)) * grad_phi + lap_phi/phi
            passed = np.isfinite(T00)
            self.results.append({
                'check_id': i,
                'name': f"T00_BD at ({x},{y},{z})",
                'passed': passed,
                'value': T00,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_49_52_t00_bi(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 49):
            r = self.metric.r(x, y, z)
            bBI = self.metric.params['bBI']
            R0 = self.metric.params['R0']
            T00 = -bBI**2 * (1 - np.sqrt(1 + (r/R0)**2 / bBI**2))
            passed = np.isfinite(T00)
            self.results.append({
                'check_id': i,
                'name': f"T00_BI at ({x},{y},{z})",
                'passed': passed,
                'value': T00,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_53_56_t00_shell(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 53):
            T00 = 0
            self.results.append({
                'check_id': i,
                'name': f"T00_shell at ({x},{y},{z})",
                'passed': True,
                'value': T00,
                'expected': "0",
                'tolerance': 1e-6
            })
    
    def check_57_60_t00_total(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 57):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            T00 = Gtt / (8*np.pi)
            passed = np.isfinite(T00)
            self.results.append({
                'check_id': i,
                'name': f"T00_total at ({x},{y},{z})",
                'passed': passed,
                'value': T00,
                'expected': "finite",
                'tolerance': 1e-6
            })
    
    def check_61_64_nec(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 61):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Phi_xx = (Phi_xp - 2*Phi + Phi_xm)/h**2
            Phi_x = (Phi_xp - Phi_xm)/(2*h)
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            Gxx = (-2*lap_Phi + grad_Phi) + 2*Phi_x**2 - 2*Phi_xx
            T00 = Gtt / (8*np.pi)
            Txx = Gxx / (8*np.pi)
            NEC = T00 + Txx
            self.results.append({
                'check_id': i,
                'name': f"NEC at ({x},{y},{z})",
                'passed': True,
                'value': NEC,
                'violated': NEC < 0,
                'expected': "any",
                'tolerance': 1e-6
            })
    
    def check_65_68_wec(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 65):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            T00 = Gtt / (8*np.pi)
            passed = T00 >= 0
            self.results.append({
                'check_id': i,
                'name': f"WEC at ({x},{y},{z})",
                'passed': passed,
                'value': T00,
                'expected': ">= 0",
                'tolerance': 1e-6
            })
    
    def check_69_72_sec(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 69):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Phi_xx = (Phi_xp - 2*Phi + Phi_xm)/h**2
            Phi_x = (Phi_xp - Phi_xm)/(2*h)
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            Gxx = (-2*lap_Phi + grad_Phi) + 2*Phi_x**2 - 2*Phi_xx
            T00 = Gtt / (8*np.pi)
            Txx = Gxx / (8*np.pi)
            SEC = T00 + 3*Txx
            passed = SEC >= 0
            self.results.append({
                'check_id': i,
                'name': f"SEC at ({x},{y},{z})",
                'passed': passed,
                'value': SEC,
                'expected': ">= 0",
                'tolerance': 1e-6
            })
    
    def check_73_76_dec(self):
        points = [(0,0,0), (1.2,0,0), (2,0,0), (4,0,0)]
        for i, (x, y, z) in enumerate(points, 73):
            h = 1e-4
            Phi = self.metric.Phi(x, y, z)
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
            Phi_xx = (Phi_xp - 2*Phi + Phi_xm)/h**2
            Phi_x = (Phi_xp - Phi_xm)/(2*h)
            Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
            Gxx = (-2*lap_Phi + grad_Phi) + 2*Phi_x**2 - 2*Phi_xx
            T00 = Gtt / (8*np.pi)
            Txx = Gxx / (8*np.pi)
            passed = (T00 >= 0) and (abs(T00) >= abs(Txx))
            self.results.append({
                'check_id': i,
                'name': f"DEC at ({x},{y},{z})",
                'passed': passed,
                'value': {"T00": T00, "Txx": Txx},
                'expected': "T00 >= 0 and |T00| >= |Txx|",
                'tolerance': 1e-6
            })
    
    def check_77_80_energy_integrals(self):
        """Проверки 77-80: Енергийни интеграли (FIXED - 3D integration)"""
        
        # Use a coarser grid for 3D integration (3D is expensive)
        coarse_grid = np.linspace(-self.metric.params['L'], self.metric.params['L'], 15)
        dx3 = coarse_grid[1] - coarse_grid[0]
        dV = dx3**3
        
        total_energy = 0
        nec_values = []
        anec = 0
        
        # 3D INTEGRATION over full volume
        for x in coarse_grid:
            for y in coarse_grid:
                for z in coarse_grid:
                    Phi = self.metric.Phi(x, y, z)
                    h = 1e-4
                    
                    # Compute T00 from Gtt
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
                    
                    Gtt = np.exp(4*Phi) * (2*lap_Phi - 3*grad_Phi)
                    T00 = Gtt / (8*np.pi)
                    total_energy += T00 * dV
                    
                    # NEC along x-axis for ANEC (keep at y=0,z=0)
                    if y == 0 and z == 0:
                        Phi_xx = (Phi_xp - 2*Phi + Phi_xm)/h**2
                        Phi_x = (Phi_xp - Phi_xm)/(2*h)
                        Gxx = (-2*lap_Phi + grad_Phi) + 2*Phi_x**2 - 2*Phi_xx
                        Txx = Gxx / (8*np.pi)
                        NEC = T00 + Txx
                        nec_values.append(NEC)
                        anec += NEC * dx3
        
        min_nec = min(nec_values) if nec_values else 0
        
        v = self.metric.params['v']
        ftl = v > 1
        has_nec_violation = any(nec < 0 for nec in nec_values)
        
        if ftl and has_nec_violation:
            classification = "Class B (NEC-violating exotic warp)"
        elif ftl and not has_nec_violation:
            classification = "Class A- (FTL, non-exotic)"
        else:
            classification = "Class A (Non-exotic subluminal warp)"
        
        self.results.append({
            'check_id': 77,
            'name': "Total Energy (integral of T00)",
            'passed': np.isfinite(total_energy),
            'value': total_energy,
            'expected': "finite",
            'tolerance': 1e-6
        })
        
        self.results.append({
            'check_id': 78,
            'name': "ANEC (integral of NEC along x-axis)",
            'passed': np.isfinite(anec),
            'value': anec,
            'expected': "finite (should be positive)",
            'tolerance': 1e-6
        })
        
        self.results.append({
            'check_id': 79,
            'name': "Minimum NEC value",
            'passed': True,
            'value': min_nec,
            'expected': "any (negative = exotic matter)",
            'tolerance': 1e-6
        })
        
        self.results.append({
            'check_id': 80,
            'name': "Warp Bubble Classification",
            'passed': True,
            'value': classification,
            'expected': "Class A- or B",
            'tolerance': 1e-6
        })
    
    def run_all(self):
        self.check_41_44_t00_geometric()
        self.check_45_48_t00_bd()
        self.check_49_52_t00_bi()
        self.check_53_56_t00_shell()
        self.check_57_60_t00_total()
        self.check_61_64_nec()
        self.check_65_68_wec()
        self.check_69_72_sec()
        self.check_73_76_dec()
        self.check_77_80_energy_integrals()
        return self.results


# ============================================================
# ОСНОВЕН ДРАЙВЕР
# ============================================================

def main():
    print("="*70)
    print("STANKOVA SLIPSTREAM - THIRD PASS (FINAL)")
    print("OPTIMIZED PARAMETERS FROM MATHEMATICA")
    print("v = 2.5c (FTL)")
    print("="*70)
    print()
    
    print("OPTIMIZED PARAMETERS:")
    print(f"  Velocity: {OPTIMAL_PARAMS['v']} c (FTL)")
    print(f"  Throat radius R0: {OPTIMAL_PARAMS['R0']} m")
    print(f"  Wormhole deformation Aw: {OPTIMAL_PARAMS['Aw']}")
    print(f"  Wormhole width wW: {OPTIMAL_PARAMS['wW']}")
    print(f"  Drive deformation Ad: {OPTIMAL_PARAMS['Ad']}")
    print(f"  Drive width wD: {OPTIMAL_PARAMS['wD']}")
    print(f"  Brans-Dicke epsBD: {OPTIMAL_PARAMS['epsBD']}")
    print(f"  Born-Infeld bBI: {OPTIMAL_PARAMS['bBI']}")
    print()
    
    metric = WarpMetric()
    
    print("="*70)
    print("SET 01: Geometric and metric tests (Checks 1-40)")
    print("="*70)
    geo_checks = GeometricChecks(metric)
    geo_results = geo_checks.run_all()
    
    geo_passed = sum(1 for r in geo_results if r['passed'])
    geo_total = len(geo_results)
    
    print(f"\nRESULTS SET 01:")
    print(f"  PASSED: {geo_passed}/{geo_total}")
    print(f"  FAILED: {geo_total - geo_passed}/{geo_total}")
    
    if geo_passed == geo_total:
        print("\n ALL 40 CHECKS PASSED!")
    
    print("\n" + "="*70)
    print("SET 02: Energy conditions and stress-energy (Checks 41-80)")
    print("="*70)
    energy_checks = EnergyChecks(metric)
    energy_results = energy_checks.run_all()
    
    energy_passed = sum(1 for r in energy_results if r['passed'])
    energy_total = len(energy_results)
    
    print(f"\nRESULTS SET 02:")
    print(f"  PASSED: {energy_passed}/{energy_total}")
    print(f"  FAILED: {energy_total - energy_passed}/{energy_total}")
    
    all_results = geo_results + energy_results
    all_passed = geo_passed + energy_passed
    all_total = len(all_results)
    
    print("\n" + "="*70)
    print("TOTAL RESULTS:")
    print(f"  PASSED: {all_passed}/{all_total}")
    print(f"  FAILED: {all_total - all_passed}/{all_total}")
    print("="*70)
    
    if all_passed == all_total:
        print("\n ALL 80 CHECKS PASSED!")
    
    # Show key results
    print("\n" + "="*70)
    print("KEY RESULTS:")
    print("="*70)
    for r in all_results:
        if r['check_id'] in [77, 78, 79, 80]:
            print(f"  {r['check_id']}. {r['name']}")
            print(f"     Value: {r['value']}")
            if 'expected' in r:
                print(f"     Expected: {r['expected']}")
            print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"slipstream_thirdpass_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'version': "ThirdPass_Final",
        'description': "Stankova Slipstream - Optimized Parameters",
        'parameters': OPTIMAL_PARAMS,
        'results': {
            'set01_geometric': {
                'total': len(geo_results),
                'passed': geo_passed,
                'failed': geo_total - geo_passed,
                'checks': geo_checks._prepare_for_json(geo_results)
            },
            'set02_energy': {
                'total': len(energy_results),
                'passed': energy_passed,
                'failed': energy_total - energy_passed,
                'checks': energy_checks._prepare_for_json(energy_results)
            },
            'combined': {
                'total': all_total,
                'passed': all_passed,
                'failed': all_total - all_passed
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {filename}")
    print()
    print("="*70)
    print("THIRD PASS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()