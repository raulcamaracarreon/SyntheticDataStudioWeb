import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# SciPy para ppf/cdf
try:
    from scipy.stats import (
        norm, truncnorm, lognorm, gamma as gamma_dist,
        beta as beta_dist, t as t_dist
    )
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ----------------- Esquema / Especificaciones -----------------

@dataclass
class ContinuousSpec:
    name: str
    dist: str  # 'normal' | 'uniform' | 'truncnorm' | 'lognormal' | 'gamma' | 'exponential' | 'weibull' | 'beta' | 'triangular' | 'student_t' | 'chisquare' | 'pareto' | 'poisson' | 'binomial'
    params: Dict[str, float]  # incluye opciones extra (no_negative, enforce_minmax, min_clip, max_clip, decimals, missing_pct, outlier_pct, outlier_mult, transform, keep_outliers)

@dataclass
class CategoricalSpec:
    name: str
    categories: List[str]
    probs: List[float]  # se normaliza si no suma 1

@dataclass
class VariableSpec:
    kind: str  # 'continuous' | 'categorical'
    continuous: Optional[ContinuousSpec] = None
    categorical: Optional[CategoricalSpec] = None

@dataclass
class Schema:
    n_rows: int = 1000
    seed: Optional[int] = 42
    variables: List[VariableSpec] = field(default_factory=list)
    # Correlaciones (solo continuas)
    use_copula: bool = False
    corr_matrix: Optional[List[List[float]]] = None  # tamaño kxk (k = #continuas, en orden)


# ----------------- Utilidades de correlación -----------------

def _nearest_psd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Proyecta A a la PSD más cercana (clip de eigenvalores)."""
    B = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(B)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T

def _sample_gaussian_copula(n: int, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Devuelve Z ~ N(0, corr) de tamaño (n, k)."""
    corr = _nearest_psd(corr)
    k = corr.shape[0]
    jitter = 1e-12
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(corr + jitter * np.eye(k))
    z = rng.standard_normal(size=(n, k)) @ L.T
    return z

def _u_from_z(z: np.ndarray) -> np.ndarray:
    """Mapea Z normal estándar a U(0,1) por la CDF normal."""
    if HAS_SCIPY:
        return norm.cdf(z)
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


# ----------------- PPF por distribución -----------------

def _ppf_uniform(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * u

def _ppf_normal(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if HAS_SCIPY:
        return norm.ppf(u, loc=mu, scale=max(sigma, 1e-12))
    x = np.sqrt(2) * np.erfinv(2*u - 1)
    return mu + sigma * x

def _ppf_truncnorm(u: np.ndarray, mu: float, sigma: float, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    if HAS_SCIPY:
        a = (low - mu) / sigma
        b = (high - mu) / sigma
        return truncnorm.ppf(u, a=a, b=b, loc=mu, scale=sigma)
    # fallback simple
    return np.clip(mu + sigma * rng.standard_normal(size=u.shape[0]), low, high)

def _ppf_lognormal(u: np.ndarray, mu_log: float, sigma_log: float) -> np.ndarray:
    if HAS_SCIPY:
        return lognorm.ppf(u, s=max(sigma_log, 1e-12), scale=np.exp(mu_log))
    x = np.sqrt(2) * np.erfinv(2*u - 1)
    return np.exp(mu_log + sigma_log * x)

def _ppf_gamma(u: np.ndarray, k_shape: float, theta_scale: float) -> np.ndarray:
    if HAS_SCIPY:
        return gamma_dist.ppf(u, a=max(k_shape, 1e-12), scale=max(theta_scale, 1e-12))
    # fallback tosco
    if abs(k_shape - round(k_shape)) < 1e-9 and k_shape >= 1:
        k = int(round(k_shape))
        r = -np.log(1 - u)
        for _ in range(k - 1):
            r += -np.log(1 - u)
        return theta_scale * r
    return theta_scale * (-np.log(1 - u))

def _ppf_exponential(u: np.ndarray, rate: float) -> np.ndarray:
    rate = max(rate, 1e-12)
    return -np.log(1.0 - u) / rate

def _ppf_weibull(u: np.ndarray, k_shape: float, lam_scale: float) -> np.ndarray:
    k_shape = max(k_shape, 1e-12)
    lam_scale = max(lam_scale, 1e-12)
    return lam_scale * (-np.log(1.0 - u))**(1.0 / k_shape)

def _ppf_beta(u: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if not HAS_SCIPY:
        raise RuntimeError("Beta con cópula requiere SciPy para la PPF.")
    alpha = max(alpha, 1e-12); beta = max(beta, 1e-12)
    return beta_dist.ppf(u, a=alpha, b=beta)

def _ppf_student_t(u: np.ndarray, df: float) -> np.ndarray:
    if not HAS_SCIPY:
        raise RuntimeError("Student-t con cópula requiere SciPy para la PPF.")
    df = max(df, 1e-6)
    return t_dist.ppf(u, df)

def _ppf_chisquare(u: np.ndarray, df: float) -> np.ndarray:
    # Chi2(df) = Gamma(k=df/2, theta=2)
    df = max(df, 1e-6)
    return _ppf_gamma(u, k_shape=df/2.0, theta_scale=2.0)

def _ppf_pareto(u: np.ndarray, alpha: float, xm: float) -> np.ndarray:
    alpha = max(alpha, 1e-12); xm = max(xm, 1e-12)
    # F^{-1}(u) = xm * (1 - u)^(-1/alpha)
    return xm * (1.0 - u)**(-1.0 / alpha)

def _ppf_triangular(u: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
    # Inversa de triangular piecewise
    if high < low:
        low, high = high, low
    mode = min(max(mode, low), high)
    c = (mode - low) / (high - low) if high > low else 0.5
    x = np.empty_like(u)
    left = (u <= c)
    x[left] = low + np.sqrt(u[left] * (high - low) * (mode - low))
    x[~left] = high - np.sqrt((1.0 - u[~left]) * (high - low) * (high - mode))
    return x


# ----------------- Post-procesado (faltantes, outliers, límites, transform, decimales) -----------------

def _apply_postproc(x: np.ndarray, p: Dict, rng: np.random.Generator) -> np.ndarray:
    """
    Orden:
    0) Outliers (antes de límites/transform)
    1) No negativos
    2) Clip por min/max
    3) Transform: 'ceil' | 'floor' | 'log1p'
    4) Decimales
    5) Faltantes
    """
    x = x.astype(float, copy=False)

    # 0) Outliers
    outlier_pct = float(p.get("outlier_pct", 0.0) or 0.0)
    outlier_mult = float(p.get("outlier_mult", 0.0) or 0.0)
    keep_outliers = bool(p.get("keep_outliers", True))
    out_idx = np.array([], dtype=int)

    if outlier_pct > 0 and outlier_mult != 0:
        n = x.shape[0]
        m = max(1, int(round(outlier_pct * n)))
        med = np.nanmedian(x)
        dev = np.abs(x - med)
        if m >= n:
            out_idx = np.arange(n, dtype=int)
        else:
            out_idx = np.argpartition(dev, -m)[-m:]
        x[out_idx] = med + (x[out_idx] - med) * outlier_mult

    # 1) No negativos
    if p.get("no_negative"):
        if p.get("enforce_minmax"):
            if "min_clip" in p:
                try:
                    p["min_clip"] = max(0.0, float(p["min_clip"]))
                except Exception:
                    p["min_clip"] = 0.0
            else:
                p["min_clip"] = 0.0
        else:
            x = np.clip(x, 0.0, None)

    # 2) Clip por min/max
    if p.get("enforce_minmax"):
        lo = p.get("min_clip", None)
        hi = p.get("max_clip", None)
        lo = float(lo) if lo is not None and lo != "" else None
        hi = float(hi) if hi is not None and hi != "" else None
        if (lo is not None) and (hi is not None) and (hi < lo):
            lo, hi = hi, lo

        if keep_outliers and out_idx.size > 0:
            mask = np.ones_like(x, dtype=bool)
            mask[out_idx] = False
            if lo is not None:
                x[mask] = np.maximum(x[mask], lo)
            if hi is not None:
                x[mask] = np.minimum(x[mask], hi)
        else:
            lo_eff = lo if lo is not None else -np.inf
            hi_eff = hi if hi is not None else  np.inf
            x = np.clip(x, lo_eff, hi_eff)

    # 3) Transform
    t = (p.get("transform") or "").lower()
    if t == "ceil":
        x = np.ceil(x)
    elif t == "floor":
        x = np.floor(x)
    elif t == "log1p":
        xmin = np.nanmin(x)
        if xmin < -1.0:
            x = x - (xmin + 1.0)
        x = np.log1p(np.clip(x, -1.0, None))

    # 4) Decimales
    d = p.get("decimals", None)
    if d is not None and d != "":
        try:
            d = int(d)
            d = max(0, min(7, d))
            x = np.round(x, d)
        except Exception:
            pass

    # 5) Faltantes
    miss = float(p.get("missing_pct", 0.0) or 0.0)
    if miss > 0:
        n = x.shape[0]
        m = max(1, int(round(miss * n)))
        if m >= n:
            x[:] = np.nan
        else:
            idx_nan = rng.choice(n, size=m, replace=False)
            x[idx_nan] = np.nan

    return x


# ----------------- Generación principal -----------------

def generate_dataset(schema: Schema) -> pd.DataFrame:
    rng = np.random.default_rng(schema.seed)
    data: Dict[str, np.ndarray] = {}

    # Separa continuas/categóricas manteniendo orden
    cont_specs: List[ContinuousSpec] = []
    cat_specs: List[CategoricalSpec] = []
    for v in schema.variables:
        if v.kind == "continuous" and v.continuous:
            cont_specs.append(v.continuous)
        elif v.kind == "categorical" and v.categorical:
            cat_specs.append(v.categorical)

    # --- Continuas ---
    if cont_specs:
        k = len(cont_specs)
        if schema.use_copula and k >= 2:
            if not HAS_SCIPY:
                raise RuntimeError("Las correlaciones requieren SciPy. Instala con: pip install scipy")
            # Matriz R
            if schema.corr_matrix is None:
                R = np.eye(k)
            else:
                R = np.array(schema.corr_matrix, dtype=float)
                if R.shape != (k, k):
                    raise ValueError("La matriz de correlación no coincide con el número de variables continuas.")

            Z = _sample_gaussian_copula(schema.n_rows, R, rng)
            U = _u_from_z(Z)

            for j, spec in enumerate(cont_specs):
                dist = (spec.dist or "").lower()
                p = dict(spec.params)  # copia (se modifica en postproc)

                if dist == "uniform":
                    x = _ppf_uniform(U[:, j], float(p.get("low", 0.0)), float(p.get("high", 1.0)))

                elif dist == "normal":
                    x = _ppf_normal(U[:, j], float(p.get("mu", 0.0)), float(p.get("sigma", 1.0)))

                elif dist == "truncnorm":
                    x = _ppf_truncnorm(
                        U[:, j],
                        float(p.get("mu", 0.0)),
                        float(p.get("sigma", 1.0)),
                        float(p.get("low", -np.inf)),
                        float(p.get("high",  np.inf)),
                        rng
                    )

                elif dist == "lognormal":
                    x = _ppf_lognormal(U[:, j], float(p.get("mu_log", 0.0)), float(p.get("sigma_log", 1.0)))

                elif dist == "gamma":
                    x = _ppf_gamma(U[:, j], float(p.get("shape_k", 1.0)), float(p.get("scale_theta", 1.0)))

                elif dist == "exponential":
                    x = _ppf_exponential(U[:, j], float(p.get("rate", 1.0)))

                elif dist == "weibull":
                    x = _ppf_weibull(U[:, j], float(p.get("shape_k", 1.5)), float(p.get("scale_lambda", 1.0)))

                elif dist == "beta":
                    x = _ppf_beta(U[:, j], float(p.get("alpha", 2.0)), float(p.get("beta", 2.0)))

                elif dist == "triangular":
                    x = _ppf_triangular(U[:, j], float(p.get("low", 0.0)), float(p.get("mode", 0.5)), float(p.get("high", 1.0)))

                elif dist == "student_t":
                    x = _ppf_student_t(U[:, j], float(p.get("df", 5.0)))

                elif dist == "chisquare":
                    x = _ppf_chisquare(U[:, j], float(p.get("df", 4.0)))

                elif dist == "pareto":
                    x = _ppf_pareto(U[:, j], float(p.get("alpha", 3.0)), float(p.get("xm", 1.0)))

                # Discretas no compatibles con cópula vía PPF simple; se recomienda usarlas sin cópula
                elif dist in {"poisson", "binomial"}:
                    raise ValueError(f"La distribución '{dist}' no se soporta con cópula. Desactiva correlaciones o usa una continua equivalente.")

                else:
                    raise ValueError(f"Distribución no soportada con cópula: {dist}")

                x = _apply_postproc(x, p, rng)
                data[spec.name] = x

        else:
            # Independientes
            for spec in cont_specs:
                dist = (spec.dist or "").lower()
                p = dict(spec.params)

                if dist == "uniform":
                    low = float(p.get("low", 0.0))
                    high = float(p.get("high", 1.0))
                    if high <= low:
                        high = low + 1.0
                    x = rng.uniform(low, high, schema.n_rows)

                elif dist == "normal":
                    mu = float(p.get("mu", 0.0))
                    sigma = float(p.get("sigma", 1.0))
                    x = rng.normal(mu, sigma, schema.n_rows)

                elif dist == "truncnorm":
                    mu = float(p.get("mu", 0.0))
                    sigma = float(p.get("sigma", 1.0))
                    low = float(p.get("low", mu - 3*sigma))
                    high = float(p.get("high", mu + 3*sigma))
                    if high <= low:
                        high = low + 1e-6
                    U = rng.random(schema.n_rows)
                    x = _ppf_truncnorm(U, mu, sigma, low, high, rng)

                elif dist == "lognormal":
                    mu_log = float(p.get("mu_log", 0.0))
                    sigma_log = float(p.get("sigma_log", 1.0))
                    U = rng.random(schema.n_rows)
                    x = _ppf_lognormal(U, mu_log, sigma_log)

                elif dist == "gamma":
                    k_shape = float(p.get("shape_k", 1.0))
                    theta = float(p.get("scale_theta", 1.0))
                    U = rng.random(schema.n_rows)
                    x = _ppf_gamma(U, k_shape, theta)

                elif dist == "exponential":
                    rate = float(p.get("rate", 1.0))
                    x = rng.exponential(scale=1.0 / max(rate, 1e-12), size=schema.n_rows)

                elif dist == "weibull":
                    k_shape = float(p.get("shape_k", 1.5))
                    lam = float(p.get("scale_lambda", 1.0))
                    x = lam * rng.weibull(k_shape, schema.n_rows)

                elif dist == "beta":
                    a = float(p.get("alpha", 2.0))
                    b = float(p.get("beta", 2.0))
                    a = max(a, 1e-12); b = max(b, 1e-12)
                    x = rng.beta(a, b, schema.n_rows)

                elif dist == "triangular":
                    low = float(p.get("low", 0.0))
                    high = float(p.get("high", 1.0))
                    mode = float(p.get("mode", (low + high) / 2.0))
                    if high < low:
                        low, high = high, low
                    mode = min(max(mode, low), high)
                    x = rng.triangular(low, mode, high, schema.n_rows)

                elif dist == "student_t":
                    df = float(p.get("df", 5.0))
                    df = max(df, 1e-6)
                    x = rng.standard_t(df, size=schema.n_rows)

                elif dist == "chisquare":
                    df = float(p.get("df", 4.0))
                    df = max(df, 1e-6)
                    x = rng.chisquare(df, size=schema.n_rows)

                elif dist == "pareto":
                    alpha = float(p.get("alpha", 3.0))
                    xm = float(p.get("xm", 1.0))
                    alpha = max(alpha, 1e-12); xm = max(xm, 1e-12)
                    x = xm * (1.0 + rng.pareto(alpha, size=schema.n_rows))

                # --- Discretas (opcionales): generan enteros; tu pipeline puede luego cast a Int64 si decimals==0 ---
                elif dist == "poisson":
                    lam = float(p.get("lambda", 3.0))
                    lam = max(lam, 1e-12)
                    x = rng.poisson(lam, size=schema.n_rows).astype(float)

                elif dist == "binomial":
                    ntr = int(p.get("n", 10) or 10)
                    ntr = max(ntr, 1)
                    prob = float(p.get("p", 0.5))
                    prob = min(max(prob, 0.0), 1.0)
                    x = rng.binomial(ntr, prob, size=schema.n_rows).astype(float)

                else:
                    raise ValueError(f"Distribución no soportada: {dist}")

                x = _apply_postproc(x, p, rng)
                data[spec.name] = x

    # --- Categóricas ---
    for spec in cat_specs:
        cats = list(map(str, spec.categories))
        probs = np.array(spec.probs, dtype=float) if len(spec.probs) else np.ones(len(cats)) / max(len(cats), 1)
        if len(cats) == 0:
            continue
        if probs.shape[0] != len(cats):
            probs = np.ones(len(cats), dtype=float) / len(cats)
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones(len(cats), dtype=float) / len(cats)
        data[spec.name] = rng.choice(cats, size=schema.n_rows, p=probs)

    return pd.DataFrame(data)
