from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash
)
import io, json
import numpy as np
import pandas as pd
import uuid

from generator import (
    generate_dataset, Schema, VariableSpec,
    ContinuousSpec, CategoricalSpec, HAS_SCIPY
)

# Matplotlib (para PNGs sin pyplot)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ===================== SDV: aprender desde CSV (modo data-driven) =====================
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdmetrics.reports.single_table import QualityReport

# ------------------------------------------------------------------------------ #
# Flask app
# ------------------------------------------------------------------------------ #
app = Flask(__name__)
app.secret_key = "dev"

# Dev helpers: recarga de plantillas y sin caché
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.cache = {}

# Exponer enumerate/zip/len a Jinja (por si alguna plantilla los usa)
app.jinja_env.globals.update(enumerate=enumerate, zip=zip, len=len)

print("TEMPLATES SEARCH PATH:", getattr(app.jinja_loader, "searchpath", "N/A"))

# Estado en memoria (demo)
schema = Schema()

# ------------------ Helper: sincronizar matriz de correlación ------------------ #
def _sync_corr_matrix():
    """Ajusta schema.corr_matrix al número actual de variables continuas."""
    cont = [v.continuous for v in schema.variables
            if v.kind == "continuous" and v.continuous]
    k = len(cont)

    # Si no hay cópula o <2 continuas, no usamos matriz
    if not schema.use_copula or k < 2:
        schema.corr_matrix = None
        return

    # Crear identidad si no existe
    if schema.corr_matrix is None:
        schema.corr_matrix = np.eye(k).tolist()
        return

    # Ajustar tamaño si cambió el número de continuas
    M = np.array(schema.corr_matrix, dtype=float)
    if M.shape == (k, k):
        return

    if M.shape[0] > k:
        # Se eliminaron continuas → recortar
        M = M[:k, :k]
    else:
        # Se agregaron continuas → expandir con identidad
        M2 = np.eye(k)
        s = M.shape[0]
        M2[:s, :s] = M
        M = M2

    schema.corr_matrix = M.tolist()


def _apply_display_rounding(df):
    for v in schema.variables:
        if v.kind == "continuous" and v.continuous:
            name = v.continuous.name
            d = v.continuous.params.get("decimals", None)
            if d is not None and d != "":
                try:
                    d = int(d)
                    if d == 0:
                        s = df[name].round(0)
                        df[name] = s.astype("Int64")
                    else:
                        df[name] = df[name].round(d)
                except Exception:
                    pass
    return df


# ------------------------------------------------------------------------------ #
# Rutas
# ------------------------------------------------------------------------------ #

@app.route("/", methods=["GET"])
def index():
    cont_count = sum(1 for v in schema.variables
                     if v.kind == "continuous" and v.continuous)
    return render_template("index.html",
                           schema=schema,
                           cont_count=cont_count,
                           has_scipy=HAS_SCIPY)

# -------- Opciones generales --------
@app.route("/set_options", methods=["POST"])
def set_options():
    try:
        schema.n_rows = int(request.form.get("n_rows", "1000"))
        schema.seed = int(request.form.get("seed", "42"))
        schema.use_copula = bool(request.form.get("use_copula"))
        _sync_corr_matrix()  # mantener consistencia al cambiar el flag
        flash("Opciones guardadas.", "success")
    except Exception as e:
        flash(f"Error en opciones: {e}", "danger")
    return redirect(url_for("index"))

# -------- Crear variables --------
@app.route("/add_continuous", methods=["POST"])
def add_continuous():
    try:
        name = (request.form.get("name") or "").strip()
        dist = (request.form.get("dist") or "").strip()
        if not name:
            raise ValueError("Falta el nombre de la variable continua.")

        params = {}

        # --------- Distribución ----------
        if dist == "normal":
            params["mu"] = float(request.form.get("mu", "0"))
            params["sigma"] = float(request.form.get("sigma", "1"))

        elif dist == "uniform":
            low = float(request.form.get("low", "0"))
            high = float(request.form.get("high", "1"))
            if high <= low:
                raise ValueError("En uniforme, max debe ser > min.")
            params["low"], params["high"] = low, high

        elif dist == "truncnorm":
            low = float(request.form.get("low_t", "0"))
            high = float(request.form.get("high_t", "1"))
            mu = float(request.form.get("mu_t", "0.5"))
            sigma = float(request.form.get("sigma_t", "0.2"))
            if high <= low:
                raise ValueError("En truncada, max debe ser > min.")
            if not (low <= mu <= high):
                raise ValueError("La media debe caer dentro de [min, max].")
            params.update({"low": low, "high": high, "mu": mu, "sigma": sigma})

        # Distribuciones soportadas
        elif dist == "lognormal":
            params["mu_log"] = float(request.form.get("mu_log", "0"))
            params["sigma_log"] = float(request.form.get("sigma_log", "1"))

        elif dist == "gamma":
            params["shape_k"] = float(request.form.get("shape_k", "1"))
            params["scale_theta"] = float(request.form.get("scale_theta", "1"))

        elif dist == "exponential":
            params["rate"] = float(request.form.get("rate", "1"))

        elif dist == "weibull":
            params["shape_k"] = float(request.form.get("shape_k", "1.5"))
            params["scale_lambda"] = float(request.form.get("scale_lambda", "1"))

        elif dist == "beta":
            params["alpha"] = float(request.form.get("alpha", "2"))
            params["beta"] = float(request.form.get("beta", "2"))

        elif dist == "triangular":
            low = float(request.form.get("low_tri", "0"))
            mode = float(request.form.get("mode", "0.5"))
            high = float(request.form.get("high_tri", "1"))
            if not (low <= mode <= high):
                raise ValueError("En triangular: min ≤ mode ≤ max debe cumplirse.")
            params.update({"low": low, "mode": mode, "high": high})

        elif dist == "student_t":
            params["df"] = float(request.form.get("df", "5"))

        elif dist == "chisquare":
            params["df"] = float(request.form.get("df", "4"))

        elif dist == "pareto":
            params["alpha"] = float(request.form.get("alpha", "3"))
            params["xm"] = float(request.form.get("xm", "1"))

        elif dist == "poisson":
            if schema.use_copula:
                raise ValueError("Poisson no es compatible con la cópula gaussiana. Desactiva 'Usar correlación' primero.")
            params["lambda"] = float(request.form.get("lambda", "3"))

        elif dist == "binomial":
            if schema.use_copula:
                raise ValueError("Binomial no es compatible con la cópula gaussiana. Desactiva 'Usar correlación' primero.")
            params["n"] = int(request.form.get("n", "10"))
            p = float(request.form.get("p", "0.5"))
            if not (0 <= p <= 1):
                raise ValueError("En binomial: p debe estar entre 0 y 1.")
            params["p"] = p

        else:
            raise ValueError(f"Distribución no soportada: {dist}")

        # Enforce min/max
        if request.form.get("enforce_minmax"):
            params["enforce_minmax"] = True
            min_clip = (request.form.get("min_clip", "").strip())
            max_clip = (request.form.get("max_clip", "").strip())
            if min_clip != "":
                params["min_clip"] = float(min_clip)
            if max_clip != "":
                params["max_clip"] = float(max_clip)
            if ("min_clip" in params and "max_clip" in params
                and params["max_clip"] <= params["min_clip"]):
                raise ValueError("max_clip debe ser > min_clip.")

        # Decimales (0..7)
        decimals = (request.form.get("decimals", "").strip())
        if decimals != "":
            d = max(0, min(7, int(decimals)))
            params["decimals"] = d

        # Faltantes / outliers / transform
        missing = (request.form.get("missing_pct", "").strip())
        if missing != "":
            params["missing_pct"] = max(0.0, min(1.0, float(missing)))

        outlier_pct = (request.form.get("outlier_pct", "").strip())
        if outlier_pct != "":
            params["outlier_pct"] = max(0.0, min(1.0, float(outlier_pct)))

        outlier_mult = (request.form.get("outlier_mult", "").strip())
        if outlier_mult != "":
            params["outlier_mult"] = float(outlier_mult)

        if request.form.get("keep_outliers"):
            params["keep_outliers"] = True

        transform = (request.form.get("transform", "") or "").strip()
        if transform:
            params["transform"] = transform

        # Guardar en esquema
        spec = ContinuousSpec(name=name, dist=dist, params=params)
        schema.variables.append(VariableSpec(kind="continuous", continuous=spec))
        _sync_corr_matrix()
        flash(f"Variable continua '{name}' agregada.", "success")

    except Exception as e:
        flash(f"Error: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/add_categorical", methods=["POST"])
def add_categorical():
    try:
        name = (request.form.get("name_cat") or "").strip()
        cats_raw = (request.form.get("categories") or "").strip()
        probs_raw = (request.form.get("probs") or "").strip()

        if not name:
            raise ValueError("Falta el nombre de la variable categórica.")
        if not cats_raw:
            raise ValueError("Debes especificar categorías (separadas por coma).")

        categories = [c.strip() for c in cats_raw.split(",") if c.strip()]
        probs = [float(x.strip()) for x in probs_raw.split(",")] if probs_raw else []

        spec = CategoricalSpec(name=name, categories=categories, probs=probs)
        schema.variables.append(VariableSpec(kind="categorical", categorical=spec))
        flash(f"Variable categórica '{name}' agregada.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("index"))

# -------- Editar / Eliminar --------
@app.route("/edit/<int:i>", methods=["GET"])
def edit_var(i: int):
    if not (0 <= i < len(schema.variables)):
        flash("Índice inválido.", "danger")
        return redirect(url_for("index"))
    v = schema.variables[i]
    if v.kind == "continuous" and v.continuous:
        return render_template("edit_continuous.html", i=i, spec=v.continuous, schema=schema)
    elif v.kind == "categorical" and v.categorical:
        return render_template("edit_categorical.html", i=i, spec=v.categorical)
    flash("Variable inválida.", "danger")
    return redirect(url_for("index"))


@app.route("/edit_continuous/<int:i>", methods=["POST"])
def update_continuous(i: int):
    try:
        if not (0 <= i < len(schema.variables)):
            raise ValueError("Índice inválido.")

        name = (request.form.get("name") or "").strip()
        dist = (request.form.get("dist") or "").strip()
        if not name:
            raise ValueError("Falta el nombre.")

        params = {}

        # --------- Distribución ----------
        if dist == "normal":
            params["mu"] = float(request.form.get("mu", "0"))
            params["sigma"] = float(request.form.get("sigma", "1"))

        elif dist == "uniform":
            low = float(request.form.get("low", "0"))
            high = float(request.form.get("high", "1"))
            if high <= low:
                raise ValueError("En uniforme, max debe ser > min.")
            params["low"], params["high"] = low, high

        elif dist == "truncnorm":
            low = float(request.form.get("low_t", "0"))
            high = float(request.form.get("high_t", "1"))
            mu = float(request.form.get("mu_t", "0.5"))
            sigma = float(request.form.get("sigma_t", "0.2"))
            if high <= low:
                raise ValueError("En truncada, max debe ser > min.")
            if not (low <= mu <= high):
                raise ValueError("La media debe caer dentro de [min, max].")
            params.update({"low": low, "high": high, "mu": mu, "sigma": sigma})

        elif dist == "lognormal":
            params["mu_log"] = float(request.form.get("mu_log", "0"))
            params["sigma_log"] = float(request.form.get("sigma_log", "1"))

        elif dist == "gamma":
            params["shape_k"] = float(request.form.get("shape_k", "1"))
            params["scale_theta"] = float(request.form.get("scale_theta", "1"))

        elif dist == "exponential":
            params["rate"] = float(request.form.get("rate", "1"))

        elif dist == "weibull":
            params["shape_k"] = float(request.form.get("shape_k", "1.5"))
            params["scale_lambda"] = float(request.form.get("scale_lambda", "1"))

        elif dist == "beta":
            params["alpha"] = float(request.form.get("alpha", "2"))
            params["beta"] = float(request.form.get("beta", "2"))

        elif dist == "triangular":
            low = float(request.form.get("low_tri", "0"))
            mode = float(request.form.get("mode", "0.5"))
            high = float(request.form.get("high_tri", "1"))
            if not (low <= mode <= high):
                raise ValueError("En triangular: min ≤ mode ≤ max debe cumplirse.")
            params.update({"low": low, "mode": mode, "high": high})

        elif dist == "student_t":
            params["df"] = float(request.form.get("df", "5"))

        elif dist == "chisquare":
            params["df"] = float(request.form.get("df", "4"))

        elif dist == "pareto":
            params["alpha"] = float(request.form.get("alpha", "3"))
            params["xm"] = float(request.form.get("xm", "1"))

        elif dist == "poisson":
            if schema.use_copula:
                raise ValueError("Poisson no es compatible con la cópula gaussiana. Desactiva 'Usar correlación' primero.")
            params["lambda"] = float(request.form.get("lambda", "3"))

        elif dist == "binomial":
            if schema.use_copula:
                raise ValueError("Binomial no es compatible con la cópula gaussiana. Desactiva 'Usar correlación' primero.")
            params["n"] = int(request.form.get("n", "10"))
            p = float(request.form.get("p", "0.5"))
            if not (0 <= p <= 1):
                raise ValueError("En binomial: p debe estar entre 0 y 1.")
            params["p"] = p

        else:
            raise ValueError(f"Distribución no soportada: {dist}")

        # Opciones extra
        if request.form.get("no_negative"):
            params["no_negative"] = True

        if request.form.get("enforce_minmax"):
            params["enforce_minmax"] = True
            min_clip = (request.form.get("min_clip", "").strip())
            max_clip = (request.form.get("max_clip", "").strip())
            if min_clip != "":
                params["min_clip"] = float(min_clip)
            if max_clip != "":
                params["max_clip"] = float(max_clip)
            if ("min_clip" in params and "max_clip" in params
                and params["max_clip"] <= params["min_clip"]):
                raise ValueError("max_clip debe ser > min_clip.")

        decimals = (request.form.get("decimals", "").strip())
        if decimals != "":
            d = max(0, min(7, int(decimals)))
            params["decimals"] = d

        missing = (request.form.get("missing_pct", "").strip())
        if missing != "":
            params["missing_pct"] = max(0.0, min(1.0, float(missing)))

        outlier_pct = (request.form.get("outlier_pct", "").strip())
        if outlier_pct != "":
            params["outlier_pct"] = max(0.0, min(1.0, float(outlier_pct)))

        outlier_mult = (request.form.get("outlier_mult", "").strip())
        if outlier_mult != "":
            params["outlier_mult"] = float(outlier_mult)

        if request.form.get("keep_outliers"):
            params["keep_outliers"] = True

        transform = (request.form.get("transform", "") or "").strip()
        if transform:
            params["transform"] = transform

        schema.variables[i].continuous = ContinuousSpec(name=name, dist=dist, params=params)
        _sync_corr_matrix()
        flash("Variable continua actualizada.", "success")

    except Exception as e:
        flash(f"Error: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/edit_categorical/<int:i>", methods=["POST"])
def update_categorical(i: int):
    try:
        if not (0 <= i < len(schema.variables)):
            raise ValueError("Índice inválido.")
        name = (request.form.get("name_cat") or "").strip()
        cats_raw = (request.form.get("categories") or "").strip()
        probs_raw = (request.form.get("probs") or "").strip()

        categories = [c.strip() for c in cats_raw.split(",") if c.strip()]
        probs = [float(x.strip()) for x in probs_raw.split(",")] if probs_raw else []

        schema.variables[i].categorical = CategoricalSpec(
            name=name, categories=categories, probs=probs
        )
        flash("Variable categórica actualizada.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("index"))


@app.route("/delete/<int:i>", methods=["POST"])
def delete_var(i: int):
    try:
        if 0 <= i < len(schema.variables):
            v = schema.variables[i]
            nombre = v.continuous.name if v.continuous else v.categorical.name
            del schema.variables[i]
            _sync_corr_matrix()
            flash(f"Variable '{nombre}' eliminada.", "info")
    except Exception as e:
        flash(f"Error al eliminar: {e}", "danger")
    return redirect(url_for("index"))

# -------- Correlaciones --------
@app.route("/correlations", methods=["GET", "POST"])
def correlations():
    cont_specs = [v.continuous for v in schema.variables
                  if v.kind == "continuous" and v.continuous]
    names = [c.name for c in cont_specs]
    k = len(names)

    if request.method == "POST":
        try:
            if k < 2:
                raise ValueError("Se necesitan al menos dos variables continuas.")

            # Construye matriz leyendo SOLO el triángulo superior y espejando
            M = np.eye(k)
            for i in range(k):
                for j in range(i + 1, k):
                    field_ij = f"c_{i}_{j}"
                    field_ji = f"c_{j}_{i}"

                    sval = request.form.get(field_ij)
                    if sval is None or sval == "":
                        sval = request.form.get(field_ji)

                    if sval is None or sval == "":
                        val = 0.0
                    else:
                        val = float(sval)
                        if not (-1.0 <= val <= 1.0):
                            raise ValueError(f"Correlación fuera de rango en ({i+1},{j+1}).")

                    M[i, j] = val
                    M[j, i] = val  # espejo

            np.fill_diagonal(M, 1.0)

            schema.corr_matrix = M.tolist()
            schema.use_copula = True
            flash("Matriz de correlación guardada.", "success")
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Error al guardar correlaciones: {e}", "danger")

    # GET: mostrar la matriz actual (o identidad)
    if schema.corr_matrix is not None:
        M = np.array(schema.corr_matrix, dtype=float)
        if M.shape != (k, k):
            M = np.eye(k)
    else:
        M = np.eye(k)

    return render_template("correlations.html", names=names, M=M, has_scipy=HAS_SCIPY)

# -------- Preview / Descarga --------
@app.route("/preview", methods=["GET"])
def preview():
    _sync_corr_matrix()
    df = generate_dataset(schema)
    df = _apply_display_rounding(df)

    table_html = df.head(50).to_html(classes="table table-striped table-sm", index=False)

    cont_names = [
        v.continuous.name
        for v in schema.variables
        if v.kind == "continuous" and v.continuous
    ]

    # Categóricas puras
    cat_names = [
        v.categorical.name
        for v in schema.variables
        if v.kind == "categorical" and v.categorical
    ]

    # Discretas: continuas con decimales = 0
    disc_names = []
    for v in schema.variables:
        if v.kind == "continuous" and v.continuous:
            d = v.continuous.params.get("decimals", None)
            try:
                if d is not None and int(d) == 0:
                    disc_names.append(v.continuous.name)
            except Exception:
                pass

    # Para barras: categóricas + discretas (sin duplicar)
    bar_names = cat_names + [n for n in disc_names if n not in cat_names]

    return render_template(
        "preview.html",
        table=table_html,
        cont_names=cont_names,
        bar_names=bar_names,
    )


# Detección robusta de Matplotlib
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    HAS_MPL = True
    MPL_ERR = None
except Exception as e:
    HAS_MPL = False
    MPL_ERR = str(e)

@app.route("/hist/<string:var>.png")
def hist_png(var: str):
    if not HAS_MPL:
        return "Instala matplotlib: pip install matplotlib", 500
    try:
        _sync_corr_matrix()
        df = generate_dataset(schema)
        df = _apply_display_rounding(df)

        if var not in df.columns:
            return f"Variable '{var}' no encontrada.", 404

        s = df[var]

        # Coerción segura a float
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return "Sin datos numéricos para graficar.", 400

        # Matplotlib sin pyplot
        fig = Figure(figsize=(4, 3), dpi=120)
        ax = fig.subplots()
        ax.hist(arr, bins=30)
        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("frecuencia")
        fig.tight_layout()

        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)

        resp = send_file(buf, mimetype="image/png")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generando histograma de '{var}': {e}", 500


@app.route("/bar/<string:var>.png")
def bar_png(var: str):
    if not HAS_MPL:
        return "Instala matplotlib: pip install matplotlib", 500
    try:
        _sync_corr_matrix()
        df = generate_dataset(schema)
        df = _apply_display_rounding(df)
        if var not in df.columns:
            return f"Variable '{var}' no encontrada.", 404

        s = df[var]

        # Normaliza a texto (conservando <NA>)
        s_str = s.astype("string").fillna("<NA>")

        # Cuenta categorías
        vc = s_str.value_counts(dropna=False)

        # Limita a top-K y agrupa “Otros”
        K = int(request.args.get("k", 25))
        if len(vc) > K:
            top = vc.iloc[:K]
            otros = int(vc.iloc[K:].sum())
            vc = pd.concat([top, pd.Series({"Otros": otros})])

        labels = vc.index.astype(str).tolist()
        counts = vc.values.astype(int).tolist()

        # Dibuja
        fig = Figure(figsize=(5.0, 3.4), dpi=120)
        ax = fig.subplots()
        ax.bar(range(len(counts)), counts)
        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("frecuencia")
        ax.set_xticks(range(len(labels)))
        labels_short = [x if len(x) <= 16 else x[:14] + "…" for x in labels]
        ax.set_xticklabels(labels_short, rotation=45, ha="right")
        fig.tight_layout()

        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)

        resp = send_file(buf, mimetype="image/png")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"Error generando barras de '{var}': {e}", 500


@app.route("/download", methods=["GET"])
def download():
    _sync_corr_matrix()
    df = generate_dataset(schema)
    df = _apply_display_rounding(df)

    buf = io.StringIO()
    df.to_csv(buf, index=False, na_rep="NA")
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),  # BOM para Excel
        mimetype="text/csv",
        as_attachment=True,
        download_name="synthetic_dataset.csv",
    )


# -------- Guardar / Cargar esquema --------
@app.route("/save_schema", methods=["GET"])
def save_schema():
    obj = {
        "n_rows": schema.n_rows,
        "seed": schema.seed,
        "use_copula": schema.use_copula,
        "corr_matrix": schema.corr_matrix,
        "variables": []
    }
    for v in schema.variables:
        if v.kind == "continuous" and v.continuous:
            obj["variables"].append({
                "kind": "continuous",
                "continuous": {
                    "name": v.continuous.name,
                    "dist": v.continuous.dist,
                    "params": v.continuous.params
                }
            })
        elif v.kind == "categorical" and v.categorical:
            obj["variables"].append({
                "kind": "categorical",
                "categorical": {
                    "name": v.categorical.name,
                    "categories": v.categorical.categories,
                    "probs": v.categorical.probs
                }
            })
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    return send_file(
        io.BytesIO(data.encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        download_name="schema.json",
    )


@app.route("/load_schema", methods=["POST"])
def load_schema():
    try:
        f = request.files.get("schema_file")
        if not f:
            raise ValueError("No subiste ningún archivo.")
        data = json.loads(f.read().decode("utf-8"))
        new_schema = Schema(
            n_rows=int(data.get("n_rows", 1000)),
            seed=int(data.get("seed", 42)),
            use_copula=bool(data.get("use_copula", False)),
            corr_matrix=data.get("corr_matrix")
        )
        vars_list = []
        for v in data.get("variables", []):
            kind = v.get("kind")
            if kind == "continuous" and "continuous" in v:
                cc = v["continuous"]
                vars_list.append(VariableSpec(
                    kind="continuous",
                    continuous=ContinuousSpec(
                        name=cc["name"], dist=cc["dist"],
                        params=cc.get("params", {})
                    )
                ))
            elif kind == "categorical" and "categorical" in v:
                cg = v["categorical"]
                vars_list.append(VariableSpec(
                    kind="categorical",
                    categorical=CategoricalSpec(
                        name=cg["name"],
                        categories=cg.get("categories", []),
                        probs=cg.get("probs", [])
                    )
                ))
        new_schema.variables = vars_list

        global schema
        schema = new_schema
        _sync_corr_matrix()  # asegurar tamaño correcto tras carga
        flash("Esquema cargado.", "success")
    except Exception as e:
        flash(f"Error al cargar esquema: {e}", "danger")
    return redirect(url_for("index"))


# ============================== SDV endpoints ============================== #

_SDVMODELS = {}  # {model_id: {"synth":..., "meta":..., "real_head": DataFrame}}
_SDV_LAST_ID = None  # ID del último modelo entrenado

@app.route("/sdv", methods=["GET"], endpoint="sdv_index")
def sdv_index():
    model_id = request.args.get("model_id")
    has_model = bool(model_id and model_id in _SDVMODELS)
    return render_template("sdv/index.html", has_model=has_model, model_id=model_id)

@app.post("/sdv/fit")
def sdv_fit():
    try:
        f = request.files.get("file")
        if not f or f.filename == "":
            raise ValueError("Debes seleccionar un archivo CSV.")
        # Evita DtypeWarning y respeta BOM
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

        # Detectar metadata desde instancia
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)

        synth = GaussianCopulaSynthesizer(meta)
        synth.fit(df)

        model_id = str(uuid.uuid4())
        _SDVMODELS[model_id] = {
            "synth": synth,
            "meta": meta,
            "real_head": df.head(200)
        }

        global _SDV_LAST_ID
        _SDV_LAST_ID = model_id

        # muestra rápida + calidad
        df_syn = synth.sample(num_rows=min(len(df), 200))
        try:
            qr = QualityReport()
            qr.generate(_SDVMODELS[model_id]["real_head"], df_syn, metadata=meta.to_dict())
            score = float(qr.get_score())
        except Exception:
            score = None

        table_html = _apply_display_rounding(df_syn).head(50).to_html(
            classes="table table-striped table-sm", index=False
        )
        return render_template(
            "sdv/trained.html",
            model_id=model_id,
            n_real=len(df),
            preview_syn=table_html,
            global_score=score
        )
    except Exception as e:
        flash(f"Error al entrenar SDV: {e}", "danger")
        return redirect(url_for("sdv_index"))


@app.get("/sdv/sample")
def sdv_sample():
    model_id = request.args.get("model_id")
    if not model_id or model_id not in _SDVMODELS:
        flash("No hay un modelo SDV disponible.", "warning")
        return redirect(url_for("sdv_index"))

    try:
        n = int(request.args.get("n", 1000))
    except Exception:
        n = 1000
    n = max(1, min(n, 1_000_000))

    df_syn = _SDVMODELS[model_id]["synth"].sample(num_rows=n)
    df_syn = _apply_display_rounding(df_syn)

    buf = io.StringIO()
    df_syn.to_csv(buf, index=False, na_rep="NA")
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"synthetic_{n}.csv"
    )


@app.get("/sdv/preview")
def sdv_preview():
    model_id = request.args.get("model_id")
    if not model_id or not (model_id in _SDVMODELS):
        flash("No hay un modelo SDV disponible.", "warning")
        return redirect(url_for("sdv_index"))
    n = int(request.args.get("n", 200))
    df_syn = _SDVMODELS[model_id]["synth"].sample(num_rows=n)
    table_html = _apply_display_rounding(df_syn).head(50).to_html(
        classes="table table-striped table-sm", index=False
    )
    return render_template(
        "sdv/trained.html",
        model_id=model_id,
        n_real=len(_SDVMODELS[model_id]["real_head"]),
        preview_syn=table_html,
        global_score=None
    )


@app.get("/sdv/metrics")
def sdv_metrics():
    model_id = request.args.get("model_id")

    # Fallback: usa el último entrenado si no llega model_id
    if (not model_id) and _SDV_LAST_ID and (_SDV_LAST_ID in _SDVMODELS):
        model_id = _SDV_LAST_ID

    if not model_id or model_id not in _SDVMODELS:
        return {"ok": False, "error": "No hay modelo"}, 400

    synth = _SDVMODELS[model_id]["synth"]
    meta = _SDVMODELS[model_id]["meta"]
    real_head = _SDVMODELS[model_id]["real_head"]

    df_syn = synth.sample(num_rows=min(200, len(real_head)))
    try:
        qr = QualityReport()
        qr.generate(real_head, df_syn, metadata=meta.to_dict())
        return {"ok": True, "score": float(qr.get_score())}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


@app.get("/sdv/export_metadata")
def sdv_export_metadata():
    model_id = request.args.get("model_id")
    if not model_id or model_id not in _SDVMODELS:
        return {"ok": False, "error": "No hay modelo"}, 400
    return {"ok": True, "metadata": _SDVMODELS[model_id]["meta"].to_dict()}


# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
