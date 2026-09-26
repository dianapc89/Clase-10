# -*- coding: utf-8 -*-
"""
Random Forest - Clasificación de Tensión de Liquidez (Capital de Trabajo)
Ejecutar con:  streamlit run app_random_forest_liquidez.py
Requiere: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl, joblib
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------
ARCHIVO_LOCAL = Path(__file__).resolve().parent / "Base_Didactica_Random_Forest_Capital_Trabajo.xlsx"
HOJA_DATOS = "Datos_Modelo"
TARGET = "Tension_Liquidez_bin"
EXCLUIR = ["ID_Observacion", "Fecha", "Prob_Tension_Liquidez", "Brecha_Caja_90d_MXN"]
CATEGORICAS = ["Sector"]
TEST_SIZE = 0.30
RANDOM_STATE = 42

st.set_page_config(page_title="Random Forest · Tensión de Liquidez", page_icon="🌳", layout="wide")


# ---------------------------------------------------------------------------
# 2. FUNCIONES DE DATOS Y MODELO
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(contenido: bytes | None) -> pd.DataFrame:
    """Lee la hoja Datos_Modelo del Excel subido o, si no hay, del archivo local."""
    fuente = io.BytesIO(contenido) if contenido is not None else ARCHIVO_LOCAL
    return pd.read_excel(fuente, sheet_name=HOJA_DATOS)


def preparar_xy(df: pd.DataFrame):
    X = df.drop(columns=[c for c in EXCLUIR + [TARGET] if c in df.columns])
    y = df[TARGET].astype(int)
    return X, y


def construir_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
    cat = [c for c in CATEGORICAS if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", "passthrough", num),
        ],
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    return Pipeline([("prep", pre), ("rf", rf)])


@st.cache_resource(show_spinner=False)
def entrenar_modelo(_X_train, _y_train, params_key: tuple, datos_key: int):
    params = dict(params_key)
    modelo = construir_pipeline(_X_train, params)
    modelo.fit(_X_train, _y_train)
    return modelo


def busqueda_automatica(X_train, y_train, class_weight):
    espacio = {
        "rf__n_estimators": [100, 200, 300, 500, 800],
        "rf__max_depth": [None, 4, 6, 8, 10, 15, 20],
        "rf__min_samples_split": [2, 5, 10, 20],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", None, 0.5],
    }
    base = construir_pipeline(X_train, {"class_weight": class_weight})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    busqueda = RandomizedSearchCV(
        base, espacio, n_iter=30, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def calcular_metricas(y_true, proba, umbral):
    pred = (proba >= umbral).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
    }


@st.cache_data(show_spinner=False)
def importancia_permutacion(_modelo, _X_test, _y_test, cache_key: str):
    r = permutation_importance(
        _modelo, _X_test, _y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({"Variable": _X_test.columns, "Importancia": r.importances_mean, "Desv_Est": r.importances_std})
        .sort_values("Importancia", ascending=True)
    )


def a_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicciones")
    return buffer.getvalue()


def reporte_excel(hojas: dict) -> bytes:
    """Genera un Excel con una hoja por DataFrame y ajusta el ancho de columnas."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre, tabla in hojas.items():
            tabla.to_excel(writer, sheet_name=nombre, index=False)
            ws = writer.sheets[nombre]
            for col in ws.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(largo + 2, 10), 45)
            ws.freeze_panes = "A2"
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 3. BARRA LATERAL
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

st.sidebar.subheader("1. Datos")
archivo = st.sidebar.file_uploader("Sube la base (Excel, hoja 'Datos_Modelo')", type=["xlsx"])

st.sidebar.subheader("2. Hiperparámetros")
n_estimators = st.sidebar.slider("n_estimators (número de árboles)", 50, 1000, 300, step=50)
sin_limite = st.sidebar.checkbox("max_depth sin límite (None)", value=True)
max_depth = None if sin_limite else st.sidebar.slider("max_depth", 2, 30, 10)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 30, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 20, 1)
max_features_label = st.sidebar.selectbox("max_features", ["sqrt", "log2", "Todas (None)", "0.5"], index=0)
max_features = {"sqrt": "sqrt", "log2": "log2", "Todas (None)": None, "0.5": 0.5}[max_features_label]

st.sidebar.subheader("3. Clases y umbral")
cw_label = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
class_weight = None if cw_label == "None" else "balanced"
umbral = st.sidebar.slider("Umbral de clasificación", 0.10, 0.90, 0.50, step=0.01)

st.sidebar.subheader("4. Búsqueda automática")
st.sidebar.caption("RandomizedSearchCV · 30 combinaciones · CV 5 · métrica ROC-AUC")
lanzar_busqueda = st.sidebar.button("🔎 Ejecutar búsqueda")

# ---------------------------------------------------------------------------
# 4. CARGA Y PARTICIÓN
# ---------------------------------------------------------------------------
st.title("🌳 Random Forest · Predicción de Tensión de Liquidez")
st.caption("Clasificación binaria sobre variables de capital de trabajo · partición 70/30 estratificada · random_state=42")

contenido = archivo.getvalue() if archivo is not None else None
if contenido is None and not ARCHIVO_LOCAL.exists():
    st.info(f"Sube la base desde la barra lateral o coloca **{ARCHIVO_LOCAL.name}** en la carpeta del script.")
    st.stop()

try:
    df = cargar_datos(contenido)
except Exception as e:
    st.error(f"No se pudo leer la hoja '{HOJA_DATOS}': {e}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"La base no contiene la variable objetivo '{TARGET}'.")
    st.stop()

X, y = preparar_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
datos_key = hash(pd.util.hash_pandas_object(df, index=False).sum())

# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO (manual o búsqueda)
# ---------------------------------------------------------------------------
params_manual = {
    "n_estimators": n_estimators, "max_depth": max_depth,
    "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
    "max_features": max_features, "class_weight": class_weight,
}

if lanzar_busqueda:
    with st.spinner("Ejecutando búsqueda de hiperparámetros (puede tardar un par de minutos)..."):
        b = busqueda_automatica(X_train, y_train, class_weight)
    st.session_state["busqueda"] = {
        "modelo": b.best_estimator_,
        "params": {k.replace("rf__", ""): v for k, v in b.best_params_.items()},
        "cv_auc": b.best_score_,
        "datos_key": datos_key,
        "class_weight": class_weight,
    }

hay_busqueda = "busqueda" in st.session_state and st.session_state["busqueda"]["datos_key"] == datos_key
opciones_modelo = ["Manual (barra lateral)"] + (["Búsqueda automática"] if hay_busqueda else [])
modo = st.sidebar.radio("Modelo en uso", opciones_modelo, index=len(opciones_modelo) - 1)

if modo == "Búsqueda automática":
    modelo = st.session_state["busqueda"]["modelo"]
    params_usados = {**st.session_state["busqueda"]["params"], "class_weight": st.session_state["busqueda"]["class_weight"]}
else:
    with st.spinner("Entrenando Random Forest..."):
        modelo = entrenar_modelo(X_train, y_train, tuple(sorted(params_manual.items(), key=lambda kv: kv[0])), datos_key)
    params_usados = params_manual

proba_train = modelo.predict_proba(X_train)[:, 1]
proba_test = modelo.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= umbral).astype(int)
met_train = calcular_metricas(y_train, proba_train, umbral)
met_test = calcular_metricas(y_test, proba_test, umbral)

st.sidebar.divider()
paquete = {
    "pipeline": modelo, "umbral": umbral, "parametros": params_usados,
    "variables": list(X.columns), "target": TARGET,
}
buf = io.BytesIO()
joblib.dump(paquete, buf)
st.sidebar.download_button(
    "💾 Descargar modelo (.joblib)", buf.getvalue(),
    file_name="modelo_rf_tension_liquidez.joblib", mime="application/octet-stream",
)

# Importancia de variables (se usa en la pestaña 3 y en el reporte Excel)
nombres = modelo.named_steps["prep"].get_feature_names_out()
imp = pd.DataFrame({"Variable": nombres, "Importancia": modelo.named_steps["rf"].feature_importances_})
imp = imp.sort_values("Importancia", ascending=True)
with st.spinner("Calculando importancia por permutación..."):
    clave = f"{modo}-{datos_key}-{sorted(params_usados.items(), key=lambda kv: kv[0])}"
    perm = importancia_permutacion(modelo, X_test, y_test, clave)

# Reporte Excel con los resultados del modelo
pred_test = (proba_test >= umbral).astype(int)
cm_df = pd.DataFrame(confusion_matrix(y_test, pred_test),
                     index=["Real Normal", "Real Tensión"], columns=["Pred. Normal", "Pred. Tensión"]).reset_index()
cm_df = cm_df.rename(columns={"index": "Clase real"})
metricas_df = pd.DataFrame({"Métrica": list(met_test.keys()),
                            "Entrenamiento": list(met_train.values()),
                            "Prueba": list(met_test.values())})
metricas_df["Diferencia"] = metricas_df["Entrenamiento"] - metricas_df["Prueba"]
rep_df = pd.DataFrame(classification_report(y_test, pred_test, target_names=["Normal", "Tensión"],
                                            output_dict=True, zero_division=0)).T.reset_index()
rep_df = rep_df.rename(columns={"index": "Clase"})
config_df = pd.DataFrame(
    [("Modelo en uso", modo), ("Umbral de clasificación", umbral),
     ("Partición", "70/30 aleatoria estratificada, random_state=42"),
     ("Observaciones entrenamiento / prueba", f"{len(X_train)} / {len(X_test)}"),
     ("Variables excluidas", ", ".join(EXCLUIR)), ("Métrica principal", "ROC-AUC")]
    + [(f"Hiperparámetro: {k}", str(v)) for k, v in params_usados.items()],
    columns=["Parámetro", "Valor"],
)
pred_df = X_test.copy()
if "ID_Observacion" in df.columns:
    pred_df.insert(0, "ID_Observacion", df.loc[X_test.index, "ID_Observacion"].values)
pred_df["Tension_Real"] = y_test.values
pred_df["Prob_Tension_Predicha"] = proba_test
pred_df["Tension_Predicha"] = pred_test
pred_df["Acierto"] = (pred_df["Tension_Real"] == pred_df["Tension_Predicha"]).map({True: "Sí", False: "No"})

excel_reporte = reporte_excel({
    "Configuracion": config_df,
    "Metricas": metricas_df.round(4),
    "Matriz_Confusion": cm_df,
    "Reporte_Clasificacion": rep_df.round(4),
    "Importancia_Nativa": imp.sort_values("Importancia", ascending=False).round(4),
    "Importancia_Permutacion": perm.sort_values("Importancia", ascending=False).round(4),
    "Predicciones_Prueba": pred_df,
})
st.sidebar.download_button(
    "📥 Descargar resultados (Excel)", excel_reporte,
    file_name="resultados_random_forest_liquidez.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ---------------------------------------------------------------------------
# 6. PESTAÑAS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración", "✅ Desempeño", "🔑 Importancia de variables", "🔮 Predicción"])

# --- Pestaña 1: Exploración -------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", f"{len(df):,}")
    c2.metric("Variables predictoras", X.shape[1])
    c3.metric("Entrenamiento / Prueba", f"{len(X_train)} / {len(X_test)}")
    c4.metric("% con tensión", f"{y.mean():.1%}")

    st.subheader("Vista previa de la base")
    st.dataframe(df.head(20), width="stretch")
    st.caption("Variables excluidas del entrenamiento: " + ", ".join(EXCLUIR))

    g1, g2 = st.columns(2)
    dist = y.map({0: "0 · Normal", 1: "1 · Tensión"}).value_counts().reset_index()
    dist.columns = ["Clase", "Observaciones"]
    g1.plotly_chart(
        px.bar(dist, x="Clase", y="Observaciones", color="Clase", text="Observaciones",
               title="Distribución de la variable objetivo"),
        width="stretch",
    )
    if "Sector" in df.columns:
        sec = df.groupby(["Sector", TARGET]).size().reset_index(name="Observaciones")
        sec[TARGET] = sec[TARGET].map({0: "Normal", 1: "Tensión"})
        g2.plotly_chart(
            px.bar(sec, x="Sector", y="Observaciones", color=TARGET, barmode="group",
                   title="Observaciones por sector y clase"),
            width="stretch",
        )

    st.subheader("Estadística descriptiva")
    st.dataframe(X.describe().T, width="stretch")

# --- Pestaña 2: Desempeño ---------------------------------------------------
with tab2:
    st.markdown(f"**Modelo en uso:** {modo} · **Umbral:** {umbral:.2f}")
    if modo == "Búsqueda automática":
        st.success(f"ROC-AUC promedio en validación cruzada: {st.session_state['busqueda']['cv_auc']:.4f}")
    st.json({k: (str(v) if v is None else v) for k, v in params_usados.items()}, expanded=False)

    st.subheader("Métricas en el conjunto de prueba")
    cols = st.columns(5)
    for col, (nombre, valor) in zip(cols, met_test.items()):
        col.metric(("⭐ " if nombre == "ROC-AUC" else "") + nombre, f"{valor:.3f}")

    g1, g2 = st.columns(2)
    cm = confusion_matrix(y_test, pred_test)
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Pred. Normal", "Pred. Tensión"], y=["Real Normal", "Real Tensión"],
        title="Matriz de confusión (prueba)",
    )
    g1.plotly_chart(fig_cm, width="stretch")

    fpr, tpr, thr = roc_curve(y_test, proba_test)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {met_test['ROC-AUC']:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash")))
    idx = int(np.argmin(np.abs(thr - umbral)))
    fig_roc.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode="markers", name=f"Umbral {umbral:.2f}",
                                 marker=dict(size=12, symbol="x")))
    fig_roc.update_layout(title="Curva ROC (prueba)", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos")
    g2.plotly_chart(fig_roc, width="stretch")

    st.subheader("Entrenamiento vs. prueba (detección de sobreajuste)")
    comp = pd.DataFrame({"Entrenamiento": met_train, "Prueba": met_test})
    comp["Diferencia"] = comp["Entrenamiento"] - comp["Prueba"]
    st.dataframe(comp.style.format("{:.3f}"), width="stretch")
    if comp.loc["ROC-AUC", "Diferencia"] > 0.10:
        st.warning("La diferencia de ROC-AUC entre entrenamiento y prueba supera 0.10: posible sobreajuste. "
                   "Prueba limitar max_depth o aumentar min_samples_leaf.")

    st.download_button(
        "📥 Descargar resultados (Excel)", excel_reporte,
        file_name="resultados_random_forest_liquidez.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="descarga_excel_tab",
    )

    st.subheader("Reporte de clasificación (prueba)")
    rep = classification_report(y_test, pred_test, target_names=["Normal", "Tensión"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"), width="stretch")

# --- Pestaña 3: Importancia -------------------------------------------------
with tab3:
    g1, g2 = st.columns(2)
    g1.plotly_chart(
        px.bar(imp, x="Importancia", y="Variable", orientation="h", height=700,
               title="Importancia nativa (reducción de impureza Gini)"),
        width="stretch",
    )
    g2.plotly_chart(
        px.bar(perm, x="Importancia", y="Variable", orientation="h", error_x="Desv_Est", height=700,
               title="Importancia por permutación (caída de ROC-AUC en prueba)"),
        width="stretch",
    )
    st.caption("La importancia nativa puede repartirse entre variables correlacionadas (p. ej. CCC_dias con DSO/DIO/DPO). "
               "La importancia por permutación mide cuánto empeora el ROC-AUC al desordenar cada variable.")

# --- Pestaña 4: Predicción --------------------------------------------------
with tab4:
    st.subheader("Predicción individual")
    with st.form("form_prediccion"):
        entradas = {}
        columnas = st.columns(3)
        for i, var in enumerate(X.columns):
            col = columnas[i % 3]
            if var in CATEGORICAS:
                cats = sorted(X[var].dropna().unique().tolist())
                entradas[var] = col.selectbox(var, cats)
            elif set(X[var].dropna().unique()) <= {0, 1}:
                entradas[var] = col.selectbox(var, [0, 1], index=int(X[var].median()))
            else:
                entradas[var] = col.number_input(var, value=float(X[var].median()), format="%.4f")
        enviar = st.form_submit_button("Predecir")

    if enviar:
        caso = pd.DataFrame([entradas])[X.columns]
        p = float(modelo.predict_proba(caso)[:, 1][0])
        clase = "TENSIÓN DE LIQUIDEZ" if p >= umbral else "SITUACIÓN NORMAL"
        (st.error if p >= umbral else st.success)(f"Probabilidad de tensión: **{p:.1%}** → {clase} (umbral {umbral:.2f})")

    st.divider()
    st.subheader("Predicción por lote")
    st.caption("Sube un Excel o CSV con las mismas columnas predictoras. Las columnas adicionales se conservan en la salida.")
    lote = st.file_uploader("Archivo con nuevos casos", type=["xlsx", "csv"], key="lote")
    if lote is not None:
        try:
            nuevos = pd.read_csv(lote) if lote.name.lower().endswith(".csv") else pd.read_excel(lote)
            faltantes = [c for c in X.columns if c not in nuevos.columns]
            if faltantes:
                st.error("Faltan columnas: " + ", ".join(faltantes))
            else:
                salida = nuevos.copy()
                salida["Prob_Tension_Predicha"] = modelo.predict_proba(nuevos[X.columns])[:, 1]
                salida["Tension_Predicha_bin"] = (salida["Prob_Tension_Predicha"] >= umbral).astype(int)
                st.dataframe(salida, width="stretch")
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Descargar Excel", a_excel(salida), "predicciones_tension_liquidez.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                d2.download_button("⬇️ Descargar CSV", salida.to_csv(index=False).encode("utf-8-sig"),
                                   "predicciones_tension_liquidez.csv", "text/csv")
        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")
