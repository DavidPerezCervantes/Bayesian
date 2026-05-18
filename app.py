import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, beta, expon

# ... (PEGAR AQUÍ EL RESTO DE TU CÓDIGO) ...
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, beta, expon

# ==========================================
# FUNCIÓN MATEMÁTICA PERSONALIZADA PARA EL HPDI
# ==========================================
def calcular_hpdi(muestras, prob=0.90):
    muestras_ordenadas = np.sort(muestras)
    n = len(muestras_ordenadas)
    tamanio_ventana = max(1, int(np.round(n * prob)))
    
    indices_inf = np.arange(n - tamanio_ventana + 1)
    indices_sup = indices_inf + tamanio_ventana - 1
    anchos = muestras_ordenadas[indices_sup] - muestras_ordenadas[indices_inf]
    
    indice_min = np.argmin(anchos)
    return muestras_ordenadas[indice_min], muestras_ordenadas[indice_min + tamanio_ventana - 1]

# ==========================================
# INTERFAZ Y ESTADO (UI & State)
# ==========================================
st.set_page_config(page_title="Oráculo Bayesiano", layout="wide")
st.title("🌍 Oráculo Bayesiano: Aprendizaje Secuencial y Predicción")

# Inicializar historial de lanzamientos si no existe
if 'secuencia' not in st.session_state:
    st.session_state.secuencia = []

# --- BARRA LATERAL (Sidebar) ---
with st.sidebar:
    st.header("1. Configuración Inicial")
    grid_size = st.slider("Tamaño de la malla (puntos):", min_value=3, max_value=100, value=50)
    tipo_prior = st.selectbox("Forma del Prior INICIAL (Día 0):", 
                              ["Uniforme", "Triangular", "Escalón", "Gaussiana", "Beta", "Exponencial"])
    
    st.divider()
    st.header("2. Historial de Lanzamientos")
    
    # Texto del historial
    s = st.session_state.secuencia
    historial_str = f"[ {', '.join(s)} ]" if len(s) > 0 else "Ningún lanzamiento. ¡Agrega datos!"
    st.info(f"📊 **Historial:** {historial_str}")
    
    # Botones (usando columnas para organizarlos)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    if col1.button("+1 Agua 💧", use_container_width=True):
        st.session_state.secuencia.append("W")
        st.rerun()
    if col2.button("+1 Tierra 🪨", use_container_width=True):
        st.session_state.secuencia.append("L")
        st.rerun()
    if col3.button("Deshacer ↩️", use_container_width=True):
        if len(st.session_state.secuencia) > 0:
            st.session_state.secuencia.pop()
            st.rerun()
    if col4.button("Reset 🔄", use_container_width=True):
        st.session_state.secuencia = []
        st.rerun()

# ==========================================
# LÓGICA MATEMÁTICA BÁSICA
# ==========================================
p_grid = np.linspace(0, 1, grid_size)

if tipo_prior == "Uniforme": base_prior = np.ones(grid_size)
elif tipo_prior == "Triangular": base_prior = np.where(p_grid < 0.5, p_grid * 2, (1 - p_grid) * 2)
elif tipo_prior == "Escalón": base_prior = np.where(p_grid < 0.5, 0, 1)
elif tipo_prior == "Gaussiana": base_prior = norm.pdf(p_grid, loc=0.5, scale=0.15)
elif tipo_prior == "Beta": base_prior = beta.pdf(p_grid, 2, 2)
elif tipo_prior == "Exponencial": base_prior = expon.pdf(p_grid, scale=1/5) # scale es 1/rate en Python

if len(s) == 0:
    current_prior = base_prior / np.sum(base_prior)
    likelihood = np.ones(grid_size)
    unstd_posterior = current_prior
    posterior = current_prior
    lik_title = "Paso 3: Sin datos (Likelihood plano)"
else:
    historial_pasado = s[:-1]
    ultimo_dato = s[-1]
    
    w_pasado = historial_pasado.count("W")
    n_pasado = len(historial_pasado)
    
    if n_pasado == 0:
        current_prior = base_prior / np.sum(base_prior)
    else:
        lik_pasado = binom.pmf(w_pasado, n_pasado, p_grid)
        prior_crudo_actual = base_prior * lik_pasado
        current_prior = prior_crudo_actual / np.sum(prior_crudo_actual)
        
    es_agua = 1 if ultimo_dato == "W" else 0
    likelihood = binom.pmf(es_agua, 1, p_grid)
    unstd_posterior = current_prior * likelihood
    posterior = unstd_posterior / np.sum(unstd_posterior)
    lik_title = f"Paso 3: Likelihood de sacar 1 {'Agua 💧' if es_agua else 'Tierra 🪨'}"

# ==========================================
# PESTAÑAS PRINCIPALES (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["Capítulo 2: Receta Paso a Paso", "Capítulo 3: Muestreo y Predicción"])

with tab1:
    st.subheader("3. Controles del Capítulo 2")
    paso = st.radio("Selecciona el paso de la receta:", 
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: [
                        "1. Definir la malla 💧", "2. Prior Actualizado 🌿", 
                        "3. Likelihood del ÚLTIMO dato 🌊", "4. Posterior crudo ✖️", 
                        "5. Posterior vs Prior Actual ⚖️"][x-1],
                    horizontal=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#FDFBF7')
    
    if paso == 1:
        ax.vlines(p_grid, ymin=0, ymax=1, colors='steelblue', lw=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Proporción de agua (p)", title="Paso 1: Malla de valores posibles")
        ax.set_yticks([])
    elif paso == 2:
        ax.vlines(p_grid, ymin=0, ymax=current_prior, colors='forestgreen', lw=4)
        ax.set(xlim=(0, 1), ylim=(0, max(current_prior) * 1.2), xlabel="Proporción de agua (p)", ylabel="Plausibilidad a priori", title="Paso 2: Prior Actualizado")
    elif paso == 3:
        ax.vlines(p_grid, ymin=0, ymax=likelihood, colors='navy', lw=4)
        ax.vlines(p_grid, ymin=0, ymax=likelihood.max(), colors=(0.2, 0.5, 0.8, 0.2), lw=1, linestyles='dashed')
        ax.set(xlim=(0, 1), ylim=(0, max(likelihood) * 1.2), xlabel="Proporción de agua (p)", ylabel="Verosimilitud", title=lik_title)
    elif paso == 4:
        ax.vlines(p_grid, ymin=0, ymax=unstd_posterior, colors='purple', lw=4)
        ax.vlines(p_grid, ymin=0, ymax=unstd_posterior.max(), colors=(0.5, 0, 0.5, 0.1), lw=1, linestyles='dashed')
        ax.set(xlim=(0, 1), ylim=(0, max(unstd_posterior) * 1.2), xlabel="Proporción de agua (p)", ylabel="Posterior crudo", title="Paso 4: Posterior crudo (Prior Actual × Likelihood)")
    elif paso == 5:
        max_y = max(np.max(posterior), np.max(current_prior))
        ax.vlines(p_grid, ymin=0, ymax=posterior, colors='darkorange', lw=4, label="Nuevo Posterior")
        ax.vlines(p_grid, ymin=0, ymax=current_prior, colors=(0.13, 0.55, 0.13, 0.3), lw=4, label="Prior (Antes del clic)")
        ax.set(xlim=(0, 1), ylim=(0, max_y * 1.2), xlabel="Proporción de agua (p)", ylabel="Probabilidad", title="Paso 5: Aprendizaje del último lanzamiento")
        ax.legend()
        
    st.pyplot(fig)

with tab2:
    st.subheader("3. Controles del Capítulo 3")
    colA, colB, colC = st.columns(3)
    hdi_level = colA.slider("Nivel del Intervalo (HPDI):", 0.50, 0.99, 0.90, 0.01)
    N_pred = colB.number_input("Lanzamientos futuros a simular:", min_value=1, value=15)
    W_pred = colC.number_input("Agua objetivo a predecir:", min_value=0, max_value=N_pred, value=8)

    # --- LÓGICA DE SIMULACIÓN Y HPDI ---
    np.random.seed(100)
    samples = np.random.choice(p_grid, p=posterior, size=10000, replace=True)
    hdi_inf, hdi_sup = calcular_hpdi(samples, prob=hdi_level)
    w_sim = np.random.binomial(n=N_pred, p=samples, size=10000)
    prob_pred = np.mean(w_sim == W_pred)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**Límites del HPDI al {int(hdi_level*100)}%:**\n\nInferior: {hdi_inf:.4f} | Superior: {hdi_sup:.4f}")
    with c2:
        st.info(f"**Probabilidad de sacar {W_pred} en {N_pred} intentos:**\n\n{prob_pred*100:.2f}%")

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig2.patch.set_facecolor('#FDFBF7')
    
    # Gráfico Posterior
    ax1.plot(p_grid, posterior, lw=3, color='#3b5b92')
    ax1.set(xlabel="Proporción de agua (p)", ylabel="Probabilidad Posterior", title="Distribución Posterior y HPDI")
    mask = (p_grid >= hdi_inf) & (p_grid <= hdi_sup)
    ax1.fill_between(p_grid, posterior, where=mask, color=(0.2, 0.4, 0.7, 0.3))
    ax1.axvline(hdi_inf, color='darkred', linestyle='--')
    ax1.axvline(hdi_sup, color='darkred', linestyle='--')
    
    # Gráfico de Predicción (Barplot)
    valores, conteos = np.unique(w_sim, return_counts=True)
    frecuencias = dict(zip(valores, conteos))
    eje_x = np.arange(N_pred + 1)
    eje_y = [frecuencias.get(x, 0) for x in eje_x]
    
    colores = ['lightgray'] * (N_pred + 1)
    if W_pred <= N_pred: colores[W_pred] = 'darkorange'
        
    ax2.bar(eje_x, eje_y, color=colores, edgecolor='white')
    ax2.set(xlabel="Número de éxitos simulados", ylabel="Frecuencia", title="Distribución Predictiva Posterior (10,000 Simulaciones)")
    
    plt.tight_layout()
    st.pyplot(fig2)
