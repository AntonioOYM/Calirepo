
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Predicción de Precios de Vivienda (California)", layout="centered")
st.title("🏡 Predicción de Precios de Vivienda en California")

st.markdown(
    "Esta aplicación predice el valor medio de una vivienda (en unidades de $100,000) "
    "basándose en las características proporcionadas. Utiliza un modelo de Regresión Polinomial de Grado 3, "
    "aunque es importante recordar que este modelo mostró un R² negativo, lo que sugiere una baja fiabilidad." 
    "Los datos de entrada se escalan antes de la predicción."
)

# --- Cargar el Scaler y el Modelo --- 
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load('minmax_scaler_split_data.joblib')
        model = joblib.load('poly_reg_degree3_model.joblib')
        return scaler, model
    except FileNotFoundError:
        st.error("Error: Los archivos del scaler o del modelo no se encontraron. "
                 "Asegúrate de que 'minmax_scaler_split_data.joblib' y 'poly_reg_degree3_model.joblib' estén en la misma carpeta que 'app.py'.")
        st.stop()

scaler, model = load_resources()

# Para PolynomialFeatures, necesitamos que se ajuste a los datos de entrenamiento originales para recrear el espacio de características.
# Idealmente, 'poly_features_3' también debería haberse serializado. Como no lo fue, lo instanciamos aquí.
# En una aplicación real, se debería haber guardado 'poly_features_3' o entrenar 'poly_features_3' con los datos de entrenamiento
# de nuevo, lo que no es posible en un script de Streamlit sin los datos de entrenamiento.
# Para este ejemplo, asumimos que el orden y el número de características son consistentes.

# Asumimos las columnas de entrada que el modelo espera después de la eliminación de características
expected_columns = ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude']

# Instanciar PolynomialFeatures (esto es una recreación, no el objeto original ajustado)
# NOTA: En un caso de producción, poly_features_3 debería haberse serializado también.
poly_transformer = PolynomialFeatures(degree=3, include_bias=False)

# Creamos un dataframe dummy para 'entrenar' poly_transformer si no tenemos acceso a X_train_scaled_df
# Esto es un workaround para garantizar que la transformación tenga las mismas características que durante el entrenamiento.
# Un enfoque más robusto sería serializar el objeto poly_features_3 original.

dummy_data = pd.DataFrame([[0.0]*len(expected_columns)], columns=expected_columns)
poly_transformer.fit(scaler.transform(dummy_data)) # Ajustar con datos escalados (no ideales, pero simula el entrenamiento)


st.header("Introduce las características de la casa:")

# --- Entradas del Usuario --- 
medinc = st.slider("Ingreso Medio (MedInc)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
averooms = st.slider("Promedio de Habitaciones (AveRooms)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
avebedrms = st.slider("Promedio de Dormitorios (AveBedrms)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
aveoccup = st.slider("Promedio de Ocupantes (AveOccup)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
latitude = st.slider("Latitud (Latitude)", min_value=32.0, max_value=42.0, value=34.0, step=0.01)

# --- Botón de Predicción ---
if st.button("Predecir Precio"): 
    # Crear DataFrame con las entradas del usuario
    input_data = pd.DataFrame([{
        'MedInc': medinc,
        'AveRooms': averooms,
        'AveBedrms': avebedrms,
        'AveOccup': aveoccup,
        'Latitude': latitude
    }])

    # Escalar los datos de entrada
    scaled_input = scaler.transform(input_data)
    scaled_input_df = pd.DataFrame(scaled_input, columns=input_data.columns)

    # Transformar a características polinomiales
    poly_input = poly_transformer.transform(scaled_input_df)

    # Realizar la predicción
    predicted_value = model.predict(poly_input)[0]
    predicted_price_usd = predicted_value * 100000  # Convertir a USD

    st.success(f"El valor medio predicho de la vivienda es: **${predicted_price_usd:,.2f}**")
    st.info("Recuerda que este modelo de grado 3 puede no ser muy preciso.")

st.sidebar.markdown("### Información del Modelo")
st.sidebar.write("Modelo: Regresión Polinomial (Grado 3)")
st.sidebar.write("Scaler: MinMaxScaler")
st.sidebar.markdown("--- DISCLAIMER ---")
st.sidebar.write("Las predicciones de este modelo son solo para fines ilustrativos. Un R² negativo "
                 "indica un ajuste deficiente a los datos originales. Se recomienda usar modelos "
                 "con mejor rendimiento (ej. Regresión Lineal de este mismo análisis) para "
                 "predicciones más fiables en un entorno de producción.")
