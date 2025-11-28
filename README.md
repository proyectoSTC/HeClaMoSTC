# 🏥 HeClaMoSTC - Herramienta de Clasificación de Movimientos para STC

**Sistema de Detección Automática de Movimientos de Riesgo para Síndrome del Túnel Carpiano**

Desarrollado por: **Karen Nicolle Arango Valencia**  
Universidad: **Pontificia Universidad Javeriana - Cali**

---

## 📋 Descripción

HeClaMoSTC es un sistema completo de clasificación binaria que detecta automáticamente movimientos de riesgo asociados al Síndrome del Túnel Carpiano (STC) a partir de señales electromiográficas (EMG).

### Características principales:
- ✅ **Clasificación binaria**: RIESGO vs SEGURO
- ✅ **Múltiples modelos**: Machine Learning (ML) y Deep Learning (DL)
- ✅ **Sistema Dual**: Combinación de dos modelos especializados en cascada
- ✅ **Interfaz web intuitiva**: Fácil de usar, sin necesidad de código
- ✅ **Pipeline completo**: Filtrado → Normalización → Clasificación → Visualización

---

## 🎯 Movimientos Clasificados

**Movimientos de RIESGO (4):**
- Movimiento 13: Flexión de muñeca
- Movimiento 14: Extensión de muñeca  
- Movimiento 15: Desviación radial
- Movimiento 16: Desviación ulnar

**Movimientos SEGUROS (13):**
- Movimientos 1-12: Agarres y gestos básicos
- Movimiento 17: Reposo

---

## 📦 Estructura del Proyecto

```
HeClaMoSTC/
│
├── frontend/
│   ├── index.html          # Interfaz web
│   └── app.js             # Lógica del frontend
│
├── server.py              # Backend Flask API
├── models/                # Modelos entrenados
│   ├── *.pkl             # Modelos ML
│   ├── *.keras           # Modelos DL
│   ├── scaler.pkl        # Normalizador Z-score
│   └── metadata.json     # Configuración y métricas
│
├── signals/              # Señales de prueba
│   └── *.mat            # Archivos MATLAB con EMG
│
├── notebooks/            # Jupyter Notebooks
│   ├── Copy_of_HeClaMoSTC_optimized.ipynb  # Entrenamiento
│   └── EMG_Spectral_Analysis.ipynb         # Análisis espectral
│
├── requirements.txt      # Dependencias Python
└── README.md            # Este archivo
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python
- **Google Colab**: Para entrenar modelos (opcional)

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/HeClaMoSTC.git
cd HeClaMoSTC
```

### 2. Crear Entorno Virtual (Recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Rutas

Edita `server.py` línea 53 para ajustar la ruta de señales externas:

```python
EXTERNAL_SIGNALS_DIR = Path(r'TU_RUTA_AQUI')  # Opcional
```

### 5. Crear Carpetas Necesarias

```bash
mkdir models signals frontend
```

### 6. Copiar Archivos

- Copia `index.html` y `app.js` a la carpeta `frontend/`
- Copia los modelos entrenados (`.pkl`, `.keras`) a la carpeta `models/`
- Copia `scaler.pkl` y `metadata.json` a la carpeta `models/`

---

## 🎮 Uso de la Aplicación Web

### Iniciar el Servidor

```bash
python server.py
```

Deberías ver:
```
======================================================================
SERVIDOR CLASIFICADOR STC
======================================================================

 Rutas: C:\...\models
 Window: 500ms, Overlap: 25.0%

 Sistema Dual DISPONIBLE:
   SAFE: RandomForest (precision 0.XXX)
   RISK: Ensemble_KNN (recall 0.XXX)

 http://localhost:5000
======================================================================
```

### Acceder a la Aplicación

Abre tu navegador web y visita:
```
http://localhost:5000
```

### Flujo de Clasificación

#### **Opción 1: Modo Independiente**

1. **Seleccionar Modo**: Elige "Modelo Independiente"
2. **Tipo de Modelo**: Selecciona ML o DL
3. **Modelo Específico**: 
   - **ML**: `ensemble_knn` o `random_forest`
   - **DL**: `cnn_lstm_attention` o `bilstm_attention`
4. **Seleccionar Señales**: Marca las señales `.mat` que deseas clasificar
5. **Clasificar**: Haz clic en "🚀 CLASIFICAR SEÑALES"

#### **Opción 2: Sistema Dual (Recomendado)**

1. **Seleccionar Modo**: Elige "Sistema Dual"
2. **Seleccionar Señales**: Marca las señales a clasificar
3. **Clasificar**: Haz clic en "🚀 CLASIFICAR SEÑALES"

El sistema dual usa dos modelos especializados:
- **Especialista SAFE**: Alta precisión en detectar movimientos seguros
- **Especialista RISK**: Alta sensibilidad en detectar movimientos de riesgo

### Cargar Señales Propias

1. Haz clic en "Cargar .mat desde tu PC"
2. Selecciona uno o más archivos `.mat`
3. Haz clic en "📥 Subir seleccionados"
4. Las señales aparecerán en la lista automáticamente

### Formato de Archivos .mat

Los archivos `.mat` deben contener:
- **Variable principal**: `emg` (matriz de señales EMG)
- **Dimensiones**: `[n_muestras × 12_canales]`
- **Frecuencia**: 2000 Hz
- **Metadata opcional**:
  - `subject`: Número de sujeto
  - `stimulus` o `restimulus`: Número de movimiento
  - `repetition` o `rerepetition`: Número de repetición

---

## 🧠 Entrenamiento de Modelos (Google Colab)

### Acceso al Notebook

El notebook de entrenamiento está diseñado para **Google Colab** con GPU.

**Link del Notebook**: `Copy_of_HeClaMoSTC_optimized.ipynb`

### Configuración del Entrenamiento

#### 1. Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 2. Configurar Rutas

Edita las rutas en la **Sección 2**:

```python
class Config:
    BASE_DIR = Path('/content/drive/MyDrive')
    DATA_DIR = BASE_DIR / 'DB2_E1_only'  # Tu carpeta con datos
    SAVE_DIR = BASE_DIR / 'New_ML_DL_models_stc_optimized'
```

#### 3. Seleccionar Sujetos

**Sección 3**:

```python
USE_ALL_SUBJECTS = True  # Para usar todos los sujetos
# O especificar:
SELECTED_SUBJECTS = [1, 2, 3, 4, 5]  # Lista personalizada
```

#### 4. Seleccionar Modelos

```python
SELECTED_MODELS = ['1', '2', '3', '4']  # Todos
# '1': Ensemble KNN
# '2': Random Forest
# '3': CNN+LSTM+Attention
# '4': BiLSTM+Attention
```

#### 5. Técnicas de Balanceo

**Para ML**:
```python
ML_BALANCE_TECHNIQUE = 'adasyn'  # 'none', 'adasyn', 'smote'
```

**Para DL**:
```python
DL_BALANCE_TECHNIQUE = 'augment_only'  
# 'none', 'augment_only', 'focal_loss', 'focal_loss+augment'
```

### Pipeline de Entrenamiento

El notebook ejecuta automáticamente:

1. **Carga de datos**: Lee señales EMG de archivos `.mat`
2. **Filtrado**: Butterworth (20-450 Hz) + Notch (50 Hz)
3. **Separación**: Train (rep 1,3,4,6) / Val (rep 2) / Test (rep 5)
4. **Normalización**: Z-score por canal
5. **Segmentación**: Ventanas de 500ms con 25% overlap
6. **Extracción de features** (ML):
   - Temporales: RMS, MAV, VAR, WL, SSC, ZC
   - Frecuenciales: MNF, MDF, PKF
   - Wavelet: Energía de coeficientes
7. **Entrenamiento**: 
   - ML con balanceo ADASYN/SMOTE
   - DL con Data Augmentation y/o Focal Loss
8. **Optimización**: Threshold optimization para maximizar F1-Score
9. **Evaluación**: Métricas en test set
10. **Guardado**: Modelos `.pkl` (ML) y `.keras` (DL)

### Resultados Generados

Al finalizar, encontrarás en tu Google Drive:

```
New_ML_DL_models_stc_optimized/
├── run_YYYYMMDD_HHMMSS/
│   ├── artifacts/
│   │   ├── ensemble_knn.pkl
│   │   ├── random_forest.pkl
│   │   ├── cnn_lstm_attention.keras
│   │   ├── bilstm_attention.keras
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   │
│   └── plots/
│       ├── metrics_comparison.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── training_history.png
```

### Descargar Modelos

1. Navega a `artifacts/` en Google Drive
2. Descarga todos los archivos:
   - `*.pkl` y `*.keras` (modelos)
   - `scaler.pkl` (normalizador)
   - `metadata.json` (configuración)
3. Copia los archivos a la carpeta `models/` de tu proyecto local

---

## 📊 Modelos Disponibles

### Machine Learning (ML)

#### 1. Ensemble Subspace KNN
- **Tipo**: Bagging de 30 clasificadores KNN
- **Características**: 
  - K-vecinos: 7
  - Métrica: Distancia euclidiana
  - Subsampling: 70% datos, 80% features
- **Ventaja**: Robusto a ruido, alta precisión

#### 2. Random Forest
- **Tipo**: Ensemble de 200 árboles de decisión
- **Características**:
  - Max depth: 30
  - Min samples split: 5
  - Max features: sqrt
- **Ventaja**: Rápido, interpretable

### Deep Learning (DL)

#### 3. CNN+LSTM+Attention
- **Arquitectura**:
  - 3 capas convolucionales (extracción espacial)
  - BiLSTM (patrones temporales)
  - Mecanismo de atención (enfoque selectivo)
- **Parámetros**: ~500K
- **Ventaja**: Captura patrones espacio-temporales complejos

#### 4. BiLSTM+Attention
- **Arquitectura**:
  - 3 capas BiLSTM profundas
  - Atención multiplicativa
  - Regularización L2 + Dropout 35%
- **Parámetros**: ~400K
- **Ventaja**: Excelente para secuencias largas

### Sistema Dual

Combina dos modelos ML en cascada:
- **Filtro SAFE**: Modelo con mayor precisión (descarta falsos positivos)
- **Detector RISK**: Modelo con mayor recall (captura todos los riesgos)

**Decisión**:
1. Si ambos coinciden → Alta confianza
2. Si difieren → Confianza media, prevalece detector RISK (conservador)

---

## 🔧 Configuración Técnica

### Parámetros de Procesamiento

```python
FS = 2000                # Frecuencia de muestreo (Hz)
WINDOW_SIZE_MS = 500     # Tamaño de ventana (ms)
OVERLAP = 0.25           # Solapamiento (25%)
N_CHANNELS = 12          # Canales EMG

# Filtrado
LOWCUT = 20              # Frecuencia baja (Hz)
HIGHCUT = 450            # Frecuencia alta (Hz)
NOTCH_FREQ = 50          # Filtro notch (Hz)
```

### Hiperparámetros DL

```python
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
DROPOUT = 0.35
L2_REG = 0.001
```

---

## 📊 Análisis Espectral de Señales (Opcional)

### Notebook de Visualización

Se incluye el notebook **`EMG_Spectral_Analysis.ipynb`** para análisis avanzado de señales EMG.

#### ¿Para qué sirve?

Esta herramienta te permite:
- 📈 **Visualizar espectros de frecuencia** de señales EMG sin filtrar
- 🔍 **Comparar movimientos RISK vs SAFE** en el dominio espectral
- 📊 **Analizar consistencia** entre repeticiones del mismo movimiento
- 🎯 **Identificar bandas de frecuencia dominantes** por canal
- 🧪 **Validar calidad** de señales antes de entrenar modelos

#### ¿Cómo usarlo?

1. Abre el notebook en **Google Colab**
2. Monta tu Google Drive con los datos
3. Ajusta la ruta en la configuración:
   ```python
   DATA_DIR = BASE_DIR / 'DB2_E1_only' / 'train'
   ```
4. Ejecuta todas las celdas secuencialmente
5. Revisa las gráficas generadas:
   - Espectros de potencia por canal
   - Comparación temporal de señales
   - Análisis de bandas de frecuencia
   - Mapas de calor de energía espectral

#### Características

- **Señales RAW**: Analiza señales sin filtrado previo
- **12 canales EMG**: Visualización individual y comparativa
- **40 sujetos**: Análisis poblacional completo
- **17 movimientos**: Clasificados en RISK (13-16) y SAFE (1-12, 17)
- **Bandas de frecuencia**: Very Low (0-20 Hz), Low (20-100 Hz), Mid (100-200 Hz), High (200-400 Hz), Very High (400-500 Hz)

#### Salidas típicas

- Gráficas de espectro de potencia
- Análisis de correlación entre repeticiones
- Comparación espectral RISK vs SAFE
- Distribución de energía por banda de frecuencia
- Estadísticas descriptivas por movimiento

**Nota**: Este notebook es complementario y no es necesario para el funcionamiento de la aplicación principal. Úsalo para exploración y análisis de datos.

## 📈 Interpretación de Resultados

### Tarjetas de Resultados

Cada señal clasificada muestra:

- **Estado**: RIESGO (rojo, animado) o SEGURO (verde)
- **Confianza**: Alta o Media (solo en Sistema Dual)
- **Probabilidades**: % de SAFE vs RISK
- **Ventanas**: Total analizadas
- **Metadata**: Sujeto, movimiento, repetición
- **Gráfica**: Señal EMG del primer canal

### Métricas de Evaluación

- **Accuracy**: Precisión general
- **Precision**: De todos los RISK predichos, cuántos son realmente RISK
- **Recall**: De todos los RISK reales, cuántos detectamos
- **F1-Score**: Balance entre precision y recall
- **AUC**: Área bajo la curva ROC

## 👥 Contribuciones

Este proyecto es parte de un trabajo de investigación académica. Para sugerencias o mejoras, contacta al autor.


**¡Gracias por usar HeClaMoSTC!**✨
