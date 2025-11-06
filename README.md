# 🧠 HeClaMoSTC – Herramienta de Clasificación de Movimientos asociados a STC

**Autor:** Karen Nicolle Arango Valencia  
**Universidad:** Pontificia Universidad Javeriana – Cali  
**Versión:** 1.0 (Funcional con 2 modelos y 5 sujetos)

---

## 📘 Descripción general

**HeClaMoSTC** (Herramienta de Clasificación de Movimientos asociados a STC) es una aplicación web basada en **Flask** y **TensorFlow/Keras** que permite **clasificar movimientos mioeléctricos (EMG)** en dos categorías:

- **Movimientos de riesgo (STC)**  
- **Movimientos seguros**

El sistema combina un **backend de procesamiento de señales EMG** con un **frontend web** interactivo.  
Se puede usar tanto con modelos de *machine learning* (ML) como de *deep learning* (DL), entrenados previamente en Google Colab.

---

## ⚙️ Pipeline de procesamiento

El flujo completo de señal es idéntico al implementado en el notebook de Colab:

1. **Filtrado EMG:**  
   - Filtro pasa-banda Butterworth (20–450 Hz)  
   - Filtro Notch a 50 Hz (Q = 30)

2. **Normalización:**  
   - Z-score usando `StandardScaler` (entrenado previamente)

3. **Ventaneo:**  
   - Ventanas de 300 ms (600 muestras a 2000 Hz)  
   - Solapamiento del 50%

4. **Extracción de características (para ML):**  
   - 144 *features* (60 tiempo + 24 frecuencia + 60 wavelet)

5. **Entrada al modelo:**  
   - ML → 144 features por ventana  
   - DL → secuencias (600 × 12)

6. **Clasificación binaria:**  
   - 0 = Seguro  
   - 1 = Riesgo  

---

## 🧩 Modelos compatibles

Los modelos deben estar guardados en la carpeta `models/`:

| Tipo | Nombre | Formato |
|------|---------|----------|
| ML | `model_ensemble_knn.pkl` | `pkl` |
| ML | `model_svm_rbf.pkl` | `pkl` |
| DL | `model_cnn_lstm.h5` | `h5` |
| DL | `model_bilstm.h5` | `h5` o `keras` |
| Escalador | `scaler.pkl` | Requerido para ambos tipos |

---

## 🧠 Movimientos clasificados

| Categoría | IDs |
|------------|-----|
| **Riesgo (1)** | 13, 14, 15, 16 |
| **Seguro (0)** | 0–12, 17 |

---

## 🗂️ Estructura del proyecto

```bash
HeClaMoSTC/
├── frontend/
│   ├── index.html
│   └── app.js
├── models/
│   ├── model_ensemble_knn.pkl
│   ├── model_cnn_lstm.h5
│   └── scaler.pkl
├── signals/
│   └── (archivos .mat subidos)
├── requirements.txt
├── server.py
└── .gitignore
