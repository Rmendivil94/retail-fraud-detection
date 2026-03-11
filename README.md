# Profit-Driven Fraud Detection Engine: A Cost-Sensitive XGBoost Approach

*[Read in English](#english-version) | [Leer en Español](#versión-en-español)*

---

<a id="english-version"></a>
## English Version

### Executive Summary
This project implements a scalable, production-ready Machine Learning pipeline to detect fraudulent credit card transactions in a retail e-commerce environment. Moving beyond academic metrics, the final model is strictly optimized for **Business Profitability (Cost-Sensitive Learning)**. 

By finding the exact financial equilibrium between intercepting fraud (True Positives) and minimizing customer friction (False Positives), the deployed XGBoost model generated a net savings of **$149,188.50 USD** on a single Out-of-Time test set.

### The Business Trade-off
In the retail and payments industry, optimizing for traditional metrics like F1-Score often destroys business value due to asymmetric costs:
* **The Cost of Fraud (False Negative):** Losing the physical product plus bank chargeback fees (>100% loss, estimated at $75.00).
* **The Cost of Friction (False Positive):** Losing the profit margin and alienating a legitimate customer (Opportunity cost, approx. 10% margin, estimated at $6.85).

This engine evaluates predictive probabilities against actual P&L (Profit & Loss) impact, moving the decision threshold from an arbitrary `0.50` to a mathematically optimal **`0.5859`**, maximizing retained revenue.

### Architectural Pipeline & MLOps
1. **Advanced Feature Engineering:** Target Encoding for high-cardinality categorical features and Cyclical Time Extraction for behavioral signatures.
2. **Algorithmic Dimensionality Reduction:** Applied the Elbow Method on LightGBM's cumulative information gain, pruning the feature space from ~400 down to **67 elite features** for ultra-low inference latency.
3. **Out-of-Time (OOT) Validation:** Prevented Data Leakage by chronologically sorting the dataset, using the final 20% (future data) as a strict hold-out test set to simulate production data drift.
4. **Cost-Sensitive XGBoost:** Trained with `scale_pos_weight` for extreme class imbalance and serialized using XGBoost's native JSON format for C++/Java deployment compatibility.

### Key Results
* **ROC-AUC Score:** `0.9027` (Out-of-Time validation).
* **Optimal Profit Threshold:** `0.5859` 
* **Net Financial Impact:** **+$149,188.50 USD** saved in a single validation window.

---

<a id="versión-en-español"></a>
## Versión en Español

### Resumen
Este proyecto implementa un pipeline de Machine Learning escalable y listo para producción, diseñado para detectar transacciones fraudulentas con tarjetas de crédito en un entorno de retail y comercio electrónico. Yendo más allá de las métricas académicas, el modelo final está estrictamente optimizado para la **Rentabilidad del Negocio (Cost-Sensitive Learning)**.

Al encontrar el equilibrio financiero exacto entre interceptar el fraude (Verdaderos Positivos) y minimizar la fricción del cliente (Falsos Positivos), el modelo XGBoost desplegado generó un ahorro neto validado de **$149,188.50 USD** en un único conjunto de prueba Out-of-Time.

### El Trade-off del Negocio
En la industria del retail y pasarelas de pago, optimizar para métricas tradicionales como el F1-Score suele destruir valor debido a la asimetría de los costos:
* **El Costo del Fraude (Falso Negativo):** Pérdida del producto físico más multas bancarias por contracargo (Pérdida >100%, estimada en $75.00).
* **El Costo de la Fricción (Falso Positivo):** Pérdida del margen de ganancia y frustración de un cliente legítimo (Costo de oportunidad, margen aprox. 10%, estimado en $6.85).

Este motor evalúa las probabilidades predictivas frente al impacto real en el P&L (Estado de Resultados), desplazando el umbral de decisión de un arbitrario `0.50` a un **`0.5859`** matemáticamente óptimo, maximizando el dinero retenido.

### Arquitectura y MLOps
1. **Ingeniería de Variables Avanzada:** Target Encoding para variables categóricas de alta cardinalidad y Extracción de Tiempo Cíclico para firmas de comportamiento.
2. **Reducción de Dimensionalidad Algorítmica:** Aplicación del Método del Codo sobre la ganancia de información acumulada de LightGBM, reduciendo el espacio de ~400 a **67 variables élite** para garantizar una latencia de inferencia ultrabaja.
3. **Validación Out-of-Time (OOT):** Prevención de fuga de datos (Data Leakage) ordenando el dataset cronológicamente y reservando el último 20% (el futuro) como prueba estricta para simular condiciones reales de producción.
4. **XGBoost Sensible al Costo:** Entrenamiento con `scale_pos_weight` para el desbalanceo extremo y serialización en el formato nativo JSON de XGBoost para compatibilidad de despliegue en C++/Java.

### Resultados Clave
* **ROC-AUC Score:** `0.9027` (Validación Out-of-Time).
* **Umbral Financiero Óptimo:** `0.5859`
* **Impacto Financiero Neto:** **+$149,188.50 USD** de ahorro en una sola ventana de validación.

---
## Repository Structure / Estructura del Repositorio
* `/notebooks/`
  * `01_EDA_and_Data_Cleaning.ipynb`: Initial data ingestion and analysis. / *Ingesta inicial y análisis de datos.*
  * `02_Feature_Engineering_and_Selection.ipynb`: Hybrid encoding and dimensionality reduction. / *Codificación híbrida y reducción de dimensionalidad.*
  * `03_Model_Training_and_Evaluation.ipynb`: Out-of-Time validation, XGBoost training, and Cost-Sensitive optimization. / *Validación Out-of-Time, entrenamiento XGBoost y optimización financiera.*
* `/models/`: Contains the serialized `.json` model and `.joblib` metadata for API deployment. / *Contiene el modelo `.json` serializado y la metadata `.joblib` listos para producción.*
* `requirements.txt`: Python environment dependencies. / *Dependencias del entorno.*
* `README.md`: Project documentation and executive summary. / *Documentación del proyecto y resumen ejecutivo.*

## How to Run / Cómo Ejecutar
1. Clone the repository / Clona el repositorio.
2. Install dependencies / Instala las dependencias: `pip install -r requirements.txt`
3. Load the model / Carga el modelo: `xgboost.Booster()` with `xgboost_fraud_model.json`.
