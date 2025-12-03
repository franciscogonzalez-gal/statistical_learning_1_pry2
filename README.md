# Proyecto 2 - Statistical Learning I: Clasificación de Incumplimiento de Préstamos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/franciscogonzalez-gal/statistical_learning_1_pry2/blob/main/proyecto_2_clasificacion.ipynb)

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación para predecir el incumplimiento de préstamos utilizando técnicas de aprendizaje automático supervisado. El objetivo principal es construir y comparar diferentes modelos de clasificación que puedan predecir si un solicitante de préstamo incumplirá con sus pagos (`defaulted`).

**Autor:** Francisco González  
**Carnet:** 24002914  
**Curso:** Statistical Learning I

## 🎯 Objetivo

Desarrollar modelos de clasificación que permitan predecir el riesgo de incumplimiento de préstamos basándose en características demográficas, financieras y de historial crediticio de los solicitantes.

## 📊 Dataset

El proyecto utiliza un conjunto de datos de solicitudes de préstamos que incluye las siguientes variables:

### Variables del Dataset

- **id**: Identificador único del solicitante
- **age**: Edad del solicitante
- **income**: Ingresos del solicitante
- **credit_score**: Puntaje de crédito
- **loan_amount**: Monto del préstamo solicitado
- **loan_term_months**: Plazo del préstamo en meses
- **employment_status**: Estado laboral
- **marital_status**: Estado civil
- **num_dependents**: Número de dependientes
- **education_level**: Nivel de educación
- **home_ownership**: Tipo de propiedad de vivienda
- **city**: Ciudad de residencia
- **application_date**: Fecha de solicitud
- **savings_balance**: Balance de ahorros
- **checking_balance**: Balance de cuenta corriente
- **defaulted**: Variable objetivo (0 = No incumplió, 1 = Incumplió)

## 🛠️ Tecnologías y Librerías Utilizadas

El proyecto está desarrollado en Python y utiliza las siguientes librerías principales:

```python
# Análisis de datos
- pandas
- numpy

# Visualización
- matplotlib
- seaborn

# Machine Learning
- scikit-learn
  - LogisticRegression
  - GaussianNB
  - LinearSVC (SVM)
  - RandomForestClassifier
  - GridSearchCV
  - Pipeline
  - ColumnTransformer

# Persistencia
- joblib

# Entorno
- Google Colab
```

## 📁 Estructura del Proyecto

```
statistical_learning_1_pry2/
│
├── proyecto_2_clasificacion.ipynb  # Notebook principal con todo el análisis
├── LICENSE                          # Licencia CC0 1.0 Universal
└── README.md                        # Este archivo
```

## 🔍 Metodología

El proyecto sigue un flujo de trabajo estructurado de ciencia de datos:

### 1. Importación de Librerías
Carga de todas las dependencias necesarias para el análisis.

### 2. Carga de Datos
Importación del dataset desde Google Drive.

### 3. Definición de Variables
Identificación y documentación de todas las variables del dataset.

### 4. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones
- Identificación de valores faltantes
- Análisis de correlaciones
- Visualizaciones estadísticas
- Detección de desbalance de clases

### 5. Limpieza de Datos e Imputación
- Tratamiento de valores faltantes
- Manejo de outliers
- Estandarización de formatos

### 6. Ingeniería de Características
- Creación de pipeline de preprocesamiento
- Transformación de variables categóricas (One-Hot Encoding)
- Escalado de variables numéricas (StandardScaler)
- Uso de `ColumnTransformer` para procesar diferentes tipos de variables

### 7. Separación de Datos
División del dataset en conjuntos de entrenamiento y prueba.

### 8. Definición de Modelos y Pipelines
Configuración de cuatro modelos de clasificación:

#### a) Regresión Logística
- Solver: SAGA
- Penalización: L1 y L2 (búsqueda de hiperparámetros)
- Class weight: balanced
- Parámetro C: [0.01, 0.1, 1, 10]

#### b) Naive Bayes Gaussiano
- Modelo probabilístico base
- Sin hiperparámetros a optimizar

#### c) SVM Lineal
- LinearSVC con calibración de probabilidades
- Dual: False
- Class weight: balanced
- Parámetro C: búsqueda con GridSearchCV

#### d) Random Forest
- Conjunto de árboles de decisión
- Optimización de hiperparámetros:
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf

### 9. Entrenamiento y Validación
- Uso de `GridSearchCV` para búsqueda de hiperparámetros
- Validación cruzada durante el entrenamiento
- Selección de los mejores modelos

### 10. Evaluación en Conjunto de Prueba
Evaluación exhaustiva de cada modelo con:
- **Matriz de confusión**: Visualización de predicciones correctas e incorrectas
- **Métricas de clasificación**:
  - Precision
  - Recall
  - F1-Score
  - Accuracy
- **Curva ROC**: Análisis del trade-off entre tasa de verdaderos positivos y falsos positivos
- **AUC**: Área bajo la curva ROC

### 11. Persistencia de Modelos
Almacenamiento de los modelos entrenados usando `joblib` para uso futuro.

## 🚀 Cómo Usar Este Proyecto

### Opción 1: Google Colab (Recomendado)
1. Haz clic en el badge "Open in Colab" al inicio de este README
2. El notebook se abrirá en Google Colab
3. Asegúrate de tener el dataset en tu Google Drive
4. Ejecuta las celdas secuencialmente

### Opción 2: Entorno Local

#### Requisitos Previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

#### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/franciscogonzalez-gal/statistical_learning_1_pry2.git
cd statistical_learning_1_pry2

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

#### Ejecutar el Notebook

```bash
# Iniciar Jupyter Notebook
jupyter notebook proyecto_2_clasificacion.ipynb
```

**Nota:** Deberás modificar las rutas de acceso a los datos según tu configuración local.

## 📈 Resultados Principales

El proyecto implementa y compara cuatro modelos de clasificación:

1. **Regresión Logística**: Modelo lineal con regularización L1/L2
2. **Naive Bayes Gaussiano**: Modelo probabilístico basado en el teorema de Bayes
3. **SVM Lineal**: Clasificador de máxima margen con calibración de probabilidades
4. **Random Forest**: Ensemble de árboles de decisión

Cada modelo se evalúa en términos de:
- Capacidad de predicción (Accuracy)
- Balance entre Precision y Recall
- Capacidad de discriminación (AUC-ROC)
- Matriz de confusión

## 🔄 Pipeline de Preprocesamiento

El proyecto implementa un pipeline automatizado que:

1. **Variables Numéricas**:
   - Imputa valores faltantes con la mediana
   - Aplica estandarización (StandardScaler)

2. **Variables Categóricas**:
   - Imputa valores faltantes con el valor más frecuente
   - Aplica One-Hot Encoding

Este pipeline garantiza que el preprocesamiento se aplique de manera consistente tanto en entrenamiento como en predicción.

## 📝 Conclusiones y Recomendaciones

El análisis completo de conclusiones y recomendaciones se encuentra en la sección 11 del notebook. Se recomienda revisar:
- Comparación de rendimiento entre modelos
- Análisis de características más importantes
- Recomendaciones para implementación práctica
- Posibles mejoras futuras

## 📄 Licencia

Este proyecto está licenciado bajo [CC0 1.0 Universal](LICENSE) - Dominio Público.

Puedes copiar, modificar, distribuir y ejecutar el trabajo, incluso para propósitos comerciales, sin pedir permiso.

## 🤝 Contribuciones

Este proyecto es parte de un trabajo académico. Si deseas contribuir o tienes sugerencias, siéntete libre de abrir un issue o pull request.

## 📧 Contacto

**Francisco González**  
Carnet: 24002914

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!
