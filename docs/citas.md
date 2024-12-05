
\subsection{Características de los participantes}
Finalmente, las características utilizadas para el proyecto incluyeron las siguientes datos:
\begin{itemize}
    \item Un \textbf{ID ficticio y único} para identificar a cada estudiante.
    \item Su \textbf{programa de estudio}, es decir, la carrera a la que pertenece.
    \item La \textbf{cantidad de intentos exitosos y fallidos} registrados en los envíos de ejercicios realizados durante el semestre.
    \item Las \textbf{notas de las cuatro pruebas solemnes} impartidas a lo largo del semestre.
    \item Los \textbf{puntajes asociados al nivel de abstracción, reconocimiento de patrones, descomposición y algoritmia} obtenidos en la prueba diagnóstica.
    \item La \textbf{interacción con los ejercicios realizados por el estudiante}. Esta interacción se representó mediante características binarias para cada ejercicio, donde se asignó un valor de 1 si el estudiante realizó el ejercicio, y 0 si no lo realizó. Este enfoque permitió analizar cómo la interacción con los ejercicios influyó en la aprobación o reprobación de la asignatura.
\end{itemize}



\subsection{Recopilación de datos}

Es importante destacar que todos los datos recopilados fueron entregados y autorizados por el jefe de carrera de Ingeniería Civil Informática del campus Antonio Varas de la Universidad Andrés Bello, garantizando en todo momento la integridad y confidencialidad de la información de los estudiantes.













\section{Resultados}
Actualmente, se han desarrollado cinco modelos de recomendación de diferentes aspectos para encontrar el mejor para implementarlo como el sistema recomendador de ejercicios para los estudiantes. Se esta utilizando el mismo dataset de datos para cada modelo, en cierto sentido, solo cambia la forma de según el modelo que se este utilizando.

los modelos como el que implementa la libreria Surprise \cite{Hug2020}, se organizaron los registros de ejercicios realizados por cada usuario en formato de ranking.  
- tablas y resultados
dos modelos con el objetivo de encontrar el mejor para la implementación en un sistema de recomendación de ejercicios. En ambos modelos, se utilizaron los mismos conjuntos de datos, los cuales fueron previamente tratados para obtener un dataset de mayor calidad. Este dataset incluye estudiantes de diferentes carreras que hayan aprobado las solemnes de la asignatura, ya que el conjunto de datos recopila información relevante sobre ejercicios que ayudan a los estudiantes a practicar de forma autodidacta para cuatro solemnes dentro de la asignatura.

En el conjunto de datos de los ejercicios, se creó una variable llamada Score, que se generaba a partir de las características en formato one-hot de cada ejercicio, creando así un puntaje de 12 bits para cada uno. Debido a esto, métricas como RMSE y MAE arrojaron resultados muy altos al evaluar los resultados obtenidos. Sin embargo, esto se puede entender gracias a los rangos de puntaje que tienen cada conjunto de ejercicios por hitos. A medida que se avanza en los hitos, se incrementa el puntaje de ese conjunto de ejercicios perteneciente al hito correspondiente.

Para el desarrollo del primer modelo, se implementó la librería Surprise \cite{Hug2020}, donde, asignando como puntaje del ranking el valor del score del ejercicio realizado. Para el entrenamiento y testeo del modelo se utilizó el algoritmo SVD basado en matrices de factorización. Al normalizar el puntaje, se obtuvo un RMSE de 256.8664. Cabe aclarar que, normalizando o no el puntaje, el modelo sigue recomendando un conjunto de ejercicios similares o desordenados, pero en general, esto no afecta las recomendaciones.

El segundo modelo, desarrollado con la librería TensorFlow, se encarga de predecir, según las características de los ejercicios realizados por un usuario en formato de ranking, utilizando las variables one-hot de los ejercicios para generar o intentar predecir un puntaje ficticio de preferencia del usuario para un ejercicio determinado. Es decir, busca predecir cuáles serían los ejercicios más adecuados según el conocimiento previo del usuario para que pueda realizarlos.








\section{Experimentos}

\begin{itemize}
    \item El número total de participantes en cada grupo y en cada etapa del estudio.
    \item El flujo de participantes a través de cada etapa del estudio.
    \item Proporcione las fechas que definan los períodos de reclutamiento, medidas repetidas o seguimiento.
    \item Proporcione información detallada sobre los métodos estadísticos y analíticos utilizados
    \begin{itemize}
        \item Datos faltantes (frecuencia o porcentajes de datos faltantes).
        \item Evidencia empírica y/o argumentos teóricos sobre las causas de los datos faltantes.
        \item Métodos utilizados para tratar los datos faltantes, si los hubo.
        \item Descripciones de cada resultado principal y secundario, incluyendo el tamaño total de la muestra y de cada subgrupo, así como el número de casos, medias de celda, desviaciones estándar y otras medidas que caracterizan los datos utilizados.
    \end{itemize}
    \item Diferenciación clara entre las hipótesis principales y sus pruebas-estimaciones, las hipótesis secundarias y sus pruebas-estimaciones, y las hipótesis exploratorias y sus pruebas-estimaciones.
\end{itemize}




// RESSULTADOS



\begin{itemize}
    \item Análisis complejos, por ejemplo, modelado de ecuaciones estructurales, modelos lineales jerárquicos, análisis factoriales, análisis multivariados, entre otros, incluyendo:
    \begin{itemize}
        \item Detalles de los modelos estimados.
        \item Matrices de varianza-covarianza (o correlación) asociadas.
        \item Identificación del software estadístico utilizado para realizar los análisis (por ejemplo, SAS PROC GLM o el paquete específico de R).
    \end{itemize}
    \item Problemas de estimación (por ejemplo, falta de convergencia, espacios de solución defectuosos), diagnósticos de regresión o anomalías analíticas detectadas y soluciones a estos problemas.
    \item Otros análisis de datos realizados, incluyendo análisis ajustados, indicando cuáles fueron planeados y cuáles no (aunque no necesariamente con el nivel de detalle de los análisis principales).
    \item Informe sobre cualquier problema con los supuestos estadísticos y/o distribuciones de datos que puedan afectar la validez de los hallazgos.
\end{itemize}





Objetivo:
    Desarrollar e implementar un sistema de recomendación de ejercicios de programación para estudiantes universitarios que aborde la sobreabundancia de recursos, la falta de orientación personalizada y la diversidad de niveles de habilidad, con el propósito de mejorar la experiencia de aprendizaje, fomentar un enfoque más efectivo en la resolución de ejercicios y contribuir al crecimiento e innovación en el ámbito educativo. 

Metodos de estudio
Metodologia usada CRISP-DM para el analisis y preparacion de datos, desarrollo y evaluacion de modelos que cumplan con alguna solucion aceptable para el problema planteado de la investigacion. Este proyecto es de tipo experimental ya que se necesitara algunas similaciones o ayuda de alumnos de primer año para sacar la concluciones pertinenntes que demuestren que un estudio personalizado mejora el rendimineto academico de los estudiantes en la asignaturas.


problema -> bibliografia -> objetivos -> metodologia -> plan de trabajo -> resultados -> conclusion -> limitaciones -> trabajos futuros

\subsection{Objetivos}
Desarrollar e implementar un sistema de recomendación de ejercicios de programación para estudiantes de la Universidad Andrés Bello, con el objetivo de ofrecer una orientación personalizada que se adapte a la diversidad de niveles de habilidad de los estudiantes y ayude a mejorar sus habilidades de programación de forma personalizada y efectiva.
\begin{itemize}
\item Implementar algoritmos de recomendación que personalicen las sugerencias de ejercicios de programación según el nivel de habilidad, preferencias individuales y objetivos de aprendizaje de cada estudiante.

\item Desarrollar mecanismos de evaluación efectivos para determinar el conocimiento previo de cada estudiante, permitiendo recomendaciones precisas y adaptadas a sus necesidades específicas.

\item Asegurar que el sistema recomiende ejercicios que abarquen una variedad de lenguajes de programación y tecnologías, garantizando relevancia para estudiantes con diferentes áreas de enfoque.
\end{itemize}









Los sistemas de recomendación de modelos de dos torres (Two-Tower Models) son una arquitectura de aprendizaje profundo diseñada específicamente para abordar problemas de recomendaciones personalizadas. Estos modelos son populares por su capacidad para procesar grandes volúmenes de datos y capturar relaciones complejas entre usuarios y elementos. A continuación, te describo los aspectos más importantes de este enfoque:

Concepto básico
El modelo de dos torres utiliza dos redes neuronales independientes, conocidas como "torres", que representan al usuario y al ítem (producto, servicio, contenido, etc.). El objetivo principal es aprender representaciones (embeddings) para usuarios e ítems en un espacio compartido, donde la similitud en este espacio indica afinidad o preferencia.

Arquitectura del modelo
Torre del usuario:

Toma características relacionadas con el usuario, como:
Datos demográficos: edad, género, ubicación.
Historial: comportamiento previo (por ejemplo, clics o compras).
Contexto: dispositivo, hora del día, etc.
Procesa estos datos a través de una red neuronal (MLP, LSTM, CNN, etc.) para generar un vector de embedding del usuario.
Torre del ítem:

Toma características del ítem, como:
Atributos del producto: categoría, precio, descripción.
Datos contextuales: popularidad, reseñas, etiquetas.
Representaciones pre-entrenadas: embeddings de texto o imágenes.
También usa una red neuronal para generar un vector de embedding del ítem.
Cálculo de afinidad:

Los embeddings de usuario e ítem se combinan utilizando una métrica de similitud, como el producto escalar o la distancia coseno.
Este valor representa qué tan bien encaja el ítem con las preferencias del usuario.
Entrenamiento
El modelo aprende ajustando los pesos de ambas torres para maximizar la correspondencia entre usuarios e ítems preferidos. Se entrena utilizando un objetivo de clasificación binaria o ranking, donde el modelo predice:

Si un usuario interactuará con un ítem específico (clic, compra, etc.).
La posición relativa de un ítem en una lista ordenada por relevancia.
Funciones de pérdida comunes:
Binary Cross-Entropy (BCE):
Se usa para predecir probabilidades de interacción.
Ranking Loss (Margin Loss):
Optimiza el ranking relativo de ítems relevantes frente a ítems no relevantes.
Softmax o Triplet Loss:
Asegura que los ítems relevantes estén más cerca del usuario en el espacio embedding.
Ventajas del modelo de dos torres
Escalabilidad:

Una vez entrenado, el modelo puede generar embeddings para usuarios e ítems de forma independiente, lo que reduce el costo computacional durante la inferencia.
Facilita búsquedas eficientes en grandes catálogos mediante técnicas como Approximate Nearest Neighbors (ANN).
Flexibilidad:

Permite incluir diferentes tipos de datos (estructurados, no estructurados) para representar usuarios e ítems.
Soporta actualizaciones rápidas: puedes generar embeddings para nuevos ítems sin necesidad de reentrenar el modelo completo.
Modularidad:

Las dos torres pueden ser personalizadas o pre-entrenadas, lo que mejora la capacidad de generalización y permite la transferencia de conocimiento.
Desafíos
Cold Start:
Es difícil manejar usuarios o ítems nuevos si no tienen suficientes datos asociados.
Desbalance en los datos:
Las interacciones positivas suelen ser mucho menos frecuentes que las negativas, lo que puede sesgar el modelo.
Capacidad de representación:
Si los embeddings no capturan correctamente las relaciones entre características, el rendimiento se ve afectado.
Casos de uso
Recomendación de productos:
En comercio electrónico (Amazon, Mercado Libre) para sugerir productos relevantes.
Recomendación de contenido:
Plataformas de streaming (Netflix, YouTube) para recomendar películas, series o videos.
Emparejamiento en redes sociales:
Sugerencias de amigos, conexiones o citas (LinkedIn, Tinder).
Publicidad personalizada:
Sistemas de anuncios que muestran contenido relevante según el perfil del usuario.
Implementación práctica
Frameworks y herramientas:

TensorFlow Recommenders (TFRS): Soporte específico para modelos de dos torres.
PyTorch: Flexibilidad para construir la arquitectura desde cero.
ScaNN: Para realizar búsquedas ANN de manera eficiente.
Pasos clave:

Preprocesamiento: Preparar y limpiar datos de usuarios e ítems.
Feature engineering: Crear representaciones relevantes para ambas torres.
Entrenamiento: Usar un conjunto de datos con interacciones etiquetadas.
Evaluación: Medir métricas como precisión, recall o NDCG.
Evoluciones del modelo
Incorporación de aprendizaje contrastivo para mejorar los embeddings.
Uso de arquitecturas basadas en Transformers para capturar relaciones más complejas entre usuarios e ítems.
Implementación de multi-task learning para entrenar el modelo en múltiples objetivos relacionados (como clics, compras, duración de interacción).
¿Te interesa explorar un caso práctico con datos específicos? 😊











Aquí está mi imagen:
\begin{figure}[h] % 'h' posiciona la imagen aquí
    \centering
    \includegraphics[width=0.5\textwidth]{imagenes/off-policy.JPG} % Ajusta el tamaño
    \caption{Este es un ejemplo de imagen.}
    \label{fig:ejemplo}
\end{figure}
