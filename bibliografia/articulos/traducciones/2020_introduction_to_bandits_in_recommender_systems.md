Introducción a los Bandits en Sistemas de Recomendación

RESUMEN
El problema del bandido multi-brazo modela a un agente que simultáneamente intenta adquirir nuevo conocimiento (exploración) y optimizar sus decisiones basado en el conocimiento existente (explotación). El agente intenta balancear estas tareas competidoras para maximizar su valor total durante el periodo considerado. Hay muchas aplicaciones prácticas de los algoritmos de bandit, incluyendo ensayos clínicos, enrutamiento adaptativo o diseño de carteras. Durante la última década ha habido un interés creciente en el desarrollo de algoritmos de bandit para abordar problemas específicos en sistemas de recomendación, como la mejora de la recomendación de productos, el problema de inicio en frío o la personalización. El objetivo de este tutorial es proporcionar una breve introducción al problema del bandit con un panorama de las diversas aplicaciones de los algoritmos de bandit en recomendación.

1. INTRODUCCIÓN
El problema del bandido multi-brazo (MAB) modela a un agente que simultáneamente intenta adquirir nuevo conocimiento (exploración) y optimizar sus decisiones basado en el conocimiento existente (explotación). El agente intenta balancear estas tareas competidoras para maximizar su valor total durante el periodo considerado. Hay muchas aplicaciones prácticas del modelo de bandit, como ensayos clínicos, enrutamiento adaptativo o diseño de carteras. Durante la última década ha habido un creciente interés en el desarrollo de algoritmos de bandit para problemas específicos en sistemas de recomendación (RS), tales como recomendación de noticias y anuncios, el problema de inicio en frío en recomendación, personalización, filtrado colaborativo con bandits o la combinación de redes sociales con bandits para mejorar la recomendación de productos. El objetivo de este tutorial es proporcionar a los participantes el conocimiento básico de los siguientes conceptos: (a) el dilema de exploración-explotación y su conexión con el aprendizaje mediante interacción; (b) la formulación del problema de RS como una tarea interactiva de decisión secuencial que necesita equilibrar la exploración y la explotación; (c) fundamentos básicos detrás de los enfoques de bandit que abordan el dilema de exploración-explotación; y (d) una visión general del estado del arte de los RS basados en bandit. Con este tutorial esperamos capacitar a los participantes para comenzar a trabajar en RS basados en bandit y proporcionar un marco que les permita desarrollar enfoques más avanzados. Este tutorial sigue una serie de tutoriales sobre temas similares que tuvieron lugar en años recientes [17, 19].

Este tutorial introductorio está dirigido a una audiencia con formación en informática, recuperación de información o RS que tenga un interés general en la aplicación de técnicas de aprendizaje automático en RS. El conocimiento previo necesario incluye una familiaridad básica con el aprendizaje automático y conocimientos básicos de estadística y teoría de probabilidades. El tutorial proporcionará ejemplos prácticos basados en código Python y Jupyter Notebooks.

2. VISIÓN GENERAL DEL TUTORIAL
El tutorial está dividido en tres secciones enfocadas en: (1) motivación general e introducción a los enfoques clásicos de bandit; (2) sesión práctica donde se utilizará una tarea simple de recomendación sintética que representa un problema de bandit con recompensas lineales; y (3) visión general de una variedad de aplicaciones de algoritmos de bandit en sistemas de recomendación, resumiendo el estado actual y delineando los desafíos de aplicar algoritmos de bandit en sistemas de recomendación.

Las siguientes secciones describen con más detalle los temas cubiertos en el tutorial.

2.1 Introducción a los Enfoques Clásicos de Bandit
Esta sección proporciona una introducción a los conceptos fundamentales necesarios para entender los enfoques clásicos/básicos de bandit. Las suposiciones subyacentes e intuiciones detrás de estos enfoques clásicos sirven como base esencial para comprender cómo se aplican las ideas de bandit en los sistemas de recomendación (RS).

Motivación: Introducir el dilema de exploración-explotación y su relevancia en los sistemas de recomendación [6, 18]. Discutir casos de uso de recomendación basados en bandit y aplicaciones del mundo real [1, 20, 29, 30, 37, 41].

Introducción a los Bandidos de Múltiples Brazos (MAB) Clásicos: Describir el problema MAB estocástico y sus suposiciones [36]. Introducir enfoques clásicos de bandit, incluyendo e-greedy [5, 36], Upper Confidence Bound (UCB) [5, 27] y Thompson Sampling [11]. Los objetivos principales de esta parte del tutorial son:

- Introducir el concepto de arrepentimiento (regret) y recompensa.
- Discutir el impacto del uso de diferentes nociones de incertidumbre para definir un enfoque de bandit, como exploración no guiada/ingenua (por ejemplo, e-greedy) o exploración guiada (por ejemplo, UCB).
- Resaltar mejoras a los enfoques clásicos que abordan el problema MAB estocástico (por ejemplo, e-first [5], e-decreasing [5, 36], variaciones de UCB [5, 27]).

Bandidos y Aprendizaje por Refuerzo: Introducción al Aprendizaje por Refuerzo (RL) [36] y su conexión con los algoritmos de bandit [8]. Introducción a los Entornos como representación de la tarea (en este caso, el problema MAB estocástico) que debe ser resuelta por el Agente (también conocido como algoritmo de bandit o sistema de recomendación) [7, 36].

Variaciones de Bandit: Destacar diferentes variaciones del problema MAB estocástico y por qué existen, por ejemplo, bandidos de múltiples jugadas [33], bandidos adversariales [10] y bandidos contextuales [30].

2.2 Sesión Práctica
Los participantes tendrán la oportunidad de probar en un entorno práctico diferentes configuraciones de algoritmos de bandit y observar los resultados. Durante la sección práctica, se utilizará el framework BEARS [8] y se proporcionarán Jupyter Notebooks. BEARS es un framework de Python de código abierto que tiene como objetivo proporcionar un código limpio y bien documentado para ser utilizado en entornos académicos/investigativos, permitiendo evaluaciones reproducibles de sistemas de recomendación basados en bandit.

- Introducción a los Bandidos Contextuales: Los bandidos estocásticos sin contexto (clásicos) no pueden representar completamente las complejidades del problema de RS. Una variación importante de MAB fue la introducción de bandidos contextuales y la intuición de recompensas lineales con respecto a datos contextuales. El tutorial enfatiza los bandidos contextuales y discute su aplicación en sistemas de recomendación, como se ejemplifica con la recomendación de noticias de Yahoo utilizando LinUCB [30].

- Ejercicio Práctico:
  - Descripción de un entorno sintético de sistemas de recomendación con recompensas lineales.
  - Introducción a BEARS [8] y configuración de experimentos (componentes de Entorno, Agente, Evaluador y Experimento). Se proporcionará un Jupyter Notebook para este ejercicio.
  - Permitir que los participantes interactúen con el Jupyter Notebook. BEARS permitirá a los participantes experimentar con diferentes algoritmos de bandit (y diferentes valores de parámetros), diferentes configuraciones de experimentos (por ejemplo, cambiando el número de iteraciones/horizonte y episodios/ejecuciones) y diferentes configuraciones de métricas (por ejemplo, recompensa, recompensa acumulada, arrepentimiento).
  - Discusión sobre cualquier observación o hallazgo que los participantes hayan tenido con respecto al ejercicio. Por ejemplo, cómo configurar los valores de exploración muy altos o muy bajos tuvo un impacto en los resultados.
  - Resaltar la importancia de compartir todos los detalles de los experimentos para evaluaciones reproducibles (por ejemplo, semillas aleatorias).

- Desafíos de Evaluación: Para el ejercicio, se desarrolló un entorno sintético. Sin embargo, crear entornos para RS (por ejemplo, basados en un conjunto de datos) para lograr evaluaciones no sesgadas es un desafío [7]. Los presentadores tienen como objetivo revisar brevemente la importancia de los temas relacionados con el sesgo al tratar con valores faltantes y revisar soluciones que se han propuesto (por ejemplo, muestreo de rechazo [30, 31] y razonamiento contrafactual [9, 22]).

2.3 Bandidos en Sistemas de Recomendación
El objetivo es proporcionar un panorama de soluciones representativas existentes para mostrar cómo se utilizan los MAB en sistemas de recomendación. A lo largo de la presentación, proporcionaremos explicaciones basadas en los componentes generales previamente introducidos, relacionados con el marco de RL y BEARS. De esta manera, los participantes tendrán un marco general para entender la variedad de soluciones. Esta parte del tutorial se centrará en la aplicación de algoritmos de bandit en diversas áreas de sistemas de recomendación, así como en problemas relacionados con la implementación, escalabilidad, entrenamiento y manejo de tipos específicos de datos (por ejemplo, anuncios, artículos de periódico, multimedia, etc.) [18]. Los temas cubiertos incluyen las siguientes aplicaciones de algoritmos de bandit en el contexto de sistemas de recomendación:

- El problema de inicio en frío [38]
- Redes sociales y sistemas de recomendación [15, 16]
- Filtrado colaborativo y factorización de matrices en sistemas de recomendación [21, 32, 40, 42]
- Recomendación con vida útil limitada [28, 39]
- Recomendación de noticias [30, 31]
- Publicidad en línea [11]
- Recomendación y recuperación de multimedia: imágenes, música, video [23, 26, 41]
- Bandidos clasificados y recomendación de listas [25, 28, 33]
- Ejemplos de sistemas interactivos de recuperación y recomendación basados en algoritmos de bandit [3, 14, 20, 34]
- Personalización y optimización del sistema [2, 4, 12, 13, 24, 35]

Se pueden encontrar más referencias sobre algoritmos de bandit en sistemas de recomendación en [18].

3. MATERIALES DEL TUTORIAL
Todos los materiales, incluyendo diapositivas y código, estarán disponibles después del tutorial en un repositorio público3.
