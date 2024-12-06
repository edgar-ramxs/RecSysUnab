# RecSysUnab: Sistema de recomendación de ejercicios de programación basado en un modelo de dos torres
+ Autor : Edgar Alejandro Ramos Vivenes
+ Institución : Universidad Andrés Bello, Chile. 

## Instalación

### 1. Clonar repositorio
```shell
git clone https://github.com/edgar-ramxs/RecSysUnab.git ~/RecSysUnab
cd ~/RecSysUnab
code .
```

### 2. Instalar entorno virtual en Python (opcional)
+ Instalar entorno virtual
```shell
pip install virtualenv
```

+ Crear entorno virtual 
```shell
python -m venv .RecSysUnabVenv
```

+ Activar entorno virtual (Linux)
```shell
source .RecSysUnabVenv/bin/activate
```

### 3. Instalar paquetes 
+ Por medio del gestor de paquetes de Python (pip)
```shell
pip install numpy pandas matplotlib seaborn scikit-learn scikit-surprise tensorflow tensorflow-recommenders torch torchvision torchaudio
```

+ Por medio de instalación de paquetes especificos
```bash
pip install -r requirements.txt
```
