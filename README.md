# Fraud-Detection
Il modello proposto in questo progetto ha lo scopo di predire le frodi all'interno di un contesto specifico, come transazioni finanziarie, acquisti online o interazioni tra clienti e fornitori.
<br>Utilizzando un approccio basato su tecniche di machine learning, il modello analizza variabili e indicatori storici per identificare comportamenti sospetti e anomalie nei dati, permettendo così un intervento tempestivo e mirato.

<p align="center">
    <img src="https://i.imgur.com/zSNY5tM.png" alt="hacker" height="400"/>
</p>

## Ambiente di sviluppo
L'intero progetto è stato sviluppato su ambiente linux (Ubuntu 22.04.5), il che ha permesso di lavorare interamente con GPU NVIDIA ove possibile.
<br>La maggior parte degli algoritmi usati dunque nel progetto ha sfruttato la potenza di calcolo della scheda grafica NVIDIA (RTX 3060) e qual'ora, dunque si volesse _runnare_ il progetto in locale si dovrebbe avere a disposizione un ambiente che lo possa permettere.
<br>In alternativa sarebbe possibile covertire il codice, per farlo girare su CPU (sconsigliato).
<br>Per fare ció, basterebbe installare le seguenti librerie ed sostituire gli eventuali usi di cuML e Dask ML con algoritmi della libreria scikit-learn (CPU based).
```
numpy
seaborn
scikit-learn
matplotlib
pandas
inflection
pyswip==0.2.09
kneed
```

## Set up ambiente di lavoro - LINUX

1. Bisogna aver installato e configurato correttamente i supporti [CUDA](https://developer.nvidia.com/cuda-12-3-0-download-archive) (nel mio caso 12.3) e [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (nel mio caso 8.9) per usufruire della potenza di calcolo della propria GPU NVIDIA.
2. Clonare le repository:
```
git clone https://github.com/Vincy02/Fraud-Detection
cd Fraud-Detection
```
3. Creazione ambiente di lavoro che supporti GPU NVIDIA:
```
conda create -n fraud_detection -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.10 python=3.12 'cuda-version>=12.0,<=12.5'
```
4. Attivazione ambiente di lavoro:
```
conda activate fraud_detection
```
5. Installare CMake:
```
sudo apt-get -y install cmake
```
6. Installre SWI-Prolog: si puó seguire [documentazione ufficiale](https://www.swi-prolog.org/build/unix.html)
7. Installare librerie OS essenziali:
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev
sudo apt-get install libopenblas-dev liblapack-dev -y
```
8. Installare i requirements per eseguire il progetto:
```
pip install -r requirements.txt
```
9. Se eventualmente il processo di installazione dei requirements fallisce è possibile scaricare le libreria citate nel paragrafo precedente + ```dask-ml```.<br>
Per eventualmente, osservare le prestazioni della GPU è possibile scaricare la libreria ```nvitop```.
10. Installare e spostare nella cartella root del progetto la cartella _./Data_ dal seguente link: [LINK](https://mega.nz/folder/oow1CAZC#dfN2ThUn8A8jCii7q9526Q)
