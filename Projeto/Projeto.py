import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# (a) Carrega o arquivo de audio formato wave
samplerate, amostras = wavfile.read('music.wav')

# Este bloco de códigos abaixo corresponde a gerar e plotar o sinal de audio em tempo continuo

# amostras: corresponde a quantidade de amostras colhidas no sinal de audio
# samplerate: taxa de amostragem
time = np.arange(0.0, len(amostras) * 1/samplerate, 1/samplerate)
plt.figure(1)
plt.title("Sinal do áudio")
plt.plot(time, amostras)
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.grid()

# (b) Este bloco de códigos abaixo corresponde a
# gerar a FFT e plotar o espectro
# N: quantidade de amostras
N = len(amostras)
# T: corresponde ao período de amostragem
T = 1.0 / samplerate

# algoritmo fft com ajustes para nosso projeto (documentacao scipy)
x = np.linspace(0.0, N*T, N, endpoint=False) #np.linspace(inicio do intervalo, fim do intervalo, número de amostras)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.figure(2)
plt.title("Espectro FFT")
plt.xlabel('Frequencia [rad/s]')
plt.ylabel('Amplitude do Espectro')
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()

# (c)Carrega os coeficientes utilizando a biblioteca pyfda para plotagem
fir = np.genfromtxt('fir.csv', delimiter=',')

# Plota coeficientes do filtro FIR, o sinal de audio e o espectro em frequencia
plt.figure(3)
plt.title("Sinal Discreto - Coeficientes")
plt.xlabel('Amostras [n]')
plt.ylabel('Amplitude do Sinal')
plt.stem(fir)
plt.show()

# (d)Este bloco de código realiza a convolução
tam = len(fir)
nSeq = range(tam)
#Inicializando saída
ySeq = np.zeros(tam)

#Soma de convolução
for n in nSeq:
    aux = 0
    for k in range(tam):
        aux = aux + fir[k] * fir[n - k]
    ySeq[n] = aux

#Plota o sinal filtrado para convolução
plt.figure(4)
plt.xlabel("Amostras [n]")
plt.title("Sequência")
plt.stem(nSeq, fir)
plt.grid()

# Plota o resultado da convolução
plt.figure(5)
plt.xlabel("Amostras [n]")
plt.title("Sequência y[n]")
plt.stem(nSeq, ySeq)
plt.grid()



# (e) Dizimação M
# Soma de convolução
M = 2
tam = len(fir)
tamD = 0

# Verificando tamanho da lista com dizimação em 2
for i in range(0, tam, M):
   tamD = tamD + 1

# Inicializando nSeqD e ySeqD
nSeqD = range(tamD)
ySeqD = []

# Passando elementos com fator M para ySeqD
for n in range(0, tam, M): 
    ySeqD.append(ySeq[n])


plt.figure(6)
plt.xlabel("Amostras [M*n]")
plt.title("Sequência y[M*n]")
plt.stem(nSeqD, ySeqD)
plt.grid()

