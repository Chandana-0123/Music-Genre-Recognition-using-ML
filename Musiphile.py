#In[1]:
import os
import subprocess
from numpy.core.fromnumeric import reshape
import sklearn
import librosa
import numpy
import matplotlib.pyplot
import sounddevice
import soundfile
import sys
import tensorflow
import pandas
import wave
import functools


#In[3]:
class Bank:
    def __init__(self):
        subprocess.call(['ffmpeg', '-i', sys.argv[1] , 'Test_Audio.wav'])
        self.Input = wave.open('Test_Audio.wav', 'rb')
        self.Input = self.Input.readframes(661794)
        self.Input = numpy.frombuffer(self.Input, numpy.short)
        self.FloatingInput = self.Input.astype(float)
       #Bank.Solo(self)
       #Bank.Collector(self)
        Divider.length(self)
        
    
    def Collector(self):
        self.duration = 10
        sounddevice.default.samplerate = 22500
        sounddevice.default.channels = 2
        self.recording = sounddevice.rec(int(self.duration*22500)) 
        sounddevice.play(self.recording, 22500, blocking= False)
        soundfile.write("Audio.wav", self.recording, 22500)

    def Solo(self):
        self.samplerate = 22500
        self.frquency = 600000
        self.duration = 25
        self.time_variable = numpy.linspace(0, self.duration, int(self.duration*22500), endpoint= False)
        self.generated = 0.5*numpy.sin(2*numpy.pi*self.frquency*self.time_variable)
        sounddevice.play(self.generated, 22500, blocking= False)
        soundfile.write("Solo.wav", self.generated, 22500)


#In[5]:
class Artist:
    def plotter(self):
        matplotlib.pyplot.figure(0)
        matplotlib.pyplot.title("Waveform")
        matplotlib.pyplot.plot(self.Input)
        matplotlib.pyplot.savefig('Given Music in Waveform.png',dpi = 180)
        matplotlib.pyplot.show()
        Artist.Spectogram_Plotter(self)
    
    def Spectogram_Plotter(self):
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.title("Spectogram")
        matplotlib.pyplot.specgram(self.Input)
        matplotlib.pyplot.savefig('Spectogram.png',dpi = 180)
        matplotlib.pyplot.show()
        
    def Chroma_STFT_Plotter(self):
        matplotlib.pyplot.figure(4)
        matplotlib.pyplot.title("Chroma STFT")
        matplotlib.pyplot.plot(self.temp1)
        matplotlib.pyplot.savefig('Chroma STFT.png',dpi = 180)
        matplotlib.pyplot.show()

    def MFCC_Plotter(self):
        matplotlib.pyplot.figure(5)
        matplotlib.pyplot.title("Mel Frequency Cepstral Coefficents")
        matplotlib.pyplot.plot(self.temp2)
        matplotlib.pyplot.savefig('MFCC.png',dpi = 180)
        matplotlib.pyplot.show()

    def RMSE_Plotter(self):
        matplotlib.pyplot.figure(6)
        matplotlib.pyplot.title("Root Mean Square")
        matplotlib.pyplot.plot(self.temp3)
        matplotlib.pyplot.savefig('RMS.png',dpi = 180)
        matplotlib.pyplot.show()

    def Spectral_Bandwidth_Plotter(self):
        matplotlib.pyplot.figure(7)
        matplotlib.pyplot.title("Spectral Bandwidth")
        matplotlib.pyplot.plot(self.temp4)
        matplotlib.pyplot.savefig('Spectral Bandwidth.png',dpi = 180)
        matplotlib.pyplot.show()

    def Spectral_Rolloff_Plotter(self):
        matplotlib.pyplot.figure(9)
        matplotlib.pyplot.title("Spectral Rolloff")
        matplotlib.pyplot.plot(self.temp5)
        matplotlib.pyplot.savefig('Spectral Rolloff.png',dpi = 180)
        matplotlib.pyplot.show()

    def zcr_Plotter(self):
        matplotlib.pyplot.figure(11)
        matplotlib.pyplot.title("Zero Crossing Rate")
        matplotlib.pyplot.plot(self.temp6[10000:10500])
        matplotlib.pyplot.savefig('ZCR.png',dpi = 180)
        matplotlib.pyplot.show()
    
    def Harmony_Plotter(self):
        matplotlib.pyplot.figure(11)
        matplotlib.pyplot.title("Harmony")
        matplotlib.pyplot.plot(self.temp7)
        matplotlib.pyplot.savefig('Harmony.png',dpi = 180)
        matplotlib.pyplot.show()

    def Tempo_Plotter(self):
        matplotlib.pyplot.figure(11)
        matplotlib.pyplot.title("Tempo")
        matplotlib.pyplot.plot(self.temp8)
        matplotlib.pyplot.savefig('Tempo.png',dpi = 180)
        matplotlib.pyplot.show()

    def Spectral_Centroid_Plotter(self):
        matplotlib.pyplot.figure(11)
        matplotlib.pyplot.title("Spectral Centeroid")
        matplotlib.pyplot.plot(self.temp9)
        matplotlib.pyplot.savefig('Spectral Centroid.png',dpi = 180)
        matplotlib.pyplot.show()


#In[6]:
class Divider:

    def length(self):
        self.temp = len(self.Input)
        self.a = self.temp
        Divider.chroma_stft(self)

    def chroma_stft(self):
        self.temp1 = librosa.feature.chroma_stft(self.FloatingInput)
        self.b = Simplify.Stat_Extract(self.temp1)
        Divider.rmse(self)
    
    def rmse(self):
        self.temp2 = librosa.feature.rms(self.FloatingInput)
        self.c = Simplify.Stat_Extract(self.temp2)
        Divider.Spectral_Centroid(self)

    def Spectral_Centroid(self):
        self.temp3 = librosa.feature.spectral_centroid(self.FloatingInput)
        self.d = Simplify.Stat_Extract(self.temp3)
        Divider.Spectral_Bandwidth(self)

    def Spectral_Bandwidth(self):
        self.temp4 = librosa.feature.spectral_bandwidth(self.FloatingInput)
        self.d = Simplify.Stat_Extract(self.temp4)
        Divider.Spectral_Rolloff(self)

    def Spectral_Rolloff(self):
        self.temp5 = librosa.feature.spectral_rolloff(self.FloatingInput)
        self.e = Simplify.Stat_Extract(self.temp5)
        Divider.zcr(self)

    def zcr(self):
        self.temp6 = librosa.feature.zero_crossing_rate(self.FloatingInput)
        self.f = Simplify.Stat_Extract(self.temp6)
        Divider.harmony(self)
    
    def harmony(self):
        self.temp7 = librosa.feature.tempogram(self.FloatingInput)
        self.g = Simplify.Stat_Extract(self.temp7)
        Divider.tempo(self)

    def tempo(self):
        self.temp8 = librosa.beat.tempo(self.FloatingInput)
        self.h = self.temp8
        Divider.mfcc(self)

    def mfcc(self):
        self.temp9 = librosa.feature.mfcc(self.FloatingInput)
        self.i = Simplify.Stat_Extract(self.temp9[0])
        self.j = Simplify.Stat_Extract(self.temp9[1])
        self.k = Simplify.Stat_Extract(self.temp9[2])
        self.l = Simplify.Stat_Extract(self.temp9[3])
        self.m = Simplify.Stat_Extract(self.temp9[4])
        self.n = Simplify.Stat_Extract(self.temp9[5])
        self.o = Simplify.Stat_Extract(self.temp9[6])
        self.p = Simplify.Stat_Extract(self.temp9[7])
        self.q = Simplify.Stat_Extract(self.temp9[8])
        self.r = Simplify.Stat_Extract(self.temp9[9])
        self.s = Simplify.Stat_Extract(self.temp9[10])
        self.t = Simplify.Stat_Extract(self.temp9[11])
        self.u = Simplify.Stat_Extract(self.temp9[12])
        self.v = Simplify.Stat_Extract(self.temp9[13])
        self.w = Simplify.Stat_Extract(self.temp9[14])
        self.x = Simplify.Stat_Extract(self.temp9[15])
        self.y = Simplify.Stat_Extract(self.temp9[16])
        self.z = Simplify.Stat_Extract(self.temp9[17])
        self.aa = Simplify.Stat_Extract(self.temp9[18])
        self.ab = Simplify.Stat_Extract(self.temp9[19])

        self.x_prediction_test = Simplify.array_constructor(self)
        self.x_train_NN = pandas.read_csv("D:\\Personal_Projects\\Python\\Musiphile\\Dataset\\features_30_sec_Edited.csv")
        self.x_train_NN = Simplify.normalize(self.x_train_NN)
        self.y_train_NN = pandas.read_csv("D:\\Personal_Projects\\Python\\Musiphile\\Dataset\\features_30_sec_Output.csv")

        if(sys.argv[2] == '1'):
            Artist.plotter(self)
            Artist.Chroma_STFT_Plotter(self)
            Artist.RMSE_Plotter(self)
            Artist.Spectral_Centroid_Plotter(self)
            Artist.Spectral_Bandwidth_Plotter(self)
            Artist.Spectral_Rolloff_Plotter(self)
            Artist.zcr_Plotter(self)
            Artist.Harmony_Plotter(self)
            Artist.Tempo_Plotter(self)
            Artist.MFCC_Plotter(self)
    
        if(sys.argv[3] == '1'):
            print(self.x_prediction_test)

        if(sys.argv[4] == '1'):
            Neural_Nexus.brain(self)

        Neural_Nexus.SavedMemory(self)


#In[7]:
class Neural_Nexus:
    def brain(self):
        self.Connectors = tensorflow.keras.Sequential()
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 2048, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 1024, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 512, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 256, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 128, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 64, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 32, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 16, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation= 'sigmoid'))
        self.Connectors.add(tensorflow.keras.layers.BatchNormalization())
        self.Connectors.add(tensorflow.keras.layers.Flatten())
        self.Connectors.add(tensorflow.keras.layers.Dense(units = 10, kernel_initializer = 'random_normal', bias_initializer = 'zeros', activation='softmax'))
        self.Connectors.compile(optimizer='adam', loss= 'SparseCategoricalCrossentropy', metrics=['accuracy'])
        self.Connectors.fit(self.x_train_NN, self.y_train_NN, batch_size= 32, epochs= 50, validation_split= 0.1)
        self.Connectors.save("D:\\Personal_Projects\\Python\\Musiphile\\Models\\TrainedModel.h5")
        self.Connectors.save_weights("D:\\Personal_Projects\\Python\\Musiphile\\Models\\TrainedModelWeights")
        self.Classified_Genre = numpy.argmax(self.Connectors.predict(self.x_prediction_test.reshape((1,56))))
        Classifier.Output(self)

    def SavedMemory(self):
        model = tensorflow.keras.models.load_model("D:\\Personal_Projects\\Python\\Musiphile\\Models\\TrainedModel.h5")
        model.load_weights("D:\\Personal_Projects\\Python\\Musiphile\\Models\\TrainedModelWeights")
        self.Classified_Genre = numpy.argmax(model.predict(self.x_prediction_test.reshape((1,56))))
        Classifier.Output(self)



        #Tensor_Machine.Polynomial_kernel(self)

#class Tensor_Machine:
 #   def Polynomial_kernel(self):
  #      self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.x_train_NN, self.y_train_NN, test_size = 0.10)
   #     self.Classifier = sklearn.svm.SVC(kernel = 'rbf')
    #    self.Classifier.fit(numpy.reshape(self.x_train,[900,56]), numpy.reshape(self.y_train,[900,1]))
     #   self.Classified_Genre = self.Classifier.predict(numpy.reshape(self.x_test,[100,56]))
      #  print(self.Classified_Genre)
       # print(sklearn.metrics.confusion_matrix(self.y_test, self.Classified_Genre))
        #print(sklearn.metrics.classification_report(self.y_test, self.Classified_Genre))



#In[8]:
class Classifier:
    def Output(self):
        
        self.Genre_Classification_Labels = { 0 : "Blues", 1 : "Classical", 2 : "Country", 3 : "Disco", 4 : "Hip-Hop", 5 : "Jazz", 6 : "Metal", 7 : "Pop", 8 : "Reggae", 9 : "Rock"}
        print(self.Genre_Classification_Labels[self.Classified_Genre])
        exit()

#In[4]:
class Simplify:
    def Stat_Extract(temp):
        var1 = temp.mean()
        var2 = temp.var()
        return [var1, var2]

    def normalize(x, axis=0):
        return sklearn.preprocessing.normalize(x,axis=axis) 

    def array_constructor(self):
        return numpy.array([self.a, self.b[0], self.b[1],self.c[0], self.c[1], self.d[0], self.d[1], self.e[0], self.e[1], self.f[0], self.f[1], self.g[0], self.g[1], self.h[0], self.i[0], self.i[1], self.j[0], self.j[1], self.k[0], self.k[1], self.l[0], self.l[1], self.m[0], self.m[1], self.n[0], self.n[1], self.o[0], self.o[1], self.p[0], self.p[1], self.q[0], self.q[1], self.r[0], self.r[1], self.s[0], self.s[1], self.t[0], self.t[1],self.s[0], self.s[1], self.u[0], self.u[1], self.v[0], self.v[1], self.w[0], self.w[1], self.x[0], self.x[1], self.y[0], self.y[1], self.z[0], self.z[1], self.aa[0], self.aa[1], self.ab[0], self.ab[1]])

#In[2]:
if __name__ == '__main__':
    os.chdir("D:\\Personal_Projects\\Python\\Musiphile")
    
    if(os.path.isfile(sys.argv[1]) == 1):
        os.remove("D:\\Personal_Projects\\Python\\Musiphile\\Test_Audio.wav")
    else:
        print("Input File doesn't Exist!!!!")
        exit()
    var = Bank()


