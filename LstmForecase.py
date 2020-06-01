from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout,concatenate ,  Flatten
from keras.models import Model
from keras.models import load_model
import pickle
import pandas as pd
from keras.models import Sequential
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing
from keras.optimizers import SGD
import random

class LstmForecase():


    def __init__(self):
        super(LstmForecase, self).__init__()
        self.dataset=[]
        self.OtherData=[]

        self.LongMinuteData=[]
        self.LongMinuteOtherData=[]

        self.LoadDataset()
        self.DataSetLength = len(self.dataset)

        self.window_size=2
        self.Feature_length=len(self.dataset[0])



        self.current_step = 0
        self.episode_reward = 0
        self.episode = 0
        self.ActionPossibility=[0,1,2] ## Hold , UP , DOWN
        self.ActionDistributionCounter = {}
        self.PredictionScore = {}
        self.ActionTotalScore={}
        self.min_max_scaler = preprocessing.MinMaxScaler()



        #self.ActionDistResetter()

        self.WrongCalls=[]

        self.RandomCurrentStep=True
        self.RandomSet=False

        self.previousCorrect=True ## FlagToMake sure, we increment if and only if , the previous results are corrected .
        self.ForceCorrect=True
        self.XInput=[]
        self.YInput=[]
        self.GenerateXandY()




    def LoadDataset(self):

        DataSetPath = "./Dataset/"
        df = pd.read_csv(DataSetPath + "5min-training-data.csv")

        self.dataset = df.drop(['date_and_time', 'upperLnrReg', 'lowerLnrReg','midLnrReg','upperLnrRegStart','lowerLnrRegStart'], axis=1)
        self.dataset=self.dataset.to_numpy()




    def GenerateXandY(self):


        x_scaled = self.min_max_scaler.fit_transform(self.dataset)
        for i in range(self.window_size,len(self.dataset)-1):

            #print(self.XInput)
            NextCPValue = self.dataset[i][2] ##CP
            CurrentCPValue = self.dataset[i-1][2]  ##CP



            YAction=self.ActualOutput(CurrentCPValue,NextCPValue)


            Encoding = to_categorical(YAction,num_classes=3)

            #print(YAction)
            if(YAction!=0):
                self.XInput.append(x_scaled[i - self.window_size:i])
                self.YInput.append(x_scaled[i][2])
            # else:
            #     randomNumber=random.randint(0,100)
            #     if(randomNumber>70):
            #         self.XInput.append(x_scaled[i - self.window_size:i])
            #         self.YInput.append(x_scaled[i][2])








    def ActualOutput(self,CurrentCPValue,NextCPValue):

            if (abs(CurrentCPValue-NextCPValue)<5):# CurrentCPValue+5<NextCPValue

                return 0

            if (CurrentCPValue <= NextCPValue - 5 and CurrentCPValue < NextCPValue):

                return 1

            if (CurrentCPValue >= NextCPValue + 5 and CurrentCPValue > NextCPValue):

                return 2




    def MakeModel(self):
        self.train_x, self.train_y =self.XInput,self.YInput
        self.train_x=np.asarray(self.train_x)
        self.train_y = np.asarray(self.train_y)



        verbose, epochs, batch_size = 1, 500, 128
        n_timesteps, n_features = self.train_x.shape[1], self.train_x.shape[2]
        # define model
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu',return_sequences=True,dropout=0.5, recurrent_dropout=0.5), input_shape=(n_timesteps, n_features)))
        model.add(Bidirectional(LSTM(50, activation='relu',return_sequences=True,dropout=0.5, recurrent_dropout=0.5)))
        model.add(Bidirectional(LSTM(50, activation='relu')))
        model.add((Dense(100, activation='relu')))
        model.add((Dense(1, activation='linear')))
        model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
        # fit network
        model.load_weights('my_model.h5')
        hist=model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        print(hist.history)
        f = open('history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        #
        model.save_weights('my_model.h5')
        return model


Forecaster=LstmForecase()
model=Forecaster.MakeModel()
#model.save_weights('my_model.h5')


def predicting(x_input):





    x_scaled = Forecaster.min_max_scaler.fit_transform(Forecaster.dataset)
    print(Forecaster.min_max_scaler.inverse_transform(x_input))
    x_input = x_input.reshape((1, x_input.shape[0],x_input.shape[1]))
    Result=model.predict(x_input)

    Zeroes=np.zeros(x_input[0][0].shape)
    Zeroes[2]=Result
    Result=Zeroes
    #print(Result)
    Result=Forecaster.min_max_scaler.inverse_transform([Result])
    print(Result[0][2])
    print("-------------")


predicting(Forecaster.XInput[1])
predicting(Forecaster.XInput[2])


