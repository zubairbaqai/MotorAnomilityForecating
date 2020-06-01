
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor




class LstmForecase():


    def __init__(self):
        super(LstmForecase, self).__init__()
        self.dataset=[]
        self.OtherData=[]

        self.LongMinuteData=[]
        self.LongMinuteOtherData=[]

        self.LoadDataset()
        self.DataSetLength = len(self.dataset)

        self.window_size=10
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


            #Encoding = to_categorical(YAction,num_classes=3)

            #print(YAction)
            if(YAction!=0):
                randomNumber=random.randint(0,100)
                if(randomNumber>70):
                    self.XInput.append(self.dataset[i - self.window_size:i])
                    self.YInput.append(self.dataset[i][2])

            else:
                randomNumber=random.randint(0,100)
                if(randomNumber>70):
                    #Features=
                    self.XInput.append(self.dataset[i - self.window_size:i])
                    self.YInput.append(self.dataset[i][2])
        self.XInput = np.asarray(self.XInput)
        self.YInput = np.asarray(self.YInput)

        self.FlattenedXInput=np.reshape(self.XInput,
                           (self.XInput.shape[0], self.XInput.shape[1] * self.XInput.shape[2]))








    def ActualOutput(self,CurrentCPValue,NextCPValue):

            if (abs(CurrentCPValue-NextCPValue)<5):# CurrentCPValue+5<NextCPValue

                return 0

            if (CurrentCPValue <= NextCPValue - 5 and CurrentCPValue < NextCPValue):

                return 1

            if (CurrentCPValue >= NextCPValue + 5 and CurrentCPValue > NextCPValue):

                return 2





Forecaster=LstmForecase()



try:
    with open('Regressor.pickle', 'rb') as f:
        est = pickle.load(f)
except:
    print("Training")
    est = HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=100, min_samples_leaf=40).fit( Forecaster.FlattenedXInput,Forecaster.YInput)
    with open('Regressor.pickle', 'wb') as f:
        pickle.dump(est, f)

Correct=0
Wrong=0

def EvaluateInput(index):
    global Correct,Wrong
    if(index<Forecaster.window_size):
        return "Please Choose an index with enough Windows Values"
    CorrectNextCP=Forecaster.dataset[index][2]
    CurrentCP=Forecaster.dataset[index-1][2]
    CorrectAction=Forecaster.ActualOutput(CurrentCP,CorrectNextCP)
    InputToRegressor=Forecaster.dataset[index-10:index]


    FlattenedInput=np.reshape(InputToRegressor,(1, InputToRegressor.shape[0] *  InputToRegressor.shape[1]))
    Result=est.predict(FlattenedInput)

    PredictedAction=Forecaster.ActualOutput(CurrentCP,Result)

    if(PredictedAction==CorrectAction):
        print("Correct")
        Correct+=1
    else:
        print("Wrong")
        Wrong+=1







print(est.score(Forecaster.FlattenedXInput, Forecaster.YInput))

EvaluateInput(10)
print(Correct/(Correct+Wrong))

