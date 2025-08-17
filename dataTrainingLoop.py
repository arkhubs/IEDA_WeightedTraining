import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers
import copy
import time
from sklearn.linear_model import LogisticRegression
#import scikit-learn
from sklearn.linear_model import LinearRegression
np.set_printoptions(precision=6)
batchSize = 128
class trainModel(object):
    def __init__(self, beta_finish = 0, beta_time = 0):
        self.beta_finish = beta_finish
        self.beta_time = beta_time
    def copy(self):
        return trainModel(self.beta_finish.copy(),self.beta_time.copy())
   
    def train_onestep(self, features, finishes, watch_time, w = []):
        learning_rate = 0.1
        pred = self.compute_finish(features)
        if len(w) == 0:
            gradient = np.dot(features.T, (pred - finishes)) / len(finishes)
        else:
            gradient = np.dot(features.T, (pred - finishes) * w) / len(finishes)

        self.beta_finish -= learning_rate * gradient
        learning_rate = 0.1
        
        pred_time = self.compute_time(features)
        if len(w) == 0:
            gradient = np.dot(features.T, (pred_time - watch_time)) / len(watch_time)
        else:
            gradient = np.dot(features.T, (pred_time - watch_time) * w) / len(watch_time)
        self.beta_time -= learning_rate * gradient


    def train_init(self, features, finishes, watch_time):
        logistic_regression = LogisticRegression(fit_intercept=False)
        
        logistic_regression.fit(features, finishes)
       
        self.beta_finish = logistic_regression.coef_[0]
        linear_regression = LinearRegression(fit_intercept=False)
        linear_regression.fit(features, watch_time)
        self.beta_time = linear_regression.coef_


    
        

    def compute_finish(self, features):
        if len(features.shape) == 3:
            [T,N,d] = features.shape

            if len(self.beta_finish) > d:
                is_short = features[:,:,-1]
                
                is_short = is_short.reshape((T,N,1))
                
                features = np.concatenate([features*is_short, features*(1-is_short)],axis = 2)

        

        finish_rate_logit = features @ self.beta_finish 
        finish_rate = 1/(1+np.exp(-finish_rate_logit))
        return finish_rate
    def compute_time(self, features):
        if len(features.shape) == 3:
            [T,N,d] = features.shape

            if len(self.beta_time) > d:
                is_short = features[:,:,-1]
                is_short = is_short.reshape((T,N,1))
                features = np.concatenate([features*is_short, features*(1-is_short)],axis = 2)
            
        time_mu = features @ self.beta_time
        return time_mu



class recAlgo(object):

    def __init__(self, finish_const, model):
        self.finish_const = finish_const
        self.model = model
    def recommend(self, features):
        finish_const = self.finish_const
        
       
        finish_rate = self.model.compute_finish(features)
        time_mu = self.model.compute_time(features)
        rec_item = np.argmax (finish_const * finish_rate  +  time_mu,axis = 1 )
        
        return rec_item
    




def compute_true_val(features, model):
    [T,N,d] = features.shape

    time_mu = model.compute_time(features)
    watch_time = np.random.exponential(time_mu)
   
    finish_rate = model.compute_finish(features)
    finishes = np.random.binomial(1,finish_rate)
    finishes = finishes.astype(bool) 
    return finishes, watch_time

def recommend_record(data, algo):
    [features,   finishes, watch_time] = data
    [T,N,d] = features.shape
    is_short = features[:,:,-1]
 
    rec_item = algo.recommend(features)
    is_short = is_short[np.arange(T),rec_item]
    
    features = features[np.arange(T),rec_item,:]

    finishes = finishes[np.arange(T),rec_item]
    watch_time = watch_time[np.arange(T),rec_item]
    return rec_item, is_short, features, finishes, watch_time


def create_features(T,N,d,p):
    
    features = np.random.beta(1,1,size=(T,N,d))  
    is_short = np.random.binomial(1,p,size=(T,N,1))
    features = np.concatenate([features, np.ones((T,N,1))],axis = 2)
    features = np.concatenate([features, is_short],axis = 2)  
    is_short = is_short.reshape((T,N))

    return features, is_short
def build_data(features,finishes, watch_time, Treatment):
    data_t = [features[Treatment==1,:,:],finishes[Treatment==1,:], watch_time[Treatment==1,:]]
    data_c = [features[Treatment==0,:,:],finishes[Treatment==0,:], watch_time[Treatment==0,:]]
    return data_t, data_c
class weightModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        
        self.final = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense1( self.dense0(inputs))

        return self.final(x)

class weightDistribution (object):
    def __init__(self, total_features, treatment):
        
        if total_features.shape[1] == 2:
            self.total_short = np.sum(total_features[:,1])
            self.total_short_treatment = np.sum(total_features[:,1] * treatment)
            self.total_long = np.sum(1 - total_features[:,1])
            self.total_long_treatment = np.sum((1-total_features[:,1]) * treatment)
            print(self.total_short,self.total_short_treatment,self.total_long,self.total_long_treatment)

        else:
            
            
            
            self.optimizer = optimizers.experimental.Adam(learning_rate=0.001)
            self.model = weightModel()
            print(total_features.shape)
            self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy',metrics=['accuracy'])
            history = self.model.fit(total_features, treatment, epochs=10, batch_size=batchSize)
            
            #self.model_train_one_step(total_features, treatment)
            print(self.model.summary())
            print('loss', history.history['loss'][-1] / np.log(2))
           
    def predict(self, features):
        if features.shape[1] == 2:
            pred = features[:,1] * self.total_short_treatment/self.total_short + (1-features[:,1]) * self.total_long_treatment/self.total_long
            return pred
        else:
            return self.model(features)[:,0]
    @tf.function
    def model_train_one_step(self, total_features, treatment):
        bce = tf.keras.losses.BinaryCrossentropy()
        
        with tf.GradientTape(persistent=True) as tape:
            loss= bce(treatment,self.model(total_features)[:,0])
        
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss

       
    def weight_one_step(self, total_features, treatment):
        if total_features.shape[1] == 2:
            self.total_short += np.sum(total_features[:,1])
            self.total_short_treatment += np.sum(total_features[:,1] * treatment)
            self.total_long += np.sum(1 - total_features[:,1])
            self.total_long_treatment += np.sum((1-total_features[:,1]) * treatment)
            
        else:
 

            
            return self.model_train_one_step(total_features, treatment)  / np.log(2)
            
        


    
        


def do_experiments(type,rep=-1):
    print(type)
    
    if type == 'global' or type == 'data_sep' or type == 'weighted':
        controlModel = InitModel.copy()
        controlAlgo = recAlgo(finish_const, controlModel)
        treatModel = InitModel.copy()
        TreatmentAlgo = recAlgo(treat_const, treatModel)
    if type == 'naive' or type == 'not_train':
        cur_model = InitModel.copy()
        controlAlgo = recAlgo(finish_const, cur_model)
        TreatmentAlgo = recAlgo(treat_const, cur_model)
   
    

    
    T = batchSize

    outputname = type+'base_T'+str(T)+'length'+str(total_length) + str(finish_const) + '_'+str(treat_const) +'_'+str(N) +'_' + str(d) +'_alpha' + str(alpha)
    outputname = outputname + '_' + str(rep)
    
    file.write(type +'\n')

    print(outputname)
    rec = np.zeros((total_length*T,5))
    recT = np.zeros((total_length*T,3))
    recC = np.zeros((total_length*T,3))
    
    recW = np.zeros((total_length,3))
    total_features=[]
    total_treatment = []

    for t in range(total_length):
        
        cur_features,cur_is_short = create_features(T,N,d,p)
        cur_finishes, cur_watch_time = compute_true_val(cur_features, TrueModel)
        
        
        
        
        if type =='global':
            
            Treatment =  np.random.binomial(1,1,size=(T))
            data, _ = build_data(cur_features,cur_finishes, cur_watch_time, Treatment)
            rec_item_t,is_short_t,features_t,finishes_t,watch_time_t = recommend_record(data, TreatmentAlgo)
            rec_item_c,is_short_c,features_c,finishes_c,watch_time_c = recommend_record(data,  controlAlgo)

            TreatmentAlgo.model.train_onestep(features_t,finishes_t,watch_time_t)
            controlAlgo.model.train_onestep(features_c,finishes_c,watch_time_c)
        else:

            Treatment =  np.random.binomial(1,alpha,size=(T))
            data_t,data_c = build_data(cur_features,cur_finishes, cur_watch_time, Treatment)
            rec_item_t,is_short_t,features_t,finishes_t,watch_time_t = recommend_record(data_t, TreatmentAlgo)
            rec_item_c,is_short_c,features_c,finishes_c,watch_time_c = recommend_record(data_c, controlAlgo)
            feature_merge = np.concatenate([features_t, features_c])
            watch_time_merge = np.concatenate([watch_time_t,watch_time_c])
            finishes_merge = np.concatenate([finishes_t,finishes_c])
            if type == 'naive':
                controlAlgo.model.train_onestep(feature_merge,finishes_merge,watch_time_merge)
            elif type == 'data_sep':
                TreatmentAlgo.model.train_onestep(features_t,finishes_t,watch_time_t)
                controlAlgo.model.train_onestep(features_c,finishes_c,watch_time_c)
            elif type == 'weighted':
                
                init_num = 200
                if t < init_num:
                    if t == 0:
                        total_features = np.concatenate([features_t, features_c])

                        total_treatment =np.concatenate([np.ones(features_t.shape[0]), np.zeros(features_c.shape[0])])
                    else:
                        total_features = np.append(total_features, features_t, axis = 0)
                        total_features = np.append(total_features, features_c, axis = 0)
                        
                        total_treatment = np.append(total_treatment, np.ones(features_t.shape[0]), axis = 0)
                        total_treatment = np.append(total_treatment, np.zeros(features_c.shape[0]), axis = 0)
                    
                    TreatmentAlgo.model.train_onestep(features_t,finishes_t,watch_time_t)
                    controlAlgo.model.train_onestep(features_c,finishes_c,watch_time_c)


                  
                else:
                    
                    if t == init_num:
                        cur_start = t
                        W = weightDistribution(total_features, total_treatment)
                        loss = 0
                    else: 
                        if t % init_num == 0:   cur_start = t
                        loss = W.weight_one_step(selected_features, cur_treatment)

                    

                    weights_t = W.predict(features_t)
                    weights_c = W.predict(features_c)
                    weights = W.predict(feature_merge)
                    
                    selected_features = np.concatenate([features_t, features_c])
                    
                    
                    cur_treatment =np.concatenate([np.ones(features_t.shape[0]), np.zeros(features_c.shape[0])])

                    logloss =  -np.mean(np.log2(weights) * cur_treatment  + (1-cur_treatment) * np.log2(1-weights))
                    recW[t,:] = [logloss, np.sum(weights_t)/len(weights_t), np.sum(weights)]
                    if t %1000 == 1:
                        print(t, 'weight',np.mean(recW[init_num:(t+1),], axis = 0), 'loss',loss )
                        print([logloss, np.sum(weights_t)/len(weights_t), np.sum(weights)])
                        print(weights)

                    
                    TreatmentAlgo.model.train_onestep(feature_merge,finishes_merge,watch_time_merge,  weights/alpha )
                    controlAlgo.model.train_onestep(feature_merge,finishes_merge,watch_time_merge, (1 - weights)/(1-alpha))
                   
        
        if type == 'global':
            cur_data_t = np.array([is_short_t,watch_time_t, finishes_t]).T 
            cur_data_c = np.array([is_short_c,watch_time_c, finishes_c]).T
            recT[t*T:(t+1)*T] = cur_data_t
            recC[t*T:(t+1)*T] = cur_data_c


        else:
            if type == 'weighted' and t >= init_num:
                cur_data_t = np.concatenate([np.array([is_short_t,watch_time_t, finishes_t, weights_t]).T, np.ones((len(is_short_t),1))],axis=1)
                cur_data_c = np.concatenate([np.array([is_short_c,watch_time_c, finishes_c, weights_c]).T, np.zeros((len(is_short_c),1))],axis=1)
            else:
                cur_data_t = np.concatenate([np.array([is_short_t,watch_time_t, finishes_t]).T, np.zeros((len(is_short_t),1)),np.ones((len(is_short_t),1))],axis=1)
                cur_data_c = np.concatenate([np.array([is_short_c,watch_time_c, finishes_c]).T,np.zeros((len(is_short_c),1)), np.zeros((len(is_short_c),1))],axis=1)
        
            rec[t*T:(t+1)*T] = np.concatenate([cur_data_t,cur_data_c],axis=0) 
        
        if t % 1000 == 0 and t >= 1000:
            print(t)

            if type != 'global':
                
                 
                rectemp =rec[(t-1000)*T:(t+1)*T,:] 
                recC = rectemp[rectemp[:,4] == 0,:]
                recT = rectemp[rectemp[:,4] == 1,:]
            print('treatment',np.mean(recT, axis = 0) )
            print('control',np.mean(recC, axis = 0) )
           
        

    if type != 'global':
        recT = rec[rec[:,4] == 1,:]
        recC = rec[rec[:,4] == 0,:]
    
    file.write('treatment' + f"{np.mean(recT, axis = 0)}\n")
    file.write('control  ' + f"{np.mean(recC, axis = 0)}\n")
    file.write('diff     ' + f"{np.mean(recC, axis = 0)-np.mean(recT, axis = 0)}\n")
    file.write('treatment_short' + f"{np.mean(recT[recT[:,0] == 1,:], axis = 0)}\n")
    file.write('control_short  ' + f"{np.mean(recC[recC[:,0] == 1,:], axis = 0)}\n")
    file.write('treatment_long' + f"{np.mean(recT[recT[:,0] == 0,:], axis = 0)}\n")
    file.write('control_long  ' + f"{np.mean(recC[recC[:,0] == 0,:], axis = 0)}\n")

    file.flush()

    print('treatment',np.mean(recT, axis = 0) )

    print('control',np.mean(recC, axis = 0) )
    print('diff',np.mean(recC, axis = 0) - np.mean(recT, axis = 0)  )

    
    
    print(np.std(recT, axis = 0) , len(recT) )
    print(np.std(recC, axis = 0)  , len(recC) )
    print('control finish' , controlAlgo.model.beta_finish)
    print('treat finish' , TreatmentAlgo.model.beta_finish)
    print('control time', controlAlgo.model.beta_time)
    print('treat time' , TreatmentAlgo.model.beta_time)
    np.save('/XXX/exp3/recT' + outputname, recT)
    np.save('/XXX/exp3/recC'+ outputname, recC)
    if type == 'weighted':
        print('weight',np.mean(recW[init_num:,], axis = 0) )
        np.save('/XXX/exp3/recW'+ outputname, recW)
    if type == 'global':
        return controlModel, recT, recC
    else: return recT, recC

        
    

if __name__ == '__main__':
    
    T = 1000000
    total_length = 10000
    N = 100
    d = 10
    p = 1/2
    finish_const = 10
    treat_const = 9
    print(finish_const,treat_const)
    features,is_short = create_features(T,N,d,p)
    alpha = 0.5
    filename = 'full' + str(batchSize) + 'base_length'+str(total_length) + str(finish_const) + '_'+str(treat_const) +'_'+str(N)   +'_'+str(d)    +'_alpha'+str(alpha)
    file = open(filename +'.txt', 'w') 


    base_finish = np.arange(0,1,0.1)
    beta_finish_short = 0.9 *  base_finish
    beta_finish_long  = 0.6 *  base_finish
    beta_finish_short = np.append(beta_finish_short,-d/4)
    beta_finish_long = np.append(beta_finish_long,-d/4)
    beta_finish_short = np.append(beta_finish_short,0)
    beta_finish_long = np.append(beta_finish_long,0)
    
    beta_finish = np.concatenate([beta_finish_short, beta_finish_long])

    
    
   
    base_time = np.arange(1,0,-0.1)
    beta_time_short = 1.0 *  base_time
    beta_time_long =  1.5 *  base_time
    beta_time_short = np.append(beta_time_short,0)          
    
    beta_time_long = np.append(beta_time_long,0)
    beta_time_short = np.append(beta_time_short,0)
    
    beta_time_long = np.append(beta_time_long,0)
    beta_time = np.concatenate([beta_time_short, beta_time_long])
    TrueModel = trainModel(beta_finish,beta_time)
    time_mu = TrueModel.compute_time(features)
    watch_time = np.random.exponential(time_mu)
   
    finish_rate = TrueModel.compute_finish(features)
    finishes = np.random.binomial(1,finish_rate)
    print("short video mean %.2f, and std finish rate %.2f" %(np.mean(finish_rate[is_short==1]),np.std(finish_rate[is_short==1])))
    
    print("long video mean %.2f and std finish rate %.2f"%(np.mean(finish_rate[is_short==0]),np.std(finish_rate[is_short==0])))
   
    
    print("short video mean time lambda %.2f" % (np.sum(time_mu * is_short)/ np.sum(is_short)))
    print("long video mean time lambda %.2f" % (np.sum(time_mu * (1-is_short))/ np.sum(1-is_short)))
    BestAlgo = recAlgo(finish_const, TrueModel)

    finishes, watch_time = compute_true_val(features, TrueModel)

    
    rec_item,is_short,selected_features, selected_finishes, selected_watch_time = recommend_record([features, finishes, watch_time], BestAlgo)
    print("control best",np.mean(is_short), np.mean(selected_finishes),np.mean(selected_watch_time))
    treatBestAlgo = recAlgo(treat_const, TrueModel)
    rec_item,is_short,selected_features, selected_finishes, selected_watch_time = recommend_record([features, finishes, watch_time], treatBestAlgo)
    print("treat best",np.mean(is_short), np.mean(selected_finishes),np.mean(selected_watch_time))

    InitModel = trainModel()
    InitModel.train_init(selected_features, selected_finishes, selected_watch_time)
    print('init finish',InitModel.beta_finish)
    print('init time',InitModel.beta_time)


    
    print('init finish',InitModel.beta_finish)
    print('init time',InitModel.beta_time)
    trainAlgoInit = recAlgo(finish_const, InitModel)
    rec_item,is_short,selected_features, selected_finishes, selected_watch_time = recommend_record([features, finishes, watch_time], trainAlgoInit)
    print('init control',np.mean(is_short), np.mean(selected_finishes),np.mean(selected_watch_time))
    treatAlgoInit = recAlgo(treat_const, InitModel)
    rec_item,is_short,selected_features, selected_finishes, selected_watch_time = recommend_record([features, finishes, watch_time], treatAlgoInit)
    print('init treat',np.mean(is_short), np.mean(selected_finishes),np.mean(selected_watch_time))
    aveT = np.zeros(3)
    aveC = np.zeros(3)
    reptimes = 100
    
    
    for rep in np.arange(reptimes):
        print(rep)
        cur = time.time()
        _,recT, recC= do_experiments('global',rep)
        aveT = aveT + np.mean(recT, axis = 0)
        aveC = aveC + np.mean(recC, axis = 0)

        print('time',time.time()-cur)
    aveT = aveT / reptimes
    aveC = aveC / reptimes
    file.write('treatment_ave ' + f"{aveT}\n")
    file.write('control_ave ' + f"{aveC}\n")
    file.write('diff_ave     ' + f"{aveT-aveC}\n")
    file.flush()
    
    aveT = np.zeros(5)
    aveC = np.zeros(5)
    for rep in np.arange(reptimes):
        print(rep)
        cur = time.time()
        recT, recC = do_experiments('weighted',rep)
        aveT = aveT + np.mean(recT, axis = 0)
        aveC = aveC + np.mean(recC, axis = 0)
        print('time',time.time()-cur)
    aveT = aveT / reptimes
    aveC = aveC / reptimes
    file.write('treatment_ave ' + f"{aveT}\n")
    file.write('control_ave ' + f"{aveC}\n")
    file.write('diff_ave     ' + f"{aveT-aveC}\n")
    file.flush()
    
 
    aveT = np.zeros(5)
    aveC = np.zeros(5)
    for rep in np.arange(reptimes):
        print(rep)
        cur = time.time()
        recT, recC = do_experiments('naive',rep)
        aveT = aveT + np.mean(recT, axis = 0)
        aveC = aveC + np.mean(recC, axis = 0)
        print('time',time.time()-cur)
    aveT = aveT / reptimes
    aveC = aveC / reptimes
    file.write('treatment_ave ' + f"{aveT}\n")
    file.write('control_ave ' + f"{aveC}\n")
    file.write('diff_ave     ' + f"{aveT-aveC}\n")
    file.flush()


    cur = time.time()
    aveT = np.zeros(5)
    aveC = np.zeros(5)
    for rep in np.arange(reptimes):
        print(rep)
        cur = time.time()
        recT, recC = do_experiments('data_sep',rep)
        aveT = aveT + np.mean(recT, axis = 0)
        aveC = aveC + np.mean(recC, axis = 0)
        print('time',time.time()-cur)
    aveT = aveT / reptimes
    aveC = aveC / reptimes
    file.write('treatment_ave ' + f"{aveT}\n")
    file.write('control_ave ' + f"{aveC}\n")
    file.write('diff_ave     ' + f"{aveT-aveC}\n")
    file.flush()

    cur = time.time()
    aveT = np.zeros(5)
    aveC = np.zeros(5)
    for rep in np.arange(reptimes):
        print(rep)
        cur = time.time()
        recT, recC = do_experiments('not_train',rep)
        aveT = aveT + np.mean(recT, axis = 0)
        aveC = aveC + np.mean(recC, axis = 0)
        print('time',time.time()-cur)
    aveT = aveT / reptimes
    aveC = aveC / reptimes
    file.write('treatment_ave ' + f"{aveT}\n")
    file.write('control_ave ' + f"{aveC}\n")
    file.write('diff_ave     ' + f"{aveT-aveC}\n")
    file.flush()



    print('init finish',InitModel.beta_finish)
    print('init time',InitModel.beta_time)
    

    
  
    
    







