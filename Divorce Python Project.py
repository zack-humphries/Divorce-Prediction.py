#!/usr/bin/env python
# coding: utf-8

# # Zachary Humphries
# ### GT-4801B
# ### Spring 2020
# 
# # Divorce Dataset
# 
# 
# With this dataset, I wanted to isolate which questions strongly correlate to whether the couple is married or not. I included a heatmap, showing all of the different questions that correlate to one another. I also included a function, which would isolate the strong correlations for the "Class" category. Finally I created a loop which visually shows how the different types couples answer to the questions. I plan on making subplots for the loop in the future to speed up the loading time.

# # Attribute Information
# 
# 1. If one of us apologizes when our discussion deteriorates, the discussion ends.
# 2. I know we can ignore our differences, even if things get hard sometimes.
# 3. When we need it, we can take our discussions with my spouse from the beginning and correct it.
# 4. When I discuss with my spouse, to contact him will eventually work.
# 5. The time I spent with my wife is special for us.
# 6. We don't have time at home as partners.
# 7. We are like two strangers who share the same environment at home rather than family.
# 8. I enjoy our holidays with my wife.
# 9. I enjoy traveling with my wife.
# 10. Most of our goals are common to my spouse.
# 11. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
# 12. My spouse and I have similar values in terms of personal freedom.
# 13. My spouse and I have similar sense of entertainment.
# 14. Most of our goals for people (children, friends, etc.) are the same.
# 15. Our dreams with my spouse are similar and harmonious.
# 16. We're compatible with my spouse about what love should be.
# 17. We share the same views about being happy in our life with my spouse
# 18. My spouse and I have similar ideas about how marriage should be
# 19. My spouse and I have similar ideas about how roles should be in marriage
# 20. My spouse and I have similar values in trust.
# 21. I know exactly what my wife likes.
# 22. I know how my spouse wants to be taken care of when she/he sick.
# 23. I know my spouse's favorite food.
# 24. I can tell you what kind of stress my spouse is facing in her/his life.
# 25. I have knowledge of my spouse's inner world.
# 26. I know my spouse's basic anxieties.
# 27. I know what my spouse's current sources of stress are.
# 28. I know my spouse's hopes and wishes.
# 29. I know my spouse very well.
# 30. I know my spouse's friends and their social relationships.
# 31. I feel aggressive when I argue with my spouse.
# 32. When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .
# 33. I can use negative statements about my spouse's personality during our discussions.
# 34. I can use offensive expressions during our discussions.
# 35. I can insult my spouse during our discussions.
# 36. I can be humiliating when we discussions.
# 37. My discussion with my spouse is not calm.
# 38. I hate my spouse's way of open a subject.
# 39. Our discussions often occur suddenly.
# 40. We srart a discussion before I know what's going on.
# 41. When I talk to my spouse about something, my calm suddenly breaks.
# 42. When I argue with my spouse, ı only go out and I don't say a word.
# 43. I mostly stay silent to calm the environment a little bit.
# 44. Sometimes I think it's good for me to leave home for a while.
# 45. I'd rather stay silent than discuss with my spouse.
# 46. Even if I'm right in the discussion, I stay silent to hurt my spouse.
# 47. When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
# 48. I feel right in our discussions.
# 49. I have nothing to do with what I've been accused of.
# 50. I'm not actually the one who's guilty about what I'm accused of.
# 51. I'm not the one who's wrong about problems at home.
# 52. I wouldn't hesitate to tell my spouse about her/his inadequacy.
# 53. When I discuss, I remind my spouse of her/his inadequacy.
# 54. I'm not afraid to tell my spouse about her/his incompetence.

# In[2]:


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns


# # Importing and Formatting Data

# In[5]:


file = "divorce.xlsx"
df = pd.read_excel(file)
df.head()


# In[6]:


def adjust(data_input):
    return (data_input - 2)

for i in df:
    df[i] = adjust(df[i])
    
df["Class"] = df["Class"] + 2

marriage = df["Class"]
questions = df.drop([5])

df


# In[7]:


df.describe()


# # Correlation Heat Map

# In[8]:


import numpy as np

figcorr = plt.figure(figsize=(20,15))
ax = figcorr.add_subplot(111)
df_corr = df.corr()
cax = ax.matshow(df_corr, vmin=0, vmax=1, cmap='RdBu')
figcorr.colorbar(cax)
ticks = np.arange(0,54,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.head(), rotation=45, ha="left")
ax.set_yticklabels(df.head())

#sns.heatmap(df.corr(), annot=True, cmap="magma")
#plt.show()


df_corr_class = df_corr["Class"]
rowname = df_corr["Class"].index
rowindex = 0
print("These are the questions more correlated (Greater than 0.9 or Less than -0.9) to whether the couple is married or not:")
for i in df_corr_class:
    if (i > 0.9 or i < -0.9) and i != 1.0:
        print(str(rowname[rowindex]) + ": " + str(round(i, 4)))
    rowindex+=1


# # Subplot of How Married and Divorced Couples Respond to Each Question

# In[9]:


df_married = df[:-86]
df_divorced = df.drop(df.index[0:85])


# In[10]:


import plotly.figure_factory as ff


# In[11]:


#making a list of all of the plots for later use
plot_list = []

for i in df:
    if i != "Class":
        hist_data = [df_married[i], df_divorced[i]]
        group_label = ['Divorced', 'Married']
        fig = ff.create_distplot(hist_data, group_label, show_hist=False, show_rug=False)
        fig.update_layout(title_text=str(i))
        fig.update_xaxes(
            ticktext=["Strongly Disagree", "Disagree", "Indifferent", "Agree", "Strongly Agree"],
            tickvals=[-2, -1, 0, 1, 2],
            tickangle=45)
        plot_list.append(fig)


# In[13]:


#Creates subplot
fig = make_subplots(
    rows=9, cols=6,
    specs=[
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"},{"type": "xy"}],],
    subplot_titles=('Atr1', 'Atr2', 'Atr3', 'Atr4', 'Atr5', 'Atr6', 
                    'Atr7', 'Atr8', 'Atr9', 'Atr10', 'Atr11', 'Atr12', 
                    'Atr13', 'Atr14', 'Atr15', 'Atr16', 'Atr17', 'Atr18', 
                    'Atr19', 'Atr20', 'Atr21', 'Atr22', 'Atr23', 'Atr24', 
                    'Atr25', 'Atr26', 'Atr27', 'Atr28', 'Atr29', 'Atr30', 
                    'Atr31', 'Atr32', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 
                    'Atr37', 'Atr38', 'Atr39', 'Atr40', 'Atr41', 'Atr42', 
                    'Atr43', 'Atr44', 'Atr45', 'Atr46', 'Atr47', 'Atr48', 
                    'Atr49', 'Atr50', 'Atr51', 'Atr52', 'Atr53', 'Atr54')
)


#Places 2 sets of data into rows and columns
figcount = 0
rows1 = range(1,10)
cols1 = range(1,7)
for r in rows1:
    for c in cols1:
        fig.add_trace(plot_list[figcount].data[0], row = r, col = c)
        fig.add_trace(plot_list[figcount].data[1], row = r, col = c)
        figcount+=1
    
fig.update_layout(title_text='Answer Responses Split by Relationship Status: Divorced (Orange) vs Married (Blue)')
fig.update_layout(height=1000, width = 1200, showlegend=False)
fig.show()


# # Simple Linear Regression Analysis
# I will be using the highest correlative variable to marriage status, Atr40.
# 

# In[14]:


DV = "Class"
X = df.drop(DV, axis = 1)
y = df[DV]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[15]:


from sklearn.linear_model import LinearRegression
modelsimp = LinearRegression()

modelsimp.fit(X_train[['Atr40']], y_train)

intercept = modelsimp.intercept_
coefficient = modelsimp.coef_

print('Likeliness to Marry = {0:0.2f} + ({1:0.2f} x Atr40)'.format(intercept, coefficient[0]))


# In[16]:


from sklearn import metrics
import numpy as np
predictions = modelsimp.predict(X_test[['Atr40']])

metrics.explained_variance_score(y_test, predictions).round(3)

print('R-Squared = {0}'.format(metrics.explained_variance_score(y_test, predictions).round(3)))


# # Multivariable Linear Regression

# In[17]:


DV = "Class"
X = df.drop(DV, axis = 1)
y = df[DV]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[18]:


modellin = LinearRegression()

modellin.fit(X_train, y_train)

intercept = modellin.intercept_
coefficient = modellin.coef_


# In[19]:


import statsmodels.api as sm
from scipy import stats


# In[20]:


# Code help from JARH from https://stackoverflow.com/a/42677750
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[21]:


predicted_class_reg = modellin.predict(X_test)
predicted_class_reg = np.round(predicted_class_reg, 0)
for i in predicted_class_reg:
    i = int(i)


# In[22]:


from sklearn.metrics import confusion_matrix

cmlin = pd.DataFrame(confusion_matrix(y_test, predicted_class_reg))

cmlin['Total'] = np.sum(cmlin, axis=1)

cmlin = cmlin.append(np.sum(cmlin, axis=0), ignore_index=True)

cmlin.columns = ['Predicted No', 'Predicted Yes', 'Total']

cmlin = cmlin.set_index([['Actual No', 'Actual Yes', 'Total']])

print(cmlin)


# In[23]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_class_reg))

# https://en.wikipedia.org/wiki/Confusion_matrix


# # Logistic Regression

# In[24]:


DV = "Class"
X = df.drop(DV, axis = 1)
y = df[DV]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[25]:


from sklearn.linear_model import LogisticRegression
modellog = LogisticRegression()

modellog.fit(X_train, y_train)

intercept = modellog.intercept_
coefficients = modellog.coef_


# In[26]:


coef_list = list(coefficients[0,:])
coef_df = pd.DataFrame({'Feature': list(X_train.columns),
    'Coefficient': coef_list})

print(coef_df)


# In[27]:


predicted_prob = modellog.predict_proba(X_test)[:,1]
predicted_class = modellog.predict(X_test)


# In[28]:


from sklearn.metrics import confusion_matrix

cmlog = pd.DataFrame(confusion_matrix(y_test, predicted_class))

cmlog['Total'] = np.sum(cmlog, axis=1)

cmlog = cmlog.append(np.sum(cmlog, axis=0), ignore_index=True)

cmlog.columns = ['Predicted No', 'Predicted Yes', 'Total']

cmlog = cmlog.set_index([['Actual No', 'Actual Yes', 'Total']])

print(cmlog)


# In[29]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_class))

# https://en.wikipedia.org/wiki/Confusion_matrix


# # Decision Tree

# In[30]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[31]:


tree.plot_tree(clf.fit(X, y)) 


# # Random Forest

# Help from Will Koehrsen at https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[33]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(X_train, y_train);


# In[35]:


# Get numerical feature importances
importances = list(rf.feature_importances_)


# In[36]:


# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.figure(figsize=(12, 6))
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, list(X_train.columns), rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[37]:


# Use the forest's predict method on the test data and condenses to dummy 1 or 0
predictionrf = rf.predict(X_test)

predictionrf = np.round(predictionrf, 0)
for i in predictionrf:
    i = int(i)
    
# Calculate the absolute errors
errors = abs(predictionrf - y_test)


# In[38]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * sum(errors)/len(errors)

# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[40]:


from sklearn.metrics import confusion_matrix

cmrf = pd.DataFrame(confusion_matrix(y_test, predictionrf))

cmrf['Total'] = np.sum(cmrf, axis=1)

cmrf = cmrf.append(np.sum(cmrf, axis=0), ignore_index=True)

cmrf.columns = ['Predicted No', 'Predicted Yes', 'Total']

cmrf = cmrf.set_index([['Actual No', 'Actual Yes', 'Total']])

print(cmrf)


# In[41]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predictionrf))

# https://en.wikipedia.org/wiki/Confusion_matrix


# # Artificial Neural Network

# Help from Adian Wilson at https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24

# In[42]:


import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training

# input data
inputs = df.drop(DV, axis = 1)

# output data
outputs = df[DV]


# In[74]:


# create NeuralNetwork class
class NeuralNetwork:
    
    
    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.linspace(0.5, 0.5, 54)
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 5,000 iterations
    def train(self, epochs=5000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            if epoch == epoc
            
            hs-1:
                self.display_weights(self.weights)
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)
            

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction
    
    def display_weights(self, weights):
        # list of x locations for plotting
        x_values = list(range(len(weights)))
        # Make a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(x_values, weights, orientation = 'vertical')
        # Tick labels for x axis
        plt.xticks(x_values, list(X_train.columns), rotation='vertical')
        # Axis labels and title
        plt.ylabel('Weight'); plt.xlabel('Variable'); plt.title('Variable Weight');


# In[75]:


# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network and prints chart with each variables' weight
NN.train()


# In[45]:


# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()


# In[47]:


print("The Accuracy of this ANN is: " + str(100*(1 - (round(NN.error_history[4999], 4)))) + "%")


# # PCA - Principle Component Analysis

# In[48]:


from sklearn.decomposition import PCA

modelpca = PCA()

modelpca.fit(X)

explained_var_ratio = modelpca.explained_variance_ratio_

print(explained_var_ratio)


# In[49]:


# Lets make this a little easier to read by using np
cum_sum_explained_var = np.cumsum(modelpca.explained_variance_ratio_)

print(cum_sum_explained_var)


# In[50]:


# Determine the number of Principle components using a threshold
threshold = .9
for i in range(len(cum_sum_explained_var)):
    if cum_sum_explained_var[i] >= threshold:
        best_n_components = i+1
        break
    else:
        pass
    
print('The best n_components is {}'.format(best_n_components))


# In[79]:


plt.matshow(modelpca.components_[0:9],cmap='viridis')
plt.colorbar()
plt.xticks(range(len(X_train.columns)),X_train.columns,rotation=65,ha='left')
plt.tight_layout()
plt.show()# 

