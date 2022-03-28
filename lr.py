import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import statistics as st

from sklearn.linear_model import LinearRegression

def visualize_3d(df, lin_reg_weights=[1,1,1], feat1=0, feat2=1, labels=2,
                 xlim=(-1.7, 1.71), ylim=(-1.32, 3.7), zlim=(.8, 1.31),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
    """ 
    3D surface plot. 
    Main args:
      - df: dataframe with feat1, feat2, and labels
      - feat1: int/string column name of first feature
      - feat2: int/string column name of second feature
      - labels: int/string column name of labels
      - lin_reg_weights: [b_0, b_1 , b_2] list of float weights in order
    Optional args:
      - x,y,zlim: axes boundaries. Default to -1 to 1 normalized feature values.
      - alpha: step size of this model, for title only
      - x,y,z labels: for display only
      - title: title of plot
    """
    

    # Setup 3D figure
    ax = plt.figure().gca(projection='3d')
   # plt.hold(True)

    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array( [lin_reg_weights[0] +
                       lin_reg_weights[1]*f1 +
                       lin_reg_weights[2]*f2  # height
                       for f1, f2 in zip(axes1, axes2)] )
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()



#prepare data
if __name__ == '__main__':
    out_filename = 'results2.csv'    
    outfile = open(out_filename, "w")
    
    df = pd.read_csv("data2.csv",names = ["Age","Weight","Height"])
    num = df[df.columns[0]].count()
    intercept = []
    for i in range(num):
        intercept.append(1)
    df.insert(0,column = 'Intercept', value = intercept)   
    arr = df.to_numpy()


# Means and Standard Devitation: 
    m_Age = st.mean(df["Age"])
    m_Weight = st.mean(df["Weight"])
    st_Age = st.stdev(df["Age"])
    st_Weight = st.stdev(df["Weight"])

#Normalize Values: 
    Intercept_list = arr[:,0]
    Age_list = arr[:,1]
    Weight_list = arr[:,2]
    Age_list[:] = [((x-m_Age)/st_Age) for x in Age_list]
    Weight_list[:] = [((x-m_Weight)/st_Weight) for x in Weight_list]

    df["Age"] = Age_list
    df["Weight"] = Weight_list
    arr2 = df.to_numpy()
#print(arr2)

#Running Gradient Descent
#initialize values
    alpha_List = [.001,.005,.01,.05,.1,.5,1,5,10]
    b0 = 0
    b1 = 0
    b2 = 0
#create column vectors
    x0 = arr2[:,[0]]
    x1 = arr2[:,[1]]
    x2 = arr2[:,[2]]
    y = arr2[:,[3]]
    s = (arr2.shape)
#Run algorithm for all alphas
#print(arr2)
    for alp in alpha_List:
        for i in range(100):
            b0 = b0-alp*(1/s[0])*(np.matmul(x0.transpose(),(b0*x0+b1*x1+b2*x2-y)))
            b1 = b1-alp*(1/s[0])*(np.matmul(x1.transpose(),(b0*x0+b1*x1+b2*x2-y)))
            b2 = b2-alp*(1/s[0])*(np.matmul(x2.transpose(),(b0*x0+b1*x1+b2*x2-y)))
        b0 = round(b0[0,0],6)
        b1 = round(b1[0,0],6)
        b2 = round(b2[0,0],6)
        outfile.write(str(alp) + '\t' + str(100) + '\t' + str(b0)+ '\t' + str(b1)+ '\t' + str(b2))
        outfile.write('\n')
      
        
    b0 = 0
    b1 = 0
    b2 = 0
    alp= .25
    for i in range(200):
        b0 = b0-alp*(1/s[0])*(np.matmul(x0.transpose(),(b0*x0+b1*x1+b2*x2-y)))
        b1 = b1-alp*(1/s[0])*(np.matmul(x1.transpose(),(b0*x0+b1*x1+b2*x2-y)))
        b2 = b2-alp*(1/s[0])*(np.matmul(x2.transpose(),(b0*x0+b1*x1+b2*x2-y)))
    b0 = round(b0[0,0],6)
    b1 = round(b1[0,0],6)
    b2 = round(b2[0,0],6)
    outfile.write(str(alp) + '\t' + str(200) + '\t' + str(b0)+ '\t' + str(b1)+ '\t' + str(b2))
    outfile.write('\n')
    
        #X = np.column_stack((x1,x2))
        #print(X)
        #reg = LinearRegression().fit(X, y) 
        #print(reg.coef_)
        #print("ALPHA:",alp," BETA0:", round(b0[0,0],5)," BETA1:",round(b1[0,0],5)," BETA2:",round(b2[0,0],5)) 
       #visualize_3d(df, lin_reg_weights=[b0[0],b1[0],b2[0]], feat1='Age', feat2='Weight', labels='Height', alpha=alp, xlabel="Age", ylabel="Weight", zlabel="Height")        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
