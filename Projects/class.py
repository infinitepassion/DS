from matplotlib import pyplot as plt
# class Circle(object):

    # def __init__(self,radius=3,color='blue'):

        # self.radius=radius
        # self.color=color

    # def add_radius(self,r):

        # self.radius=self.radius+r
        # return(self.radius)
    # def drawCircle(self):

        # plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        # plt.axis('scaled')
        # plt.show()

# RedCircle=Circle(10,'red')
# dir(RedCircle)
# RedCircle.radius=1
# RedCircle.drawCircle()
# print('Radius of object:',RedCircle.radius)
# RedCircle.add_radius(2)
# print('Radius of object of after applying the method add_radius(2):',RedCircle.radius)
# RedCircle.add_radius(5)
# print('Radius of object of after applying the method add_radius(5):',RedCircle.radius)

# BlueCircle=Circle(radius=100)
# BlueCircle.drawCircle()

# class Rectangle(object):

    # def __init__(self,width=2,height =3,color='r'):
        # self.height=height
        # self.width=width
        # self.color=color

    # def drawRectangle(self):
        # import matplotlib.pyplot as plt
        # plt.gca().add_patch(plt.Rectangle((0, 0),self.width, self.height ,fc=self.color))
        # plt.axis('scaled')
        # plt.show()

# SkinnyBlueRectangle= Rectangle(2,10,'blue')
# SkinnyBlueRectangle.drawRectangle()

# FatYellowRectangle = Rectangle(20,5,'yellow')
# FatYellowRectangle.drawRectangle()

# *****************************************
import pandas as pd
# # csv_path='https://ibm.box.com/shared/static/keo2qz0bvh4iu6gf5qjq4vdrkt67bvvb.csv'
# df = pd.read_csv('top_selling_albums.csv')
# # 
# print(df.head())

# df = pd.read_excel('top_selling_albums.xlsx')
# # print (df)
# df.head()
# x=df[['Length']]
# # print(x)
# y=df[['Artist','Length','Genre']]
# print(y)
# print (type(y))


# *****************************************

import numpy as np

# a=[[1,2,3,4,5],[1,2,3,4,5]]
# b=np.array(a)
# print (type(a),type(b))
# print(b.max(),b.min())
# c=2*b
# print(type(c),c*b,np.pi)
# x=np.array([0,np.pi/2,np.pi])
# y=np.sin(x)
# print(x,y)
# k=np.linspace(1,10,3)
# print(k)
# k=np.linspace(1,10,num=5)
# print(k)

# x=np.linspace(0,2*np.pi,100)
# y=np.sin(x)
# # plt.plot(x,y)
# # plt.show()

# A=np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
# print(A.ndim)
# print(A.shape)
# print (A.size)


# import pandas library
import pandas as pd
# read the online file by the URL provides above, and assign it to variable "df"
path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(path,header=None)


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers
df['price'] = df['price'].replace(['?'], 0)
# print(df.head(15))
# print(df.columns)
# print(df.dtypes)
df['price']=df['price'].astype(int)
# print(df.describe())
# print(df.describe(include="all"))
# print('hello')
import seaborn as sns
# df.corr()

# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(0,)
# plt.show()
# print(df.info())
# x=df['engine-size']
# y=df['price']
# plt.scatter(x,y)
# plt.title("my Graph")
# plt.xlabel("engine-size")
# plt.ylabel('price')
# plt.show()


# sns.boxplot(x="drive-wheels", y="price", data=df)
# plt.show()


# grp by panda

df['drive-wheels'].unique()
df_group_one=df[['drive-wheels','body-style','price']]
df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
print(df_group_one)
# df_gptest=df[['drive-wheels','body-style','price']]
# grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
# print(grouped_test1)

# grouped_pivot=grouped_test1.pivot(index='body-style',columns='drive-wheels')
# print(grouped_pivot)

# grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0 eg NaN
# print(grouped_pivot)

# heatmap
#use the grouped results
# plt.pcolor(grouped_pivot, cmap='RdBu')
# fig, ax=plt.subplots()
# im=ax.pcolor(grouped_pivot, cmap='RdBu')

# #label names
# row_labels=grouped_pivot.columns.levels[1]
# col_labels=grouped_pivot.index
# #move ticks and labels to the center
# ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
# ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)
# #insert labels
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(col_labels, minor=False)
# #rotate label if too long
# plt.xticks(rotation=90)

# fig.colorbar(im)
# plt.show()

