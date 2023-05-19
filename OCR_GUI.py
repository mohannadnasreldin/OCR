import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dataSet = pd.read_csv('D:/college/Level3/AI/project/AIP/Dataset/archive/emnist-letters-train.csv')

dataSet.rename(columns={'23':'label'}, inplace=True)
#_____________________________________________

#X is a numpy list that contains all the columns after 0 (the images) 
x = dataSet.iloc[:,1:].values
#y is a numpy list that contains the column 0 (the labels) 
y = dataSet.iloc[:,0].values

kf = KFold(n_splits=150)

for train_index, test_index in kf.split(x):
    x_train,x_test = x[train_index], x[test_index]
    y_train,y_test = y[train_index], y[test_index]


#________________________
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#to start using the random forest classifier so we can train the algorithm using the data we are preparing
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=150, criterion = 'entropy', random_state=42, min_samples_split =6, min_samples_leaf=1, max_depth=90, bootstrap=False)


#start using decision trees model
from  sklearn.tree  import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

#_________________________________________

#start the training
rf_model.fit(x_train,y_train)
dt_model.fit(x_train,y_train)

#_____________________________________________
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2


root = Tk(  )

#______________________________________________________



def OCR_PREPROC(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to make it in gray scale

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) #to make it binary

    # helps in detecting the chars
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

    #find the Contours of an image then create boundingRect 
    ctrs, hier= cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    images_char = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr) #to get x-axis, y-axis, width, hight.

        roi = img[y:y + h, x:x + w] #it gets the position of every char

        area = w*h  #area of the char

        if area > 250 : # we want the area that is bigger than 250 pixel to ignor all the small things on the image
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) #draw the rectangle
            images_char.append(gray[y: y + h, x :x + w ]) #take all chars from the gray image and append to list  


        #change the size to 28x28 and convert it to vector
        for i in range(len(images_char)):
            images_char[i] = cv2.resize(images_char[i],(28,28)).flatten()

    return images_char
    
    #import matplotlib.pyplot as plt
    #befor detecting all chars 
    #plt.imshow(thresh,cmap='gray')

    #after detecting all chars 
    #plt.imshow(rect)

#______________________________________________________

def OCR_RF(images_char):
    #transform StandardScaler
    images_char = ss.transform(images_char)

    #the detected word in numbers
    rf_word_in_num =  rf_model.predict(images_char)
    
    
    #change array of numbers to string
    rf_out = ''
    for x in rf_word_in_num:
        rf_out += chr(64+x)
    return rf_out
#______________________________________________    

def OCR_DT(images_char):
    images_char = ss.transform(images_char)
    #the detected word in numbers
    
    dt_word_in_num =  dt_model.predict(images_char)
    #print(dt_word_in_num)
    #change array of numbers to string
    dt_out = ''
    for x in dt_word_in_num:
        dt_out += chr(64+x)
    return dt_out



#_______________________________________________
def readFimage():
    path = PathTextBox.get('1.0','end-1c')
    if path:
        #im = Image.open(path)
        #text = pytesseract.image_to_string(im, lang = 'eng')
        textdt = OCR_DT(OCR_PREPROC(path))
        textrf = OCR_RF(OCR_PREPROC(path))
        ResultTextBox.delete('1.0',END)
        ResultTextBox.insert(END,'Decision Tree Prediction\n')
        ResultTextBox.insert(END,textdt)
        ResultTextBox.insert(END,'\nRandom Forest Prediction\n')
        ResultTextBox.insert(END,textrf)
    else:
        ResultTextBox.delete('1.0',END)
        ResultTextBox.insert(END,"FILE CANNOT BE READ")

#_______________________________________________________________________    

def OpenFile():
    name = askopenfilename(initialdir="/",
                           filetypes =(("PNG File", "*.png"),("BMP File", "*.bmp"),("JPEG File", "*.jpeg")),
                           title = "Choose a file."
                           ) 
    PathTextBox.delete("1.0",END)
    PathTextBox.insert(END,name)

#_______________________________________________

Title = root.title( "Image Reader!")
path = StringVar()

HeadLabel1 = Label(root,text="Image ")
HeadLabel1.grid(row = 1,column = 1,sticky=(E))
HeadLabel2 = Label(root,text=" Reader")
HeadLabel2.grid(row = 1,column = 2,sticky=(W))

InputLabel = Label(root,text = "INPUT IMAGE:")
InputLabel.grid(row=2,column = 1)

BrowseButton = Button(root,text="Browse",command = OpenFile)
BrowseButton.grid(row=2,column=2)

PathLabel = Label(root,text = "Path:")
PathLabel.grid(row = 3,column=1,sticky=(W))

PathTextBox = Text(root,height = 2)
PathTextBox.grid(row = 4,column = 1,columnspan=2)

ReadButton = Button(root,text="READ FROM IMAGE",command = readFimage)
ReadButton.grid(row = 5,column = 2)

DataLabel = Label(root,text = "DATA IN IMAGE:")
DataLabel.grid(row = 6,column=1,sticky=(W))

ResultTextBox = Text(root,height = 6)
ResultTextBox.grid(row = 7,column = 1,columnspan=2)



root.mainloop()