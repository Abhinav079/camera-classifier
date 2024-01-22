import pickle
import os
import os.path
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog, simpledialog

import cv2 as cv
import numpy as np
import PIL
import PIL.Image, PIL.ImageDraw
import PIL.ImageTk
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = 0, 0, 0
        self.clf = None
        self.proj_name = None
        self.root = None
        self.camera_frame = None
        self.cap = None  # Camera capture object

        self.status_label = None

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = tk.Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name!", parent=msg)
        if not self.proj_name:
            self.proj_name = "DefaultProject"

        if os.path.exists(self.proj_name):
            pickle_file_path = os.path.join(self.proj_name, f'{self.proj_name}_data.pickle')
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
            self.class1, self.class2, self.class3 = data['c1'], data['c2'], data['c3']
            self.class1_counter, self.class2_counter, self.class3_counter = data['c1c'], data['c2c'], data['c3c']
            self.clf = data['clf']
        else:
            self.class1 = simpledialog.askstring('Class1', 'What is the first class called?', parent=msg)
            self.class2 = simpledialog.askstring('Class2', 'What is the second class called?', parent=msg)
            self.class3 = simpledialog.askstring('Class3', 'What is the third class called?', parent=msg)

            self.clf = LinearSVC()

            os.makedirs(self.proj_name, exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class1), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class2), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class3), exist_ok=True)

        msg.destroy()

    def init_gui(self):
        self.root = tk.Tk()
        self.root.title(f'Doodle Detect - {self.proj_name}')

        # Initialize camera capture
        self.cap = cv.VideoCapture(0)

        # Camera frame label
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_frame.columnconfigure(0, weight=1)        
        btn_frame.columnconfigure(1, weight=1)                
        btn_frame.columnconfigure(2, weight=1)  

        self.init_buttons(btn_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def init_buttons(self, frame):
        class1_btn = tk.Button(frame, text=self.class1, command=lambda: self.save(1))
        class2_btn = tk.Button(frame, text=self.class2, command=lambda: self.save(2))
        class3_btn = tk.Button(frame, text=self.class3, command=lambda: self.save(3))

        class1_btn.grid(row=0, column=0, sticky=tk.W+tk.E)
        class2_btn.grid(row=0, column=1, sticky=tk.W+tk.E)
        class3_btn.grid(row=0, column=2, sticky=tk.W+tk.E)

        train_btn = tk.Button(frame, text='Train Model', command=self.train_model)
        train_btn.grid(row=1, column=0,sticky=tk.W+tk.E)

        predict_btn = tk.Button(frame, text='Predict', command=self.predict)
        predict_btn.grid(row=1, column=1, sticky=tk.W+tk.E)

        save_btn = tk.Button(frame, text='Save Model', command=self.save_model )
        save_btn.grid(row=1, column=2, sticky=tk.W+tk.E)        

        load_btn = tk.Button(frame, text='Load Model', command=self.load_model )
        load_btn.grid(row=2, column=0, sticky=tk.W+tk.E)

        change_btn = tk.Button(frame, text='Change Model', command=self.rotate_model )
        change_btn.grid(row=2, column=1, sticky=tk.W+tk.E)

        save_everything_btn = tk.Button(frame, text='Save Everything', command=self.save_everything )
        save_everything_btn.grid(row=2, column=2,sticky=tk.W+tk.E )     

        self.status_label = tk.Label(frame, text=f'Current Model: {type(self.clf).__name__}')
        self.status_label.grid(row=3, column=1, sticky=tk.W+tk.E)

        # Start the camera update loop
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.camera_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.camera_frame = cv.resize(self.camera_frame, (500, 500))
            self.camera_image = PIL.Image.fromarray(self.camera_frame)
            self.camera_image_tk = PIL.ImageTk.PhotoImage(image=self.camera_image)
            self.camera_label.config(image=self.camera_image_tk)
        self.root.after(10, self.update_camera)

    def save(self, class_num):
        if self.camera_frame is not None:
            img = PIL.Image.fromarray(self.camera_frame)
            img.thumbnail((50, 50), PIL.Image.BICUBIC)
            file_path = os.path.join(self.proj_name, self.class1 if class_num == 1 else self.class2 if class_num == 2 else self.class3, f'{self.class1_counter if class_num == 1 else self.class2_counter if class_num == 2 else self.class3_counter}.png')
            img.save(file_path, 'PNG')
            if class_num == 1:
                self.class1_counter += 1
            elif class_num == 2:
                self.class2_counter += 1
            elif class_num == 3:
                self.class3_counter += 1

    # ... [Other methods like train_model, predict, save_model, load_model, rotate_model, etc.]

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        # Load images and labels for training
        for class_num, (class_name, class_counter) in enumerate([(self.class1, self.class1_counter), (self.class2, self.class2_counter), (self.class3, self.class3_counter)], 1):
            for i in range(1, class_counter):
                file_path = os.path.join(self.proj_name, class_name, f'{i}.png')
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                img = img.reshape(-1)
                img_list = np.append(img_list, [img])
                class_list = np.append(class_list, class_num)

        # Reshape and fit model
        if len(img_list) > 0:
            img_list = img_list.reshape(-1, 2500)  # Assuming images are 50x50
            self.clf.fit(img_list, class_list)
            tkinter.messagebox.showinfo('Doodle Detect', 'Model Successfully Trained', parent=self.root)

    def predict(self):
        if self.camera_frame is not None:
            img = PIL.Image.fromarray(self.camera_frame)
            img.thumbnail((50, 50), PIL.Image.BICUBIC)
            img = np.array(img.convert('L'))  # Convert to grayscale
            img = img.reshape(1, -1)  # Reshape for prediction
            prediction = self.clf.predict(img)
            class_name = self.class1 if prediction[0] == 1 else self.class2 if prediction[0] == 2 else self.class3
            tkinter.messagebox.showinfo('Doodle Detect Drawing Classifier', f'The drawing is probably a {class_name}', parent=self.root)

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()            
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()            
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()
        self.status_label.config(text = f'Current Model: {type(self.clf).__name__}')    

    def save_model(self):
        # Ask the user for the file path with a default extension
        file_path = filedialog.asksaveasfilename(defaultextension='.pickle')
    
        # Check if the user canceled the file dialog
        if not file_path:
            return
    
        # Save the model to the specified file
        with open(file_path, 'wb') as f:
            pickle.dump(self.clf, f)
    
        # Display a success message
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier', 'Model Successfully Saved', parent=self.root)


    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, 'rb') as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier','Model Successfully loaded', parent=self.root)    


    def save_everything(self):
        data = {'c1': self.class1, 'c2': self.class2, 'c3': self.class3, 'c1c': self.class1_counter, 'c2c': self.class1_counter,
                'c3c': self.class1_counter, 'clf': self.clf, 'pname': self.proj_name}
        with open(f'{self.proj_name}/{self.proj_name}_data.pickle','wb') as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier','Project Successfully Saved', parent=self.root)    
                  


    def on_closing(self):
        # Ask to save everything before closing
        if tkinter.messagebox.askokcancel("Quit", "Do you want to save your work?", parent=self.root):
            self.save_everything()
        # Release the camera
        if self.cap:
            self.cap.release()
        self.root.destroy()


DrawingClassifier()
