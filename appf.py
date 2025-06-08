'''
Author: Bappy Ahmed
Email: pvuganeza@gmail.com
Date:12-Oct-2021
'''

# from curses.textpad import rectangle
from pyexpat import features
from typing import AsyncGenerator
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.feature_extraction import img_to_graph
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image,ImageEnhance
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from src import db1,db2,db3
from src.db1 import create_tabledb1, add_data, view_all_data,conn1,view_unique_tasks,delete_data
from src.db2 import create_tabledb2, add_data,view_unique_tasks,view_all_data,delete_data
from src.db3 import create_table, add_data,view_all_data,view_unique_tasks,delete_data
import pandas as pd
from keras.preprocessing import image



config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(artifacts_dir, upload_image_dir)

# pickle_format_data_dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

#Feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)


#params_path
model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
input_shapes = params['base']['input_shape']
poolings = params['base']['pooling']

#detector = MTCNN()
model = VGGFace(model=model_name,include_top=include_tops,input_shape=(224,224,3),pooling=poolings)
feature_list = pickle.load(open(features_name,'rb'))
filenames = pickle.load(open(pickle_file,'rb'))

# save_uploaded_image
def save_uploaded_image(uploaded_image):
    try:
        create_directory(dirs=[uploadn_path])

        with open(os.path.join(uploadn_path,uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False
    
    
    
def imgpath(img_path,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    return results

# extract_features
def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]
            #  extract its features
    image = Image.fromarray(face)

    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()


            
    return result

    # =========================================================================================================
    # Function to extract features in an uploaded image:It will be cxalled whenever an image is uploaded or a frame is captured
@st.cache
def extract_features1(img,model,detector):#takes arguments(image, model(vggface),and detector(MTCNN) to detect a facxe in an image)
    img = cv2.imread(img)#This function will first read the image and put in binary or digital
    results = detector.detect_faces(img)# from MTCNN detect the face in the image
    
    x, y, width, height = results[0]['box']#for detected faces,choose first face and locxxate its coordinates(x,y and width and height)

    face = img[y:y + height, x:x + width] # crop the detected face

    #  extract its features
    image = Image.fromarray(face)#take the image from array
   
    image = image.resize((224, 224))#Resize the image

    face_array = np.asarray(image)# represent this image as an array again

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)# preprocess that image
    result = model.predict(preprocessed_img).flatten()#using vggface stored in model variable,predict the image

    
    return result#Once the function is called, it will return this predicted results from the model

    # ================================================================================
similarity = []
# recommend image: Funcxtion to compare uploaded image and the images in folders(image databases)
def recommend(feature_list,features):#arguments:features= extracted from uploded image, feature_list=features already extracxxted from images in all folders
    # similarity = []
    for i in range(len(feature_list)):#Loop through the feature_list (of all images) and compare each with the uploaded image's features 
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])#put similarities found for each in an array

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]#choose the highest similarity
    
    return index_pos

# similarity1 = []

def recommend1(feature_list,features,similarity1):
    # similarity = []
    for i in range(len(feature_list)):
        # similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
        similarity1.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity1)), reverse=True, key=lambda x: x[1])[0][0]
    # similarity.clear()
    return index_pos

# Function to add special html elements and css in streamlit
def markdownstreamlit(var,h):#arguments var = text in htmt tag, h= variable that can replace each html element
    st.markdown(f'<{h} style="background-color:#800000;color:#ffffff;font-size:24px;borde-radius:2%;">{var}</{h}>',unsafe_allow_html=True)
                
                # This function is for genereting new dataset and is called wnen a new client(not already in system)
def generate_dataset1(name,id):#funcxtion with two arguments(name of the cxustomer,and customer's id(in database))
    # create a classifier
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#This is a face detector,classifier file that comes with opencv 
    # function to crop the face
    def face_cropped(img):#argument is an image containing a face
        # change the image color into gray scale
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Changing the image to grayscale

# using the classifier/detector file, detect face from the image
        faces = face_classifier.detectMultiScale(img, 1.3, 5) # Calling the above face classifier and

        if faces is (): # if face is empty return none(nothing)
            return None
        # loop through faces if they are detected(one or more). 
        for (x,y,w,h) in faces:#for each face,crop the face at x+w and y+h positions
            cropped_face = img[y:y+h,x:x+w] # to crop img from y+h position to x+w
        return cropped_face
    # Opening my camera

    cap = cv2.VideoCapture(0) # Value 0 means I am going to use camera from laptop, 1 for external camera
    img_id = 0 # number of image for each authorized person (to be increased)
    
    while True:# as long as the camera is still opened or always open camera
        ret, frame = cap.read() # read the image from cap (camera)
        
        # The funcxtion face_cropped is called to crop the frame from video or camera
        if face_cropped(frame) is not None: # Here frame is an argument img(the same)
            # passing the image to be detected and once it is not None
            img_id+=1#Incxrease the id number because next we will deal with the second frame, so give them id s.
            # Resizing the image(frame)
            face = cv2.resize(face_cropped(frame), (64,64))
            # converting it to grayscale 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Locating the path to be stored later
            file_name_path = f"data/{name}/{name}."+str(id)+"."+str(img_id)+".jpg"
            # Saving the image in a folder
            cv2.imwrite(file_name_path, face)
            # Put(write some text) the text in my cropped image
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Cropped face", face)

        # Here if you hit enter key(13) or number of taken images =120,stop this funcxtion
        if cv2.waitKey(1)==13 or int(img_id)==120: #13 is the ASCII character of Enter key(break once enter key is pressed)
            # break it when enter key is pressed or when img_id (number of images is = 200)=200
            break
#  release my camera and destroy all windows: Kill everything       
    cap.release()
    cv2.destroyAllWindows()
    st.title(" DATASET")
    st.success("INSURANCE:, Generating Dataset Completed !!!")

#==============================================================================================

# def face_cropped(img):
#     face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Changing the image to grayscale
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5) # Calling the above face classifier and

#     if faces is (): # if face is empty return none
#         return None
#     # loop through faces if it is detected. 
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)#adding a recxtangle
#         cropped_face = img[y:y+h,x:x+w] # to cop img from y+h position to x+w
#     return cropped_face

# Function main to be called when you run this file
def main():
    st.title('Insurance Customer Service in Rwanda:Face Recognition')#using streamlit to write a title
    menu1 = ["DETECTION","DATABASE&DATASET"]#A list of choices
    choice1 = st.sidebar.selectbox("Please Choose a Task: ",menu1)#selecting a sidebar in streamlit and passing a menu1 list of choices
    # st.title("Improvement of Customer Service of Insurance in Rwanda Using Face Recognition")
    menu2 = ["Create","Read","Delete"] #list of task choices
    menu3 = ["RSSB","SORAS","PRIME"]#list of database choices
    
    if choice1=="DATABASE&DATASET":#when This option from list of choices is choosen
        choice3 = st.sidebar.selectbox("Please Choose a db company: ",menu3)#cxreating a side bar for selecting a database
        if choice3=="RSSB":
            db3.create_table()
        
            st.title("INSURANCE: RSSB")
            # Detecting Buttons.
            choice2 = st.sidebar.selectbox("Please Choose a Task: ",menu2)
            if choice2 == "Create":
                st.subheader("Add a user/customer")

                # Cxreating 3 cxolumns in streamlit
                col1,col2,col3 = st.columns(3)
                with col1 :#in fist column
                    name = st.text_area("Add a Name")
                    # name=st.text_input("Enter Names:","Type Here")
                    age =st.number_input("Age: ")
                    # age=st.number_input("Enter your age")
                with col2:#in second column
                    # address = st.text_area("Address: ")
                    address=st.text_input("Enter your address:")    
                    # identity_numb = st.number_input("Identity Number: ")
                    identity_numb=st.number_input("Enter your Id card")
                    
                with col3:#Third column
                    task_due_date = st.date_input("Due Date")
                    photo = st.file_uploader("Choose an image",type=['png', 'jpg', 'jpeg'])

                    if photo is not None:
                        bytes_data = photo.getvalue()
                        photo = photo.read()
                        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #=========================================================================
                        st.write('Your IMAGE')
                        st.image(img, use_column_width=True)
                        
                if (name is None) or (address is None) or (task_due_date is None) or (photo is None): 
                    st.success("Please provide complete details of the user !!!")
                else:
                    
                    if st.button("Need dataset"):               
                        id = 1   
                        db3.create_table()
                        myResult = db3.view_all_data()
                        
                        for row in myResult:
                            id +=1
                        os.makedirs(f"data/{name}", exist_ok=True)
                        generate_dataset1(name,id)
                        db3.add_data(id,name,address,age,identity_numb,task_due_date,photo)
                        st.success("Data Added!!!")                        
                      
                    if st.button("Just Register"):
                        ids = 1   
                        db3.create_table()
                        myResult = db3.view_all_data()
                        
                        for row in myResult:
                            ids +=1
                        #os.makedirs(f"data/{name}", exist_ok=True)
                        db3.add_data(ids,name,address,age,identity_numb,task_due_date,photo)
                        st.success("Data Added!!!")                                           
                        
                if st.button("Trainining"):

                    # TRAINING IS HERE: These files are now commented to fight against disturbances of unwanted trainings
                    # =====================================================================================================

                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/01_generate_img_pkl.py").read())
                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/02_feature_extractor.py").read())
                    
                    
                
            elif choice2 == "Read":
                st.subheader("View customers")
                result = db3.view_all_data()
                # st.write(result)
                df = pd.DataFrame(result,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                st.write(df)
            # elif choice2 == "Update":
            #     pass
            elif choice2 == "Delete":
                st.subheader("Delete an Item")
                # from db3 import create_table,view_all_data,view_unique_tasks,delete_data
                
                result = db3.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age","ID_Number","Due Date","photo"])
                with st.expander("view current data"):
                    st.dataframe(df)
                # 
                list_of_tasks = [i[0] for i in db3.view_unique_tasks()]
                selected_task = st.selectbox("Task to Delete",list_of_tasks)
                st.warning("Do you want to delete:: {} ?".format(selected_task))
                if st.button("Delete"):
                    
                    db3.delete_data(selected_task)
                    st.success("Task has successifully deleted")
                # To view data after deletion
                result = db3.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age", "ID_Number","Due Date","photo"])

                with st.expander("view current data After deletion"):
                    st.dataframe(df)
            
            # elif choice2 == "About":
            #     st.write("ABOUT YOU!!!!!!!!!!!")
        elif choice3=="SORAS":
            db1.create_tabledb1()
            st.title("INSURANCE: SORAS INGABO Y'AMAHINA")
            # Detecting Buttons.
            choice2 = st.sidebar.selectbox("Please Choose a Task: ",menu2)
            if choice2 == "Create":
                st.subheader("Add a user/customer")
                col1,col2,col3 = st.columns(3)
                with col1 :
                    name = st.text_area("Add a Name")
                    age =st.number_input("Age: ")
                with col2:
                    address = st.text_area("Address: ")    
                    identity_numb = st.number_input("Identity Number: ")
                    
                with col3:
                    task_due_date = st.date_input("Due Date")
                    photo = st.file_uploader("Choose an image",type=['png', 'jpg', 'jpeg'])

                    if photo is not None:
                        bytes_data = photo.getvalue()
                        photo = photo.read()
                        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #=========================================================================
                        st.write('Your IMAGE')
                        st.image(img, use_column_width=True)
                        
                if (name is None) or (address is None) or (task_due_date is None) or (photo is None): 
                    st.success("Please provide complete details of the user !!!")
                else:
                    
                    if st.button("Need dataset"):               
                        id = 1   
                        db1.create_tabledb1()
                        myResult = db1.view_all_data()
                        
                        for row in myResult:
                            id +=1
                        os.makedirs(f"data/{name}", exist_ok=True)
                        generate_dataset1(name,id)
                        db1.add_data(id,name,address,age,identity_numb,task_due_date,photo) 
                        st.success("Data Added!!!")                       
                      
                    if st.button("Just Register"):
                        ids = 1   
                        db1.create_tabledb1()
                        myResult = db1.view_all_data()
                        
                        for row in myResult:
                            ids +=1
                        #os.makedirs(f"data/{name}", exist_ok=True)
                        db1.add_data(ids,name,address,age,identity_numb,task_due_date,photo) 
                        st.success("Data Added!!!")                                           
                        
                        
                if st.button("Trainining"):

                    # TRAINING IS HERE: These files are now commented to fight against disturbances of unwanted trainings
                    # =====================================================================================================

                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/01_generate_img_pkl.py").read())
                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/02_feature_extractor.py").read())
            
            elif choice2 == "Read":
                # st.title("INSURANCE: RSSB")
                st.subheader("View customers")
                result = db1.view_all_data()
                # st.write(result)
                df = pd.DataFrame(result,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                st.write(df)
            # elif choice2 == "Update":
            #     pass
            elif choice2 == "Delete":
                st.subheader("Delete an Item")
                result = db1.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age","ID_Number","Due Date","photo"])
                with st.expander("view current data"):
                    st.dataframe(df)
                # 
                list_of_tasks = [i[0] for i in db1.view_unique_tasks()]
                selected_task = st.selectbox("Task to Delete",list_of_tasks)
                st.warning("Do you want to delete:: {} ?".format(selected_task))
                if st.button("Delete"):
                    
                    db1.delete_data(selected_task)
                    st.success("Task has successifully deleted")
                # To view data after deletion
                result = db1.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age", "ID_Number","Due Date","photo"])

                with st.expander("view current data After deletion"):
                    st.dataframe(df)
            
            # elif choice2 == "About":
            #     st.write("ABOUT YOU!!!!!!!!!!!")
        elif choice3=="PRIME":
            db2.create_tabledb2()
            
            st.title("INSURANCE: PRIME INSURANCE")
            # Detecting Buttons.
            choice2 = st.sidebar.selectbox("Please Choose a Task: ",menu2)
            if choice2 == "Create":
                st.subheader("Add a user/customer")
                col1,col2,col3 = st.columns(3)
                with col1 :
                    name = st.text_area("Add a Name")
                    age =st.number_input("Age: ")
                with col2:
                    address = st.text_area("Address: ")    
                    identity_numb = st.number_input("Identity Number: ")
                    
                with col3:
                    task_due_date = st.date_input("Due Date")
                    photo = st.file_uploader("Choose an image",type=['png', 'jpg', 'jpeg'])

                    if photo is not None:
                        bytes_data = photo.getvalue()
                        photo = photo.read()
                        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #=========================================================================
                        st.write('Your IMAGE')
                        st.image(img, use_column_width=True)
                        
                if (name is None) or (address is None) or (task_due_date is None) or (photo is None): 
                    st.success("Please provide complete details of the user !!!")
                else:
                    
                    if st.button("Need dataset"):               
                        id = 1   
                        db2.create_tabledb2()
                        myResult = db2.view_all_data()
                        
                        for row in myResult:
                            id +=1
                        os.makedirs(f"data/{name}", exist_ok=True)
                        generate_dataset1(name,id)
                        db2.add_data(id,name,address,age,identity_numb,task_due_date,photo) 
                        st.success("Data Added!!!")                       
                      
                    if st.button("Just Register"):
                        ids = 1   
                        db2.create_tabledb2()
                        myResult = db2.view_all_data()
                        
                        for row in myResult:
                            ids +=1
                        #os.makedirs(f"data/{name}", exist_ok=True)
                        db2.add_data(ids,name,address,age,identity_numb,task_due_date,photo) 
                        st.success("Data Added!!!")                                           
                            
                if st.button("Trainining"):

                    # TRAINING IS HERE: These files are now commented to fight against disturbances of unwanted trainings
                    # =====================================================================================================

                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/01_generate_img_pkl.py").read())
                    exec(open("C:/ACEDS docs/Face recognition/face-rwanda/src/02_feature_extractor.py").read())
                
            elif choice2 == "Read":
                st.subheader("View customers")
                db2.create_tabledb2()
                result = db2.view_all_data()
                # st.write(result)
                df = pd.DataFrame(result,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                st.write(df)
            # elif choice2 == "Update":
            #     pass
            elif choice2 == "Delete":
                st.subheader("Delete an Item")
                db2.create_tabledb2()
                result = db2.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age","ID_Number","Due Date","photo"])
                with st.expander("view current data"):
                    st.dataframe(df)
                # 
                list_of_tasks = [i[0] for i in db2.view_unique_tasks()]
                selected_task = st.selectbox("Task to Delete",list_of_tasks)
                st.warning("Do you want to delete:: {} ?".format(selected_task))
                if st.button("Delete"):
                    
                    db2.delete_data(selected_task)
                    st.success("Task has successifully deleted")
                # To view data after deletion
                result = db2.view_all_data()
                # st.write(result2)
                df = pd.DataFrame(result,columns=["id","name","address","age", "ID_Number","Due Date","photo"])

                with st.expander("view current data After deletion"):
                    st.dataframe(df)

    elif choice1=="DETECTION":
        prediction_mode = st.sidebar.radio(
            "",
            ('Single image', 'Web camera'),
            index=0)
        if prediction_mode == 'Single image':
            
            # streamlit
            #st.title('Which Bollywood Celebrity You look like?')
            st.subheader('Who is This Person ?')

            uploaded_image = st.file_uploader('Choose an image')
            face_cascade = cv2.CascadeClassifier("C:/ACEDS docs/Face recognition/face-rwanda/haarcascade_frontalface_default.xml")

            if uploaded_image is not None:
                # save the image in a directory
                if save_uploaded_image(uploaded_image):

                    display_image = Image.open(uploaded_image)
                    detector = MTCNN()
                    results = imgpath(os.path.join(uploadn_path,uploaded_image.name),detector)
                    if results:
                    # extract the features
                        features = extract_features(os.path.join(uploadn_path,uploaded_image.name),model,detector)
                        # recommend
                        index_pos = recommend(feature_list,features)
                        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                        # display
                        st.write("Accessing Highest Similarity Cosine")
                        st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

                        similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
                        threshold_similarity = 55
                        
                        if similarity_big_index >= threshold_similarity:
                        
                            col1,col2 = st.columns(2)

                            with col1:
                                st.header('Detected Face in uploaded image')
                                img = cv2.imread(os.path.join(uploadn_path,uploaded_image.name))
                                display_image1=np.array(display_image)
                                x, y, width, height = results[0]['box']

                                face = img[y:y + height, x:x + width]
                                cv2.rectangle(display_image1,(x,y),(x+width,y+height),(255,0,0),10)
                                cv2.putText(display_image1, predicted_actor, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                                st.image(display_image1)
                            with col2:
                                st.header("This Face matches " + predicted_actor)
                                st.image(filenames[index_pos],width=300)
                                
                            # st.write(predicted_actor)
                            # # ================================================================
                            st.markdown("================================================================================================================================================================================")
                            # db1.conn1
                            db1.create_tabledb1()
                            cc = db1.conn1.cursor()
                            # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
                            cc.execute("SELECT* FROM tasksTabledb1")
                            m1 = cc.fetchall() #tuple


                            # db2.conn2
                            db2.create_tabledb2()
                            cb = db2.conn2.cursor()
                            cb.execute("SELECT* FROM tasksTabledb2")
                            m2 = cb.fetchall() #tuple

                            # db3.conn3
                            db3.create_table()
                            ca = db3.conn3.cursor()
                            # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
                            ca.execute("SELECT* FROM tasksTable")
                            m3 = ca.fetchall() #tuple

                            count1=0
                            for k1 in m1: # looping through rows
                                if k1[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                    count1 +=1
                                    
                            count2=0
                            for k2 in m2: # looping through rows
                                if k2[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                    count2 +=1

                            count3=0
                            for k3 in m3: # looping through rows
                                if k3[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                    count3 +=1

                            if count1==1:
                                st.subheader("Company: SORAS")
                                # df = pd.DataFrame(k1,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                                # st.title(st.write(df))
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    id1 = k1[0]
                                    name1 = k1[1]
                                    address1 = k1[2]
                                    st.subheader("ID")
                                    # st.markdown("###====================:")
                                    st.write(id1)

                                    st.subheader("NAME")
                                    # st.markdown("###====================:")
                                    st.write(name1)
                                    # st.write("NAME:",st.subheader(name1))
                                    st.subheader("ADDRESS")
                                    # st.markdown("###====================:")
                                    st.write(address1)
                                with col2:
                                    age1 = k1[3]
                                    identity1 = k1[4]
                                    date1 = k1[5]
                                    st.subheader("AGE")
                                    # st.markdown("###====================:")
                                    st.write(age1)

                                    st.subheader("ID CARD")
                                    # st.markdown("###====================:")
                                    st.write(identity1)
                                    st.subheader("DATE")
                                    # st.markdown("###====================:")
                                    st.write(date1)  
                                with col3:
                                    photo1 = k1[6]
                                    file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                    img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    st.image(img, use_column_width = True)

                                
                            elif count2==1:
                                st.subheader("Company: PRIME INSURANCE")
                                # df = pd.DataFrame(k2,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                                # st.title(st.write(df))
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    id1 = k2[0]
                                    name1 = k2[1]
                                    address1 = k2[2]
                                    st.subheader("ID")
                                    # st.markdown("====================:")
                                    st.write(id1)

                                    st.subheader("NAME")
                                    # st.markdown("====================:")
                                    st.write(name1)
                                    # st.write("NAME:",st.subheader(name1))
                                    st.subheader("ADDRESS")
                                    # st.markdown("====================:")
                                    st.write(address1)
                                with col2:
                                    age1 = k2[3]
                                    identity1 = k2[4]
                                    date1 = k2[5]
                                    st.subheader("AGE")
                                    # st.write("====================:")
                                    st.write(age1)

                                    st.subheader("ID CARD")
                                    # st.markdown("====================:")
                                    st.write(identity1)
                                    st.subheader("DATE")
                                    # st.markdown("====================:")
                                    st.write(date1)
                                with col3:
                                    
                                    photo1 = k2[6]
                                    file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                    img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    st.image(img, use_column_width = True)
                            elif count3==1:
                                st.subheader("Company: RSSB")
                                # df = pd.DataFrame(k3,columns=["id","name", "address","age","identity_numb","Due Date","photo"])
                                # st.title(st.write(df))
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    id1 = k3[0]
                                    name1 = k3[1]
                                    address1 = k3[2]
                                    st.subheader("ID")
                                    # st.markdown("====================:")
                                    st.write(id1)

                                    st.subheader("NAME")
                                    # st.markdown("====================:")
                                    st.write(name1)
                                    # st.write("NAME:",st.subheader(name1))
                                    st.subheader("ADDRESS")
                                    # st.markdown("====================:")
                                    st.write(address1)
                                with col2:
                                    age1 = k3[3]
                                    identity1 = k3[4]
                                    date1 = k3[5]
                                    st.subheader("AGE")
                                    # st.markdown("====================:")
                                    st.write(age1)

                                    st.subheader("ID CARD")
                                    # st.markdown("====================:")
                                    st.write(identity1)
                                    st.subheader("DATE")
                                    # st.markdown("====================:")
                                    st.write(date1)
                                with col3:
                                    
                                    photo1 = k3[6]
                                    file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                    img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    st.image(img, use_column_width = True)

                            else:
                                st.title("Not Found in Any Databases")
                            
                            # =============================================================================
                        else:
                            st.write("UNKNOWN PERSON") 
                            col3,col4 = st.columns(2)

                            with col3:
                                st.header('Your uploaded image')
                                img = cv2.imread(os.path.join(uploadn_path,uploaded_image.name))
                                display_image1=np.array(display_image)
                                x, y, width, height = results[0]['box']

                                face = img[y:y + height, x:x + width]
                                cv2.rectangle(display_image1,(x,y),(x+width,y+height),(255,0,0),10)
                                cv2.putText(display_image1, "Unknown", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                                st.image(display_image1)
                            with col4:
                                st.title("UNKNOWN")
                                p="h1"
                                markdownstreamlit("UNKNOWN PERSON",p)  

                    else:
                        col1,col2=st.columns(2)
                        with col1:
                            st.header('Your uploaded image')
                            st.image(display_image)
                        with col2:
                            # st.markdown(f'<p style="background-color:#ffc0cb;">{col2}</p>',unsafe_allow_html=True)
                            p="p"
                            markdownstreamlit(("No Face Found:Maybe not a person"),p)
                            # st.error("No Face Found:Maybe not a person")
                        
                                            
            else:
                st.subheader("Please Upload an image")
                     
                        
            
        elif prediction_mode == 'Web camera':
            st.subheader('Who is this person ?')
            if st.button("Detect Face"):
                cap = cv2.VideoCapture(0) #webcam
                #cap = cv2.VideoCapture('C:/Users/IS96273/Desktop/zuckerberg.mp4') #video
                face_cascade = cv2.CascadeClassifier("C:/ACEDS docs/Face recognition/face-rwanda/haarcascade_frontalface_default.xml")
                img_id=0
                while(True):
                    ret, img = cap.read()
                    #img = cv2.resize(img, (640, 360))
                    faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    
                    if faces is not (): # if face is empty return none
                        # return None
                
                        for (x,y,w,h) in faces:
                            # if w > 130: 
                            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
                            
                            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                            file_name_path = "C:/ACEDS docs/Face recognition/face-rwanda/artifacts/faces/user."+str(img_id)+".jpg"
        #                   # Saving the image in a folder
                            cv2.imwrite(file_name_path, detected_face)
                            img_id +=1
                            display_image = Image.open(file_name_path)
                            image = Image.fromarray(detected_face)
                            # image.tile =[e for e in image.tile if e[1][2]<2181 and e[1][3]<1294]
                            image = image.resize((224, 224))

                            face_array = np.asarray(image)

                            face_array = face_array.astype('float32')

                            expanded_img = np.expand_dims(face_array, axis=0)
                            preprocessed_img = preprocess_input(expanded_img)
                            result = model.predict(preprocessed_img).flatten()
                            # st.write(result)
                            similarity1 = []
                            similarity1.clear()
                            # recommend
                            index_pos = recommend1(feature_list,result,similarity1)
                            # st.write(index_pos)
                            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))


                            # display
                            # st.write("look at Here")
                            # st.write(index_pos)
                            # st.write(predicted_actor)
                            st.write("AccessingHighest Similarity Cosine")
                            st.write(sorted(list(enumerate(similarity1)), reverse=True, key=lambda x: x[1])[0][1])

                            similarity_big_index = (sorted(list(enumerate(similarity1)), reverse=True, key=lambda x: x[1])[0][1])*100
                            threshold_similarity = 55
                            
                            if similarity_big_index >= threshold_similarity:
                            
                                col1,col2 = st.columns(2)

                                with col1:
                                    st.header('Detected Face/image')
                                    st.image(display_image)
                                with col2:
                                    st.header("This Face is for " + predicted_actor)
                                    display_image1=np.array(filenames[index_pos])
                                    #cv2.putText(filenames[index_pos],predicted_actor, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                                    st.image(filenames[index_pos],width=300)

                                st.markdown("================================================================================================================================================================================")
                                
                                db1.create_tabledb1()
                                cc = db1.conn1.cursor()
                                # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
                                cc.execute("SELECT* FROM tasksTabledb1")
                                m1 = cc.fetchall() #tuple



                                
                                db2.create_tabledb2()
                                cb = db2.conn2.cursor()
                                cb.execute("SELECT* FROM tasksTabledb2")
                                m2 = cb.fetchall() #tuple

                               
                                db3.create_table()
                                ca = db3.conn3.cursor()
                                # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
                                ca.execute("SELECT* FROM tasksTable")
                                m3 = ca.fetchall() #tuple

                                count1=0
                                for k1 in m1: # looping through rows
                                    if k1[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                        count1 +=1
                                        
                                count2=0
                                for k2 in m2: # looping through rows
                                    if k2[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                        count2 +=1

                                count3=0
                                for k3 in m3: # looping through rows
                                    if k3[1]==predicted_actor:# for each row if fist column == id(predicted person)
                                        count3 +=1

                                if count1==1:
                                    st.subheader("Company: SORAS")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        id1 = k1[0]
                                        name1 = k1[1]
                                        address1 = k1[2]
                                        st.subheader("ID")
                                        # st.markdown("====================:")
                                        st.write(id1)
                                        st.subheader("NAME")
                                        # st.markdown("====================:")
                                        st.write(name1)
                                        st.subheader("ADDRESS")
                                        # st.markdown("====================:")
                                        st.write(address1)
                                    with col2:
                                        age1 = k1[3]
                                        identity1 = k1[4]
                                        date1 = k1[5]
                                        st.subheader("AGE")
                                        # st.write("====================:")
                                        st.write(age1)
                                        st.subheader("ADDRESS")
                                        # st.markdown("====================:")
                                        st.write(identity1)
                                        st.subheader("DATE")
                                        # st.markdown("====================:")
                                        st.write(date1)
                                        
                                    with col3:
                                        
                                        photo1 = k1[6]
                                        file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                        img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img, use_column_width = True)

                                    
                                elif count2==1:
                                    st.subheader("Company: PRIME")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        id1 = k2[0]
                                        name1 = k2[1]
                                        address1 = k2[2]
                                        st.subheader("ID")
                                        # st.markdown("====================:")
                                        st.write(id1)
                                        st.subheader("NAME")
                                        # st.markdown("====================:")
                                        st.write(name1)
                                        st.subheader("ADDRESS")
                                        # st.markdown("====================:")
                                        st.write(address1)
                                        
                                    with col2:
                                        age1 = k2[3]
                                        identity1 = k2[4]
                                        date1 = k2[5]

                                        st.subheader("AGE")
                                        # st.markdown("====================:")
                                        st.write(age1)
                                        st.subheader("ID CARD")
                                        # st.markdown("====================:")
                                        st.write(identity1)
                                        st.subheader("DATE")
                                        # st.markdown("====================:")
                                        st.write(date1)
                                        
                                    with col3:
                                        
                                        photo1 = k2[6]
                                        file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                        img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img, use_column_width = True)
                                elif count3==1:
                                    st.subheader("Company: RSSB")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        id1 = k3[0]
                                        name1 = k3[1]
                                        address1 = k3[2]
                                        st.subheader("ID")
                                        # st.markdown("====================:")
                                        st.write(id1)
                                        st.subheader("NAME")
                                        # st.markdown("====================:")
                                        st.write(name1)
                                        st.subheader("ADDRESS")
                                        # st.markdown("====================:")
                                        st.write(address1)
                                       
                                    with col2:
                                        age1 = k3[3]
                                        identity1 = k3[4]
                                        date1 = k3[5]
                                        st.title(age1)
                                        st.subheader(identity1)
                                        st.subheader(date1)
                                    with col3:
                                        
                                        photo1 = k3[6]
                                        file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                                        img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img, use_column_width = True)
                            
                            else:
                                st.write("UNKNOWN PERSON") 
                                col3,col4 = st.columns(2)

                                with col3:
                                    st.title('This face is not known')
                                    st.image(display_image)
                                with col4:
                                    st.title("UNKNOWN")                 
                    if cv2.waitKey(1)==13 or int(img_id)==1:
                            #13 is the ASCII character of Enter key(break once enter key is pressed)
                        break
                #kill open cv things		
                cap.release()
                cv2.destroyAllWindows()            

if __name__== '__main__':
    main()

