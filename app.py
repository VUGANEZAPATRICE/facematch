'''
Author: Bappy Ahmed
Email: pvuganeza@gmail.com
Date:12-Oct-2021
'''

from typing import AsyncGenerator
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from src import db1,db2,db3
from src.db1 import create_tabledb1, add_data, view_all_data,conn1,view_unique_tasks,delete_data
from src.db2 import create_tabledb2, add_data,view_unique_tasks,view_all_data,delete_data
from src.db3 import create_table, add_data,view_all_data,view_unique_tasks,delete_data
import pandas as pd



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
    # results = detector.detect_faces(img)
    results = imgpath(img_path,detector)
    
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    # image.tile =[e for e in image.tile if e[1][2]<2181 and e[1][3]<1294]
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    
    return result

similarity = []

# recommend image
def recommend(feature_list,features):
    # similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# # streamlit
# #st.title('Which Bollywood Celebrity You look like?')
# st.title('To whom does your face match?')

# uploaded_image = st.file_uploader('Choose an image')

# if uploaded_image is not None:
#     # save the image in a directory
#     if save_uploaded_image(uploaded_image):
#         # load the image
#         display_image = Image.open(uploaded_image)

#         # extract the features
#        
#         # recommend
#         index_pos = recommend(feature_list,features)
#         predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
#         # display
#         # for ij in filenames:
#         #     st.write(" ".join(filenames[ij].split('\\')[1].split('_')))
#         # st.write("SEE THIS FIRST")
#         # st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0])
#         st.write("Accessing Highest Similarity Cosine")
#         st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])
#         # st.write("SEE THIS themn")
#         # st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1][0])
        
#         # st.write("SIMILARITY")
#         # st.write(similarity)
#         # st.write("SORTED LIST")
#         # st.write(sorted(list(enumerate(similarity))))
#         # st.write("ENUMERATE")
#         # st.write(enumerate(similarity))
#         # st.write("INDEX_POST")
#         # st.write(index_pos)
#         # st.write("Features")
#         # st.write(features)
#         # st.write(predicted_actor)
#         # st.write("FEATURE LIST")
#         # st.write(feature_list)
#         similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
#         threshold_similarity = 60
        
#         if similarity_big_index >= threshold_similarity:
        
#             col1,col2 = st.columns(2)

#             with col1:
#                 st.header('Your uploaded image')
#                 st.image(display_image)
#             with col2:
#                 st.header("Seems like " + predicted_actor)
#                 st.image(filenames[index_pos],width=300)
                
#         else:
#             st.write("UNKNOWN PERSON")
            
# ================================================================================

def func_db1(predicted_actor):
    st.subheader("FROM SORAS")
    db1.conn1
    db1.create_tabledb1()
    cc = db1.conn1.cursor()
    # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
    cc.execute("SELECT* FROM tasksTabledb1")
    m = cc.fetchall() #tuple
        
    col1, col2, col3 = st.columns(3)

    for k in m: # looping through rows
        if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
            with col1:
                id1 = k[0]
                name1 = k[1]
                address1 = k[2]
                st.write(id1)
                st.write(name1)
                st.write(address1)
                nm=name1
            with col2:
                age1 = k[3]
                identity1 = k[4]
                date1 = k[5]
                st.write(age1)
                st.write(identity1)
                st.write(date1)
            with col3:
                
                photo1 = k[6]
                file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, use_column_width = True)
                
    # else:
    #     st.title("Unknown from SORAS")
    
    
    # #Try this way too
    # cc.execute("SELECT (id,name,address,age,identity_numb,task_due_date,photo) FROM tasksTabledb1 WHERE id =6")  
    # m1=cc.fetchone() 
    # for k in m1: # looping through rows
    #         # if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
    #     with col1:
    #         id1 = k[0]
    #         name1 = k[1]
    #         address1 = k[2]
    #         st.write(id1)
    #         st.write(name1)
    #         st.write(address1)
    #         nm=name1
    #     with col2:
    #         age1 = k[3]
    #         identity1 = k[4]
    #         date1 = k[5]
    #         st.write(age1)
    #         st.write(identity1)
    #         st.write(date1)
    #     with col3:
            
    #         photo1 = k[6]
    #         file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
    #         img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         st.image(img, use_column_width = True)
    


def func_db2(predicted_actor):
    st.subheader("FROM PRIME")
    db2.conn2
    db2.create_tabledb2()
    cb = db2.conn2.cursor()
    cb.execute("SELECT* FROM tasksTabledb2")
    m = cb.fetchall() #tuple
        
    col1, col2, col3 = st.columns(3)

    for k in m: # looping through rows
        if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
            with col1:
                id1 = k[0]
                name1 = k[1]
                address1 = k[2]
                st.write(id1)
                st.write(name1)
                st.write(address1)
            with col2:
                age1 = k[3]
                identity1 = k[4]
                date1 = k[5]
                st.write(age1)
                st.write(identity1)
                st.write(date1)
            with col3:
                
                photo1 = k[6]
                file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, use_column_width = True)
                
    # else:
    #     st.title("Unknown from PRIME")
        
        
        
    # #Try this way too
    # cb.execute("SELECT name FROM tasksTabledb2 WHERE id =6") 
    # m1=cb.fetchone()
    # for k in m1: # looping through rows
    #     # if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
    #     with col1:
    #         id1 = k[0]
    #         name1 = k[1]
    #         address1 = k[2]
    #         st.write(id1)
    #         st.write(name1)
    #         st.write(address1)
    #         nm=name1
    #     with col2:
    #         age1 = k[3]
    #         identity1 = k[4]
    #         date1 = k[5]
    #         st.write(age1)
    #         st.write(identity1)
    #         st.write(date1)
    #     with col3:
            
    #         photo1 = k[6]
    #         file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
    #         img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         st.image(img, use_column_width = True)                
    

def func_db3(predicted_actor):
    st.subheader("FROM RSSB")
    db3.conn3
    db3.create_table()
    ca = db3.conn3.cursor()
    # c.execute("SELECT* FROM tasksTable WHERE name ='{0}'".format(s))
    ca.execute("SELECT* FROM tasksTable")
    m = ca.fetchall() #tuple
    col1, col2, col3 = st.columns(3)

    for k in m: # looping through rows
        if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
            with col1:
                id1 = k[0]
                name1 = k[1]
                address1 = k[2]
                st.write(id1)
                st.write(name1)
                st.write(address1)
                namedb3=name1
            with col2:
                age1 = k[3]
                identity1 = k[4]
                date1 = k[5]
                st.write(age1)
                st.write(identity1)
                st.write(date1)
            with col3:
                
                photo1 = k[6]
                file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
                img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, use_column_width = True)
    # else:
    #     st.title("Unknown from RSSB")
    
    
            
    # #Try this way too
    # ca.execute("SELECT name FROM tasksTable WHERE id =6")
    # m1=ca.fetchone()   
    # for k in m1: # looping through rows
    #     # if k[1]==predicted_actor:# for each row if fist column == id(predicted person)
    #     with col1:
    #         id1 = k[0]
    #         name1 = k[1]
    #         address1 = k[2]
    #         st.write(id1)
    #         st.write(name1)
    #         st.write(address1)
            
    #     with col2:
    #         age1 = k[3]
    #         identity1 = k[4]
    #         date1 = k[5]
    #         st.write(age1)
    #         st.write(identity1)
    #         st.write(date1)
    #     with col3:
            
    #         photo1 = k[6]
    #         file_byte = np.asarray(bytearray(photo1),dtype =np.uint8)
    #         img =cv2.imdecode(file_byte, cv2.IMREAD_COLOR)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         st.image(img, use_column_width = True)
        
                
                
def generate_dataset1(name,id):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Changing the image to grayscale
        faces = face_classifier.detectMultiScale(gray, 1.3, 5) # Calling the above face classifier and

        if faces is (): # if face is empty return none
            return None
        # loop through faces if it is detected. 
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w] # to cop img from y+h position to x+w
        return cropped_face
    # Opening my camera

    cap = cv2.VideoCapture(0) # Value 0 means I am going to use camera from laptop, 1 for external camera
    img_id = 0 # number of image for each authorized person (to be increased)
    
    while True:
        ret, frame = cap.read() # read the image from cap (camera)
        if face_cropped(frame) is not None: # Here frame is an argument img(the same)
            # passing the image to be detected and once it is not None
            img_id+=1
            # Resizing the image(frame)
            face = cv2.resize(face_cropped(frame), (300,300))
            # converting it to grayscale 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Locating the path to be stored later
            file_name_path = f"data/{name}/{name}."+str(id)+"."+str(img_id)+".jpg"
            # Saving the image in a folder
            cv2.imwrite(file_name_path, face)
            # Put(write some text) the text in my cropped image
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Cropped face", face)

        if cv2.waitKey(1)==13 or int(img_id)==100: #13 is the ASCII character of Enter key(break once enter key is pressed)
            # break it when enter key is pressed or when img_id (number of images is = 200)=200
            break
#       release my camera and destroy all windows       
    cap.release()
    cv2.destroyAllWindows()
    st.title(" DATASET")
    st.success("INSURANCE:, Generating Dataset Completed !!!")

#==============================================================================================

def face_cropped(img):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Changing the image to grayscale
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) # Calling the above face classifier and

    if faces is (): # if face is empty return none
        return None
    # loop through faces if it is detected. 
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w] # to cop img from y+h position to x+w
    return cropped_face


def main():
    st.title('Insurance Customer Service in Rwanda:Face Recognition')
    menu1 = ["DETECTION","DATABASE&DATASET"]
    choice1 = st.sidebar.selectbox("Please Choose a Task: ",menu1)
    # st.title("Improvement of Customer Service of Insurance in Rwanda Using Face Recognition")
    menu2 = ["Create","Read","Delete"] 
    menu3 = ["RSSB","SORAS","PRIME"]
    
    if choice1=="DATABASE&DATASET":
        choice3 = st.sidebar.selectbox("Please Choose a db company: ",menu3)
        if choice3=="RSSB":
            db3.create_table()
        
            st.title("INSURANCE: RSSB")
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
                    db3.create_table()
                    db3.conn3
                    
                    db3.ca.execute("SELECT* FROM tasksTable")
                    m = db3.ca.fetchall() #tuple
                    for k in m:
                        st.write(k[0])
                        st.write(k[1])
                        st.write(k[2])
                        pass
                
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
                    db1.create_tabledb1()
                    db1.conn1
                    
                    db1.cc.execute("SELECT* FROM tasksTabledb1")
                    m = db1.cc.fetchall() #tuple
                    for k in m:
                        st.write(k[0])
                        st.write(k[1])
                        st.write(k[2])
                        pass
                
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
                    db2.create_tabledb2()
                    db2.conn2
                    
                    db2.cb.execute("SELECT* FROM tasksTabledb2")
                    m = db2.cb.fetchall() #tuple
                    for k in m:
                        st.write(k[0])
                        st.write(k[1])
                        st.write(k[2])
                        pass
                
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
            st.write('To whom does your face match?')

            uploaded_image = st.file_uploader('Choose an image')

            if uploaded_image is not None:
                # save the image in a directory
                if save_uploaded_image(uploaded_image):
                    # load the image
                    display_image = Image.open(uploaded_image)
                    detector = MTCNN()
                    # extract the features
                    features = extract_features(os.path.join(uploadn_path,uploaded_image.name),model,detector)
                    # recommend
                    index_pos = recommend(feature_list,features)
                    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                    # display
                    st.write("Accessing Highest Similarity Cosine")
                    st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

                    similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
                    threshold_similarity = 60
                    
                    if similarity_big_index >= threshold_similarity:
                    
                        col1,col2 = st.columns(2)

                        with col1:
                            st.header('Your uploaded image')
                            st.image(display_image)
                        with col2:
                            st.header("Seems like " + predicted_actor)
                            st.image(filenames[index_pos],width=300)
                            
                        st.write(predicted_actor)
                        # st.write(predicted_actor)
                        
                        func_db1(predicted_actor)
                        func_db2(predicted_actor)
                        func_db3(predicted_actor)  
                        
                        # predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_')) 
                        # predicted_id =int(os.path.split(filenames[index_pos])[1].split(".")[1]) 
                        # st.title("IDS HERE")
                        # st.write(predicted_id)
                        # st.title("Names")
                        # st.title("IDS HERE: File Names")
                        # # st.write(filenames) 
                        # fisrt = " ".join(filenames[index_pos].split('\\')[1])
                        # sec = " ".join(filenames[index_pos].split('\\'))      
                        # st.write(fisrt)
                        # st.write(sec)
                        # st.title("laster")
                        # for laster in (filenames):
                        #     st.write(laster.split('\\')[1])
                            
                    else:
                        st.write("UNKNOWN PERSON") 
                        col3,col4 = st.columns(2)

                        with col3:
                            st.header('Your uploaded image')
                            st.image(display_image)
                        with col4:
                            st.title("UNKNOWN")
                            # st.header("Seems like " + predicted_actor)
                            # st.image(filenames[index_pos],width=300)               
                    # st.write("SEE THIS FIRST")
                    # # st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0])
                    # st.write("Accessing Highest Similarity Cosine")
                    # st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])
                    # st.write("SEE THIS themn")
                    # st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1][0])
                    
                    # st.write("SIMILARITY")
                    # st.write(similarity)
                    # st.write("SORTED LIST")
                    # st.write(sorted(list(enumerate(similarity))))
                    # st.write("ENUMERATE")
                    # st.write(enumerate(similarity))
                    # st.write("INDEX_POST")
                    # st.write(index_pos)
                    # st.write("Features")
                    # st.write(features)
                    # st.write(predicted_actor)
                    # st.write("FEATURE LIST")
                    # st.write(feature_list)    
                                    
            else:
                st.subheader("Please Upload an image")
                     
                        
            
        elif prediction_mode == 'Web camera':
            st.write('To whom does your face match?')
            # if st.button("Detect Face"):
                
            #     cap = cv2.VideoCapture(0)
            #     img_id = 0 
            #     while True:
            #         ret, frame = cap.read() # read the image from cap (camera)
            #         if face_cropped(frame) is not None: # Here frame is an argument img(the same)
                        
            #             # passing the image to be detected and once it is not None
            #             img_id +=1    
            #             # Resizing the image(frame)
            #             face = cv2.resize(face_cropped(frame), (64,64))
            #             # converting it to grayscale 
            #             #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #             # Locating the path to be stored later
            #             file_name_path = "artifacts/uploads/cropped/user."+str(img_id)+".jpg"
            #             # Saving the image in a folder
            #             cv2.imwrite(file_name_path, face)
            #             display_image = Image.open(file_name_path)
            #             # display_image.tile =[e for e in display_image.tile if e[1][2]<2181 and e[1][3]<1294]
            #             # Put(write some text) the text in my cropped image
            #             cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #             cv2.imshow("Cropped face", face)
            #             detector = MTCNN()
            #             # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
            #             results = imgpath(file_name_path,detector)
            #             # for actor in actors:
            #                 # results = imgpath(actor,detector)
            #             if results:
            #                 features = extract_features(file_name_path,model,detector)
            #                 # st.write(features)
            #                 # recommend
            #                 index_pos = recommend(feature_list,features)
            #                 predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            #                 # display
            #                 st.write("look at Here")
            #                 st.write(index_pos)
            #                 st.write(predicted_actor)
            #                 st.write("Accessing Highest Similarity Cosine")
            #                 st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

            #                 similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
            #                 threshold_similarity = 60
                            
            #                 if similarity_big_index >= threshold_similarity:
                            
            #                     col1,col2 = st.columns(2)

            #                     with col1:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col2:
            #                         st.header("Seems like " + predicted_actor)
            #                         st.image(filenames[index_pos],width=300)
                                    
            #                     st.write(predicted_actor)
            #                     # st.write(predicted_actor)
                                
            #                     func_db1(predicted_actor)
            #                     func_db2(predicted_actor)
            #                     func_db3(predicted_actor)  
                            
            #                 else:
            #                     st.write("UNKNOWN PERSON") 
            #                     col3,col4 = st.columns(2)

            #                     with col3:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col4:
            #                         st.title("UNKNOWN")                 
                    
            #         if cv2.waitKey(1)==13 or int(img_id)==5: #13 is the ASCII character of Enter key(break once enter key is pressed)
            #             # break it when enter key is pressed or when img_id (number of images is = 200)=200
            #             break
            #     # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
            #     # for actor in actors: 
            #     #     os.remove(actor)                 
            # #       release my camera and destroy all windows       
            #     cap.release()
            #     cv2.destroyAllWindows()
            
            if st.button("Detect Face"):
                
                cap = cv2.VideoCapture(0)
                img_id = 0 
                while True:
                    ret, frame = cap.read() # read the image from cap (camera)
                    if face_cropped(frame) is not None: # Here frame is an argument img(the same)
                        
                        # passing the image to be detected and once it is not None
                        img_id +=1    
                        # Resizing the image(frame)
                        face = cv2.resize(face_cropped(frame), (64,64))
                        # converting it to grayscale 
                        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        # Locating the path to be stored later
                        file_name_path = "artifacts/uploads/cropped/user."+str(img_id)+".jpg"
                        # Saving the image in a folder
                        cv2.imwrite(file_name_path, face)
                        display_image = Image.open(file_name_path)
                        display_image.tile =[e for e in display_image.tile if e[1][2]<2181 and e[1][3]<1294]
                        # Put(write some text) the text in my cropped image
                        cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                        cv2.imshow("Cropped face", face)
                        detector = MTCNN()
                        actors = os.listdir("artifacts/uploads/cropped/")
                        # results = imgpath(file_name_path,detector)
                        for actor in actors:
                            
                            results = imgpath(os.path.join("artifacts/uploads/cropped/",actor),detector)
                            if results:
                                features = extract_features(os.path.join("artifacts/uploads/cropped/",actor),model,detector)
                                # st.write(features)
                                # recommend
                                index_pos = recommend(feature_list,features)
                                predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                                # display
                                st.write("look at Here")
                                st.write(index_pos)
                                st.write(predicted_actor)
                                st.write("Accessing Highest Similarity Cosine")
                                st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

                                similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
                                threshold_similarity = 60
                                
                                if similarity_big_index >= threshold_similarity:
                                
                                    col1,col2 = st.columns(2)

                                    with col1:
                                        st.header('Your uploaded image')
                                        st.image(display_image)
                                    with col2:
                                        st.header("Seems like " + predicted_actor)
                                        st.image(filenames[index_pos],width=300)
                                        
                                    st.write(predicted_actor)
                                    # st.write(predicted_actor)
                                    
                                    func_db1(predicted_actor)
                                    func_db2(predicted_actor)
                                    func_db3(predicted_actor)  
                                
                                else:
                                    st.write("UNKNOWN PERSON") 
                                    col3,col4 = st.columns(2)

                                    with col3:
                                        st.header('Your uploaded image')
                                        st.image(display_image)
                                    with col4:
                                        st.title("UNKNOWN")                 
                        
                    if cv2.waitKey(1)==13 or int(img_id)==5: #13 is the ASCII character of Enter key(break once enter key is pressed)
                        # break it when enter key is pressed or when img_id (number of images is = 200)=200
                        break
                # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
                # for actor in actors: 
                #     os.remove(actor)                 
            #       release my camera and destroy all windows       
                cap.release()
                cv2.destroyAllWindows()            

if __name__== '__main__':
    main()

