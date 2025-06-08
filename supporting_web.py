#  # Default Camera
#                 video_capture = cv2.VideoCapture(0)
#                 # detector = MTCNN()
        
#                 while True:
#                     # capture frame by frame
#                     (grabbed,frame)= video_capture.read()
#                     # ret, img_cam = video_capture.read()
#                     rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                     # output = detector.detect_faces(img_cam)
                    
#                     # Detect faces in the webcam
#                     faces = face_cascade.detectMultiScale(rgb,scaleFactor =1.3, minNeighbors =5)
                    
#                     # for each face found
                    
#                     for (x,y,w,h) in faces:
#                         roi_rgb = rgb[y:y+h,x:x+w]
                    
#                         # Draw a rectangle
#                         color = (255,0,0)#in BGR
#                         stroke=2
#                         cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
                        
#                         # Resizing the image
#                         size = (image_width,image_height)
#                         resized_image = cv2.resize(roi_rgb,size)
#                         image_array = np.array(resized_image,"uint8")
#                         img = image_array.reshape(1,image_width,image_height,3)
#                         img = img.astype('float32')
#                         img /= 255
                    
                        
                    
                    
#                     # for single_output in output:
#                     #     x,y,width,height = single_output["box"]
#                     #     cv2.rectangle(img_cam,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
                        
#                     # cv2.imshow("win",img_cam )
#                     # save the image in a directory
#                         if save_uploaded_image(img):
#                             # load the image
#                             display_image = Image.open(img)

#                             # extract the features
#                             features = extract_features(os.path.join(uploadn_path,img.name),model,detector)
#                             # recommend
#                             index_pos = recommend(feature_list,features)
#                             predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
#                             # display
#                             st.write("Accessing Highest Similarity Cosine")
#                             st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])
                            
#                             similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
#                             threshold_similarity = 60
                            
#                             if similarity_big_index >= threshold_similarity:
                            
#                                 col1,col2 = st.columns(2)

#                                 with col1:
#                                     st.header('Your uploaded image')
#                                     st.image(display_image)
#                                 with col2:
#                                     st.header("Seems like " + predicted_actor)
#                                     st.image(filenames[index_pos],width=300)
                                    
#                             else:
#                                 st.write("UNKNOWN PERSON") 
#                                 col3,col4 = st.columns(2)

#                                 with col3:
#                                     st.header('Your uploaded image')
#                                     st.image(display_image)
#                                 with col4:
#                                     st.title("UNKNOWN")
#                                     # st.header("Seems like " + predicted_actor)
#                                     # st.image(filenames[index_pos],width=300)               
                            
                            
                            
#                         key = cv2.waitKey(1) & 0xFf
#                         if cv2.waitKey(1)==ord("q"):
#                         # if cv2.waitKey(1)==13:
#                             break
#                 video_capture.release()
#                 cv2.destroyAllWindows()            
                    