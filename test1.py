import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image

def sort_array_func(val):
    return val[3]

st.set_page_config(page_title="Disease_prediction")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Tomato_Plant_Disease_prediction")

choice = st.selectbox("select",["Upload image","Real Time"])
conf = st.number_input("conf",0.2)

if choice == "Upload image":
    
    image_data = st.file_uploader("Upload the Image")
    img_summit_button = st.button("Predict",use_container_width=True)
    
    if img_summit_button:
        
        model = YOLO('TOMATO_yolov8n_best .pt')   
    
        image = Image.open(image_data)
        image.save("input_data_image.png")
        frame = cv2.imread("input_data_image.png")
        frame_without_condition = frame
                
        results = model.predict(source=frame,iou=0.7,conf= conf)
        plot_show =  results[0].plot()
        get_array = results[0].boxes.numpy().data.tolist()

        # function to sort array 
        get_array.sort(key=sort_array_func)

        cv2.putText(plot_show,"" + str(len(get_array)),(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 
        st.image(plot_show)
        #------------------------------------------ model predicted result all boxes ------------------------------------------#
        for ind,i in enumerate(get_array):
            cv2.rectangle(frame_without_condition,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(255,0,0),2)
            cv2.putText(frame_without_condition,str(ind+1),(int(i[0])-70,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2,cv2.LINE_AA) 
            cv2.line(frame_without_condition,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)                
                    
        count = str(len(get_array))
        frame_without_condition = cv2.resize(frame_without_condition,(200,750))
        cv2.putText(frame_without_condition,""+ str(len(get_array)),(1,680),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) 

        
        classes_model = model.names
          
        detected_list = [ classes_model[i] for  i in results[0].boxes.cls.tolist()]

        if 'Early Blight' in detected_list:
                    st.success("""Treatments for Early_blight:\n
1.Seed treatment with Trichodrema asperellum @4g/kg of seed or Bacillus subtilis @10g/kg of seed.\n 
2.Foliar spray of Panchagavya @3% twice at 15th and 45th days. \n 
3.Foliar spray of Neem oil@3% twice at 30th and 60th days \n 
4.Foliar spray of Bacilis subtilis @5g/l at 70th and prehaverst spray""")
        if 'Late Blight' in detected_list:
            st.success("""Treatments for Late_blight:\n
1.Growing green manure like daincha/sunhemp and insitu ploughing\n
2.Soil application of farm yard mature 50kg mixed with trichoderma asperellu @2.5 kg/ha or Bacillus subtilis @2.5kg/ha.\n
3.Application of vermicompost 7.5t/ha\n
4. Soil drenching with Jeevamruth @200/ha\n
5.Crop ratation\n
6.Summer ploughing and soil solarization""")
        if 'Yellow Leaf Curl Virus'  in detected_list:
            st.success(""" Treatments for Yellow Leaf Curl Virus:\n
1.Removal of infected plants and destroying\n
2.Control of whitefly vectors by the installating of yellow sticky traps @12nos/ha\n
3. Removal of weed host abutilon indicum\n
4. Foiler spary of Beauveria bassiam/five leaf extract 10% /3G extract Neem oil @3% twice at 30th and 60th days after planning """)
        if 'Mosaic Virus'  in detected_list:
            st.success("""Treatments for Mosaic_Virus:\n
Removal of PbNv infected plants and selection of healthy seedlings from nursey up to 25 days.\n
2. Raaising border(Barrier crop)as sorghum or maize\n
3. Foliar spraying of Beauveria bassiana/Five leaf extract 10%/3G ext6ract/Neem oil@35 twice at 30th and 60th days after planting for the control of thrips""")
        

if choice == "Real Time":
    camera_input_data = st.camera_input("Take a pic")
    if camera_input_data is not None:
        model = YOLO('TOMATO_yolov8n_best .pt') 

        image = Image.open(camera_input_data)
        image.save("input_data_image.png")
        frame = cv2.imread("input_data_image.png")
        frame_without_condition = frame
    
        results = model.predict(source=frame,iou=0.7,conf= conf)        
              
        get_array = results[0].boxes.numpy().data.tolist()

        # function to sort array 
        get_array.sort(key=sort_array_func)

        st.image(results[0].plot())
        classes_model = model.names
        
        detected_list = [classes_model[i] for i in results[0].boxes.cls.tolist()]
        
       
        if 'Early Blight' in detected_list:
                    st.success("""Treatments for Early_blight:\n
1.Seed treatment with Trichodrema asperellum @4g/kg of seed or Bacillus subtilis @10g/kg of seed.\n 
2.Foliar spray of Panchagavya @3% twice at 15th and 45th days. \n 
3.Foliar spray of Neem oil@3% twice at 30th and 60th days \n 
4.Foliar spray of Bacilis subtilis @5g/l at 70th and prehaverst spray""")
        if 'Late Blight' in detected_list:
            st.success("""Treatments for Late_blight:\n
1.Growing green manure like daincha/sunhemp and insitu ploughing\n
2.Soil application of farm yard mature 50kg mixed with trichoderma asperellu @2.5 kg/ha or Bacillus subtilis @2.5kg/ha.\n
3.Application of vermicompost 7.5t/ha\n
4. Soil drenching with Jeevamruth @200/ha\n
5.Crop ratation\n
6.Summer ploughing and soil solarization""")
        if 'Yellow Leaf Curl Virus'  in detected_list:
            st.success(""" Treatments for Yellow Leaf Curl Virus:\n
1.Removal of infected plants and destroying\n
2.Control of whitefly vectors by the installating of yellow sticky traps @12nos/ha\n
3. Removal of weed host abutilon indicum\n
4. Foiler spary of Beauveria bassiam/five leaf extract 10% /3G extract Neem oil @3% twice at 30th and 60th days after planning """)
        if 'Mosaic Virus'  in detected_list:
            st.success("""Treatments for Mosaic_Virus:\n
Removal of PbNv infected plants and selection of healthy seedlings from nursey up to 25 days.\n
2. Raaising border(Barrier crop)as sorghum or maize\n
3. Foliar spraying of Beauveria bassiana/Five leaf extract 10%/3G ext6ract/Neem oil@35 twice at 30th and 60th days after planting for the control of thrips""")
        
