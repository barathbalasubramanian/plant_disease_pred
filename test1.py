import streamlit as st
import cv2
from ultralytics import YOLO 
import numpy as np
from PIL import Image
import tempfile
import os


def load_model():
    return YOLO('TOMATO_yolov8n_best .pt')


def detect_diseases(image, model, conf):
    results = model.predict(source=image, iou=0.7, conf=conf)
    return results

def display_diseases(disease_list):
    treatments = {
        'Early Blight': "Treatments for Early Blight in Organic Way:\n1. Seed treatment with Trichodrema asperellum @4g/kg of seed or Bacillus subtilis @10g/kg of seed.\n2. Foliar spray of Panchagavya @3% twice at 15th and 45th days. \n3. Foliar spray of Neem oil@3% twice at 30th and 60th days \n4. Foliar spray of Bacilis subtilis @5g/l at 70th and prehaverst spray",
        'Late Blight': "Treatments for Late Blight:\n1. Growing green manure like daincha/sunhemp and insitu ploughing\n2. Soil application of farm yard mature 50kg mixed with trichoderma asperellu @2.5 kg/ha or Bacillus subtilis @2.5kg/ha.\n3. Application of vermicompost 7.5t/ha\n4. Soil drenching with Jeevamruth @200/ha\n5. Crop ratation\n6. Summer ploughing and soil solarization",
        'Yellow Leaf Curl Virus': "Treatments for Yellow Leaf Curl Virus:\n1. Removal of infected plants and destroying\n2. Control of whitefly vectors by the installating of yellow sticky traps @12nos/ha\n3. Removal of weed host abutilon indicum\n4. Foiler spary of Beauveria bassiam/five leaf extract 10% /3G extract Neem oil @3% twice at 30th and 60th days after planning",
        'Mosaic Virus': "Treatments for Mosaic Virus:\n1. Removal of PbNv infected plants and selection of healthy seedlings from nursey up to 25 days.\n2. Raaising border(Barrier crop)as sorghum or maize\n3. Foliar spraying of Beauveria bassiana/Five leaf extract 10%/3G ext6ract/Neem oil@35 twice at 30th and 60th days after planting for the control of thrips"
    }
    for disease in disease_list:
        if disease in treatments:
            st.success(treatments[disease])

def main():
    st.set_page_config(page_title="Disease Prediction")

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                #header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.title("Tomato Plant Disease Prediction")

    choice = st.selectbox("Select", ["Upload Image", "Upload Video", "Real Time"])
    conf = st.number_input("Confidence", 0.0, 1.0, 0.2)

    model = load_model()

    if choice == "Upload Image":
        image_data = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if image_data is not None:
            img = np.array(Image.open(image_data))
            st.image(img, caption='Uploaded Image', use_column_width=True)
            if st.button("Predict"):
                results = detect_diseases(img, model, conf)
                st.image(results[0].plot())
                detected_list = [model.names[i] for i in results[0].boxes.cls.tolist()]
                display_diseases(detected_list)

    elif choice == "Upload Video":
        video_data = st.file_uploader("Upload Video", type=['mp4'])
        if video_data is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(video_data.read())
            video_cap = cv2.VideoCapture(temp_file.name)
            st.video(temp_file.name)
            if st.button("Predict"):
                output_frames = []
                while True:
                    ret, frame = video_cap.read()
                    if not ret:
                        break
                    results = detect_diseases(frame, model, conf)
                    annotated_frame = results[0].plot()
                    output_frames.append(annotated_frame)
                    detected_list = [model.names[i] for i in results[0].boxes.cls.tolist()]
                    display_diseases(detected_list)
                
                # Save frames with predictions as images
                output_folder = "C:/Users/maddy/OneDrive/Documents/Tomato model/video/"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                for i, frame in enumerate(output_frames):
                    cv2.imwrite(os.path.join(output_folder, f"frame_{i}.jpg"), frame)

                # Compile images into a video
                output_video_path = "C:/Users/maddy/OneDrive/Documents/Tomato model/video/output_video.mp4"
                frame_rate = int(video_cap.get(cv2.CAP_PROP_FPS))
                frame_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
                for i in range(len(output_frames)):
                    img = cv2.imread(os.path.join(output_folder, f"frame_{i}.jpg"))
                    out.write(img)
                out.release()
                
                st.success("Video prediction completed. Click below to download the output video.")
                st.markdown(f"[Download Output Video](/{output_video_path})")
                
                # Cleanup temporary files
                temp_file.close()

    elif choice == "Real Time":
        st.write("Real Time Detection is not implemented yet.")

if __name__ == "__main__":
    main()