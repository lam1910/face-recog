# Face Recognition Project

1. Private source (aka images not crawl from internet and/or not free use):
	- [Image Folder](https://drive.google.com/drive/folders/1Fi7JgWgvXXalUf8QW8aRLITR7TEmpTDj?usp=sharing)
	- merge the content of the inside folder with the folder with similar name in the git repo inside the dataset folder
	
2. ~~Note: face\_recognition\_1 is the git repo of [face recognition](https://github.com/ageitgey/face_recognition) library. Use only to look for further details of the code.~~. Removed from tracking.

3. The attendence\_face is a full demo of this project. To run this demo, the simplest way is:

    ```
    <optional>
    -------------------------------------------------
    cd <path/to/the/repositories/
    -------------------------------------------------
    cd attendence_face/
    pip install -r requirements.txt
    python main.py
    ```
   
4. There is an alternative method of using pytorch and torchvision on the script retinaface_data_processing. The script utilised Adaboost instead of random forest + a simple activation function.
Note that this activation function does not fully eliminate mislabel inside people case, it is designed to limit the case of mislabel outside person as one that is in the system as much as possible