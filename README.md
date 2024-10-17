
# OwlHacks 2024 - Braizen (24 Hr Hack)

![image](https://github.com/user-attachments/assets/ac6d4682-4874-4894-b607-fb599c56e801)

`Machine Learning (CNN - VGG16 / Streamlit)`

Welcome to our OwlHacks 2024 project called Braizen where we dive into the world of MRI Scans, Machine Learning, and Ai.

![Static Badge](https://img.shields.io/badge/Temple_University-2024-red)

![Static Badge](https://img.shields.io/badge/Made_with-Love%3C3-pink)



## Inspiration
Our team was deeply inspired to create an innovative solution to a modern medical problem, particularly by the personal experience of one of our members whose grandfather tragically passed away due to rapidly spreading tumors. The lack of early detection and accessible information made us realize the urgent need for tools that could make a difference. This project is not just about technology, but about giving people a fighting chance—had something like this existed, it might have helped extend his life and provided crucial insight into his condition sooner
## What it does
This web app serves as an MRI image classifier and chatbot, designed to assist with the early detection of brain tumors. By uploading an MRI scan, the app uses a trained machine learning model to analyze the image and classify the type of tumor, if present. Additionally, it includes a chatbot feature that provides users with detailed information about the detected tumor type, helping them better understand the tumor present. This combination of technology and AI-driven insights offers a user-friendly tool aimed at improving early diagnosis and awareness of brain tumors.
## How we built it
We began our project by sourcing a high-quality brain MRI dataset from Kaggle, which included images categorized into four classes: glioma, meningioma, pituitary tumors, and no tumor. Using this dataset, we used a pre-trained convolutional neural network (CNN) model (Specifically: VGG 16) to classify the MRI images. Training the model was an intensive process, taking approximately two hours to complete and fine-tune the network for accurate predictions. Once the model was trained, we integrated it into a user-friendly web interface using Streamlit, allowing users to upload MRI images for real-time classification. We also added a chatbot feature to provide detailed information about each tumor type, creating a comprehensive tool aimed at improving understanding and early detection of brain tumors.
## Challenges we ran into
Throughout the development of our project, we encountered several significant challenges that tested our team's resilience and problem-solving skills. One of the major hurdles was the limitation of having only 8 GB of RAM, which resulted in prolonged training times for our model. This made the process not only time-consuming but also frustrating as we sought to achieve optimal accuracy.

Additionally, we had no prior experience in connecting our machine learning backend to a user-friendly frontend service. This lack of knowledge initially hindered our ability to create a seamless user experience, requiring extensive research and trial-and-error to bridge the gap between the two.

We also faced challenges with API integration, particularly in setting up triggers that would allow for informative interactions without overwhelming users with unnecessary data. Understanding how to create a responsive system that provided valuable insights while remaining user-friendly was a learning curve we had to navigate.
## Future plans
In the future, our goal is to enhance the accuracy of our model by incorporating a broader range of data variables, such as color adjustments and exposure levels. We also aim to classify the data into distinct categories, recognizing that MRI scans can vary significantly based on the angles at which they are taken. By identifying the position of the scans, we can more effectively determine the type of tumor present. This multifaceted approach will enable us to refine our analyses and improve diagnostic outcomes.



## Language & Tools
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original-wordmark.svg" 
     width="50" 
     height="50" /> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" 
     width="50"
     height="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" 
     width="50"
     height="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/keras/keras-original.svg" 
     width="50"
     height="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tensorflow/tensorflow-original.svg" 
     width="50"
     height="50"/>

## Demo
Video Demo of Braizen Brain Tumor classification: https://www.youtube.com/watch?v=JSGOsNMHmwA
![image](https://github.com/user-attachments/assets/fb675aea-8c1b-4513-95d8-7cdb7272a267)

## How to run
`Remember to have all the files/images in the same space to run locally`

`1. Download Data Set`

Link: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

`2. Train your model using this data set`

Link: [ brain_tumor_classifier.py](https://github.com/ZDavila3/Braizen-Brain-tumor-classification/blob/68ae1cf62ddd4e8efd75d9530a027ece5b05102d/brain_tumor_classifier.py)

**Please modify code lines 12 & 13 in the brin_tumor_classifier.py file to your unique path/where you want the h5 file stored**

`3. Time to connect with front end`

Link: [ mri_classifier.py](https://github.com/ZDavila3/Braizen-Brain-tumor-classification/blob/68ae1cf62ddd4e8efd75d9530a027ece5b05102d/mri_classifier.py)

**Please modify line 10 to the h5 file you generated when training your model**

**Enter your chat gpt API key to have a functioning chat bot**

## Contributors
- [@Zanett](https://github.com/ZDavila3)
- [@Barry](https://github.com/mikey6002)
