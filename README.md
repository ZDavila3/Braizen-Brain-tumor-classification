
# OwlHacks 2024 - Braizen

`Machine Learning (CNN - VGG16 / Streamlit)`

Welsome to our OwlHacks 2024 project called Braizen where we dive into the world of MRI Scans and indetificaiton.

![Static Badge](https://img.shields.io/badge/Temple_University-2024-red)


## Inspiration
Our team was deeply inspired to create an innovative solution to a modern medical problem, particularly by the personal experience of one of our members whose grandfather tragically passed away due to rapidly spreading tumors. The lack of early detection and accessible information made us realize the urgent need for tools that could make a difference. This project is not just about technology, but about giving people a fighting chance—had something like this existed, it might have helped extend his life and provided crucial insight into his condition sooner
## What it does
This web app serves as an MRI image classifier and chatbot, designed to assist with the early detection of brain tumors. By uploading an MRI scan, the app uses a trained machine learning model to analyze the image and classify the type of tumor, if present. Additionally, it includes a chatbot feature that provides users with detailed information about the detected tumor type, helping them better understand their condition. This combination of technology and AI-driven insights offers a user-friendly tool aimed at improving early diagnosis and awareness of brain tumors.
## How we built it
We began our project by sourcing a high-quality brain MRI dataset from Kaggle, which included images categorized into four classes: glioma, meningioma, pituitary tumors, and no tumor. Using this dataset, we used a pre-trained convolutional neural network (CNN) model (VGG 16) to classify the MRI images. Training the model was an intensive process, taking approximately two hours to complete and fine-tune the network for accurate predictions. Once the model was trained, we integrated it into a user-friendly web interface using Streamlit, allowing users to upload MRI images for real-time classification. We also added a chatbot feature to provide detailed information about each tumor type, creating a comprehensive tool aimed at improving understanding and early detection of brain tumors.
## Challenges we ran into
Throughout the development of our web app, we faced several challenges that required creative problem-solving. One of the major hurdles was organizing the MRI images into different categories or "buckets." MRI scans are often taken from different angles, and failing to account for this variation could negatively impact our model's performance. By categorizing the images based on angles, we were able to increase the accuracy of our predictions, ensuring more reliable results.
Another challenge we encountered was related to the chatbot feature, which provides users with additional information about tumor types. Unfortunately, the chatbot service is not free, and integrating it posed some financial constraints for our team. Balancing the need for advanced functionality while staying within budget was a key consideration, and we had to make difficult decisions on how to allocate resources effectively.

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

Insert gif or link to demo
