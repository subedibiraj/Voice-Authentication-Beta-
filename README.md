
HOW TO RUN:

Note: requirements.txt file contains dependencies for the python scripts, create virtual environment (name: myenv) to install the dependencies, the server.js code is designed to trigger the virtual environment to run the python code. Also, change the path of the directories and file from the server.js file as per your folder strucutre.

1) Start the server.js file with "node server.js" , install any dependencies if necessary(express,multer)!

2) Upload the name and the audio, a folder with the name is created inside dataset/train where all the audio files are stored. The train.py 
   file is then triggered and user's model is created with the name and stored in speakers_model.

3) To predict visit "http://localhost:3000/predict" and submit the name and the audio, the detected user's name is shown on the console.