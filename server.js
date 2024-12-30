const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // To run the python script
const app = express();
const port = 3000;

// Define the folders to save the audio files
const trainFolder = 'C:/mfcc and gmm/New folder/GMM/dataset/train';
const predictFolder = 'C:/mfcc and gmm/New folder/GMM/dataset/predict';

// Ensure the folders exist
if (!fs.existsSync(trainFolder)) {
  fs.mkdirSync(trainFolder, { recursive: true });
}

if (!fs.existsSync(predictFolder)) {
  fs.mkdirSync(predictFolder, { recursive: true });
}

// Set up multer storage configuration for 5 audio files (for training)
const storageTrain = multer.diskStorage({
  destination: function (req, file, cb) {
    const userFolder = path.join(trainFolder, req.body.username);

    // Create a new folder for the user if it doesn't exist
    if (!fs.existsSync(userFolder)) {
      fs.mkdirSync(userFolder, { recursive: true });
    }

    cb(null, userFolder); // Set the destination folder for training
  },
  filename: function (req, file, cb) {
    const filename = file.originalname.split('.')[0] + '-' + Date.now() + path.extname(file.originalname);
    cb(null, filename); // Save the file with a unique name
  }
});

// Create an upload instance for 5 audio files (for training)
const uploadTrain = multer({
  storage: storageTrain,
  fileFilter: (req, file, cb) => {
    const filetypes = /wav|mp3|flac/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);

    if (extname && mimetype) {
      return cb(null, true);
    } else {
      return cb(new Error('Only audio files are allowed!'), false);
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 } // Limit file size to 50MB
}).fields([
  { name: 'audio1', maxCount: 1 },
  { name: 'audio2', maxCount: 1 },
  { name: 'audio3', maxCount: 1 },
  { name: 'audio4', maxCount: 1 },
  { name: 'audio5', maxCount: 1 }
]);

// Set up multer storage configuration for single audio file (for prediction)
const storageSingleAudio = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, predictFolder); // Directly save the audio file in the predict folder
  },
  filename: function (req, file, cb) {
    const filename = file.originalname.split('.')[0] + '-' + Date.now() + path.extname(file.originalname);
    cb(null, filename); // Save the file with a unique name
  }
});


// Create an upload instance for a single audio file (for prediction)
const uploadSingleAudio = multer({
  storage: storageSingleAudio,
  fileFilter: (req, file, cb) => {
    const filetypes = /wav|mp3|flac/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);

    if (extname && mimetype) {
      return cb(null, true);
    } else {
      return cb(new Error('Only audio files are allowed!'), false);
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 } // Limit file size to 50MB
}).single('audio'); // Expecting only one audio file for prediction

// Serve the HTML form for 5 audio files (train)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html')); // Your form for uploading 5 audio files for training
});

// Serve the HTML form for 1 audio file (predict)
app.get('/predict', (req, res) => {
  res.sendFile(path.join(__dirname, 'single-upload.html')); // Your form for uploading 1 audio file for prediction
});

// Handle the file upload request for training (5 audio files)
app.post('/upload', (req, res) => {
  uploadTrain(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ message: 'File upload error', error: err.message });
    } else if (err) {
      return res.status(400).json({ message: 'File upload error', error: err.message });
    }

    // Successfully uploaded files for training
    const username = req.body.username;
    console.log(`Files uploaded for ${username}`);

    // Run the Python script for training after uploading the files
    runPythonTrainingScript(username, res);
  });
});

// Handle the file upload request for prediction (single audio file)
app.post('/upload-single', (req, res) => {
  uploadSingleAudio(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ message: 'File upload error', error: err.message });
    } else if (err) {
      return res.status(400).json({ message: 'File upload error', error: err.message });
    }

    // Successfully uploaded the audio file for prediction
    const audioFile = req.file;
    const username = req.body.username;
    console.log(`Audio uploaded for ${username}: ${audioFile.filename}`);

    // Run the Python script for prediction after uploading the audio file
    runPythonPredictionScript(audioFile.filename, res);
  });
});

// Function to run the Python script for training
function runPythonTrainingScript(username, res) {
  const pythonExecutable = 'C:/mfcc and gmm/New folder/GMM/myenv/Scripts/python.exe';
  const pythonScript = 'C:/mfcc and gmm/New folder/GMM/train.py';
  
  // Run the Python script with the user's name as an argument
  const pythonProcess = spawn(pythonExecutable, [pythonScript, username]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      res.json({
        message: `Files uploaded and Python training script executed successfully for ${username}!`
      });
    } else {
      res.status(500).json({
        message: 'There was an error running the Python training script.',
        errorCode: code
      });
    }
  });
}

// Function to run the Python script for prediction
function runPythonPredictionScript(audioFileName, res) {
  const pythonExecutable = 'C:/mfcc and gmm/New folder/GMM/myenv/Scripts/python.exe';
  const pythonScript = 'C:/mfcc and gmm/New folder/GMM/predict.py';
  const audioFilePath = audioFileName;
  
  // Run the Python script with the audio file name as an argument
  const pythonProcess = spawn(pythonExecutable, [pythonScript, audioFilePath]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      res.json({
        message: `Audio file uploaded and prediction successfully executed for ${audioFileName}!`,
      });
    } else {
      res.status(500).json({
        message: 'There was an error running the Python prediction script.',
        errorCode: code
      });
    }
  });
}

// Start the server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
