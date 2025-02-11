const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // To run the python script
const app = express();
const port = 3000;
const cors = require('cors');

// Allow requests from localhost:3000
app.use(cors());

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
    const username = req.body.username || "default_user"; // Fallback to avoid undefined
    const userFolder = path.join(trainFolder, username);

    if (!fs.existsSync(userFolder)) {
      fs.mkdirSync(userFolder, { recursive: true });
    }

    cb(null, userFolder);
  },
  filename: function (req, file, cb) {
    const filename = Date.now() + "-" + file.originalname;
    cb(null, filename);
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

// Dummy Whisper 
app.post('/whisper', (req, res) => {
  res.json({
    "transcription": "१२०३४५६७८९",
    "translation": "122345667"
  }) // Your form for uploading 1 audio file for prediction
});

//Dummy MFCC
app.post('/mfcc', (req, res) => {
  res.json({
    "match": true,
  }) // Your form for uploading 1 audio file for prediction
});


//Dummy Registration
app.post('/register',(req,res)=>{
  res.json({
    "message":"Successful registration!",
  })
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

  console.log(`Running: ${pythonExecutable} ${pythonScript} ${audioFilePath}`);

  const pythonProcess = spawn(pythonExecutable, [pythonScript, audioFilePath]);

  let outputData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    outputData += data.toString();
    console.log(`stdout: ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
    console.error(`stderr: ${data.toString()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);

    if (code === 0) {
      try {
        const parsedData = JSON.parse(outputData.trim()); // Parse JSON output
        res.json(parsedData); // Send JSON response
      } catch (error) {
        console.error('JSON Parse Error:', error.message);
        res.status(500).json({
          message: 'Error parsing Python script output.',
          error: error.message,
          outputData
        });
      }
    } else {
      res.status(500).json({
        message: 'Python script execution failed.',
        errorCode: code,
        errorOutput: errorData
      });
    }
  });
}

// Start the server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});