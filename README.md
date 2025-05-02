# PHOTO VALIDATION SYSTEM

# Overview
This is a system to validate an uploaded image against some conditions to meet.

# Features
- Validates image dimensions (413x531 pixels)
- Detects face
- Evaluates Background color (white)
- Neutral expression
- Absence of glasses
- Simple web interface
- Dockerized setup

## Setup

1. CLONE the repository to your local system

2. TO RUN :

    (*) DOCKER
        Start docker
        Open your terminal and run following code one by one :

          docker build -t photo-validator .
          docker run -p 5000:5000 photo-validator


      Then open http://localhost:5000 in your browser and test.


    (*) WITHOUT DOCKER
        Open terminal and run below code:
        
          python app/main.py
      
      Then open http://localhost:5000 in your browser and test.



# Limitations
- Not production-ready; false positives possible
- Limited to frontal face detection
- May fail on complex lighting/background conditions

# Future Improvements
- Use advanced ML models for fine-grained expression detection
- Add multilingual support