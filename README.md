﻿# coe_engtohindi



#to run this 
#cd whisper.cpp
#ffmpeg -i "C:\Users\Mayer\OneDrive\Documents\Sound Recordings\Recording.m4a" -ar 16000 -ac 1 -c:a pcm_s16le "C:\Users\Mayer\OneDrive\Documents\Sound Recordings\Recording.wav"
 (#here add your file path )
 .\build\bin\Release\whisper-cli.exe -f 'C:\Users\Mayer\OneDrive\Documents\Sound Recordings\Recording.wav' > output.txt

 now 
 cd ..
 python app.py (if already have model downloaded otherwise first go to new.py .. change the path where want to download and then run new.py and after that run app.py)
