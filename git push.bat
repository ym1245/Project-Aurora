@echo off
set /p commit_msg="Enter commit message: "

cd C:\Users\breadly1245\Desktop\Project Aurora
git remote add origin https://github.com/ym1245/Project-Aurora
git add .
git commit -m "%commit_msg%"
git fetch origin
git push origin main

pause