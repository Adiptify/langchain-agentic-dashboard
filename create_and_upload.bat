@echo off
echo 🚀 GitHub Repository Creation Helper
echo ====================================
echo.
echo 📋 Please follow these steps:
echo.
echo 1. Go to https://github.com/new
echo 2. Repository name: langchain-agentic-dashboard
echo 3. Description: A comprehensive web dashboard backed by a LangChain agentic system
echo 4. Make it Public
echo 5. DO NOT initialize with README, .gitignore, or license
echo 6. Click "Create repository"
echo.
echo ⏳ After creating the repository, press any key to continue...
pause
echo.
echo 🔗 Now let's push your code to GitHub...
echo.

REM Add remote origin
git remote add origin https://github.com/Adiptify/langchain-agentic-dashboard.git

REM Rename branch to main
git branch -M main

REM Push to GitHub
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Push failed. Please check:
    echo 1. Repository was created successfully
    echo 2. You're logged into GitHub
    echo 3. Repository URL is correct
    echo.
    echo 💡 Try these manual commands:
    echo    git remote add origin https://github.com/Adiptify/langchain-agentic-dashboard.git
    echo    git branch -M main
    echo    git push -u origin main
) else (
    echo.
    echo 🎉 Success! Your repository is now live at:
    echo    https://github.com/Adiptify/langchain-agentic-dashboard
    echo.
    echo 📋 Next steps:
    echo 1. Visit your repository
    echo 2. Add a description and topics
    echo 3. Enable GitHub Pages if desired
    echo 4. Share your project!
)

pause

