@echo off
echo 🔐 GitHub Authentication Helper
echo ================================
echo.
echo 📋 You need to authenticate with GitHub to push your code.
echo.
echo 🔑 Option 1: Personal Access Token (Recommended)
echo 1. Go to: https://github.com/settings/tokens
echo 2. Click "Generate new token" ^> "Generate new token (classic)"
echo 3. Note: "LangChain Dashboard Upload"
echo 4. Expiration: 90 days
echo 5. Scopes: Check "repo" (Full control of private repositories)
echo 6. Click "Generate token"
echo 7. Copy the token
echo.
echo ⏳ After creating the token, enter it below:
set /p token="Enter your Personal Access Token: "

if not "%token%"=="" (
    echo.
    echo 🔗 Updating remote URL with token...
    git remote set-url origin https://%token%@github.com/Adiptify/langchain-agentic-dashboard.git
    
    echo.
    echo 📤 Pushing to GitHub...
    git push -u origin main
    
    if errorlevel 1 (
        echo.
        echo ❌ Push failed. Please check:
        echo 1. Token is correct and has repo permissions
        echo 2. Repository exists and you have push access
        echo 3. You're connected to the internet
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
) else (
    echo.
    echo ❌ No token provided. Please run this script again with a valid token.
)

echo.
pause

