@echo off
REM Git Upload Script for LangChain Agentic Dashboard (Windows)
REM This script helps you upload your project to GitHub

echo 🚀 LangChain Agentic Dashboard - Git Upload Helper
echo ==================================================

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

REM Check if we're in a git repository
if not exist ".git" (
    echo 📁 Initializing Git repository...
    git init
)

REM Add all files
echo 📝 Adding files to Git...
git add .

REM Check if there are changes to commit
git diff --staged --quiet
if errorlevel 1 (
    REM Commit changes
    echo 💾 Committing changes...
    git commit -m "Initial commit: LangChain Agentic Dashboard with advanced features

- ✅ Core system: ingestion, embedding, routing, agent tools
- ✅ SLM integration: clean row-level summarization
- ✅ User profiles: authentication and personalization  
- ✅ Advanced search: autocomplete and intelligent filtering
- ✅ Comprehensive logging: database-stored analytics
- ✅ Streamlit UI: modern web dashboard
- ✅ Docker support: containerized deployment
- ✅ CI/CD: GitHub Actions workflow
- ✅ Documentation: comprehensive guides and examples"

    echo ✅ Changes committed successfully!
) else (
    echo ℹ️  No changes to commit.
)

REM Check if remote origin exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 🔗 Please add your GitHub repository URL:
    echo    Example: https://github.com/yourusername/langchain-agentic-dashboard.git
    set /p repo_url="Enter GitHub repository URL: "
    
    if not "%repo_url%"=="" (
        git remote add origin "%repo_url%"
        echo ✅ Remote origin added: %repo_url%
    ) else (
        echo ❌ No repository URL provided. Please add it manually:
        echo    git remote add origin https://github.com/yourusername/langchain-agentic-dashboard.git
    )
) else (
    for /f "tokens=*" %%i in ('git remote get-url origin') do set origin_url=%%i
    echo 🔗 Remote origin already exists: !origin_url!
)

REM Push to GitHub
echo 📤 Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo ❌ Failed to push to GitHub. Please check:
    echo 1. Repository URL is correct
    echo 2. You have push permissions
    echo 3. GitHub authentication is set up
    echo.
    echo 💡 Manual commands:
    echo    git push -u origin main
) else (
    echo 🎉 Successfully uploaded to GitHub!
    echo.
    echo 📋 Next steps:
    echo 1. Visit your GitHub repository
    echo 2. Add a description and topics
    echo 3. Enable GitHub Pages if desired
    echo 4. Set up branch protection rules
    echo 5. Configure GitHub Actions secrets if needed
)

echo.
echo 🔧 Useful Git commands for future updates:
echo    git add .                    # Add all changes
echo    git commit -m "message"     # Commit changes
echo    git push                     # Push to GitHub
echo    git pull                     # Pull latest changes
echo    git status                   # Check status
echo    git log --oneline            # View commit history

pause
