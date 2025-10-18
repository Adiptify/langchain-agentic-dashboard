#!/bin/bash
# Git Upload Script for LangChain Agentic Dashboard
# This script helps you upload your project to GitHub

echo "🚀 LangChain Agentic Dashboard - Git Upload Helper"
echo "=================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    # Commit changes
    echo "💾 Committing changes..."
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

    echo "✅ Changes committed successfully!"
fi

# Check if remote origin exists
if git remote get-url origin &> /dev/null; then
    echo "🔗 Remote origin already exists: $(git remote get-url origin)"
else
    echo "🔗 Please add your GitHub repository URL:"
    echo "   Example: https://github.com/yourusername/langchain-agentic-dashboard.git"
    read -p "Enter GitHub repository URL: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ Remote origin added: $repo_url"
    else
        echo "❌ No repository URL provided. Please add it manually:"
        echo "   git remote add origin https://github.com/yourusername/langchain-agentic-dashboard.git"
    fi
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
if git push -u origin main; then
    echo "🎉 Successfully uploaded to GitHub!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Visit your GitHub repository"
    echo "2. Add a description and topics"
    echo "3. Enable GitHub Pages if desired"
    echo "4. Set up branch protection rules"
    echo "5. Configure GitHub Actions secrets if needed"
else
    echo "❌ Failed to push to GitHub. Please check:"
    echo "1. Repository URL is correct"
    echo "2. You have push permissions"
    echo "3. GitHub authentication is set up"
    echo ""
    echo "💡 Manual commands:"
    echo "   git push -u origin main"
fi

echo ""
echo "🔧 Useful Git commands for future updates:"
echo "   git add .                    # Add all changes"
echo "   git commit -m 'message'     # Commit changes"
echo "   git push                     # Push to GitHub"
echo "   git pull                     # Pull latest changes"
echo "   git status                   # Check status"
echo "   git log --oneline            # View commit history"
