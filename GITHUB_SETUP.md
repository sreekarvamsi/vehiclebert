# GitHub Setup Guide for VehicleBERT

This guide will help you set up VehicleBERT on GitHub.

## Prerequisites

- Git installed on your computer
- GitHub account created
- Terminal/Command Prompt access

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website

1. Go to https://github.com and sign in
2. Click the "+" icon in the top right → "New repository"
3. Fill in:
   - **Repository name:** `vehiclebert`
   - **Description:** "Domain-Specific NLP for Automotive Entities"
   - **Visibility:** Public (or Private if preferred)
   - **Do NOT** initialize with README (we already have one)
4. Click "Create repository"

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI first if you haven't: https://cli.github.com/
gh repo create vehiclebert --public --description "Domain-Specific NLP for Automotive Entities"
```

## Step 2: Initialize Local Git Repository

Navigate to your VehicleBERT directory and run:

```bash
cd vehiclebert

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: VehicleBERT - Domain-Specific NLP for Automotive Entities"
```

## Step 3: Connect to GitHub

Replace `yourusername` with your GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/vehiclebert.git

# Verify remote
git remote -v
```

## Step 4: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 5: Verify Upload

1. Go to `https://github.com/yourusername/vehiclebert`
2. You should see all your files and folders
3. README.md should be displayed automatically

## Step 6: Configure Repository Settings (Optional)

### Add Topics

1. Go to your repository
2. Click the gear icon next to "About"
3. Add topics: `nlp`, `bert`, `named-entity-recognition`, `automotive`, `pytorch`, `transformers`, `machine-learning`

### Enable Issues and Discussions

1. Go to Settings → Features
2. Enable Issues (for bug reports)
3. Enable Discussions (for Q&A)

### Set Up Branch Protection (Optional)

1. Settings → Branches
2. Add rule for `main` branch
3. Require pull request reviews
4. Require status checks to pass

## Step 7: Add README Badges (Optional)

Add these to the top of your README.md:

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/vehiclebert.svg)](https://github.com/yourusername/vehiclebert/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/vehiclebert.svg)](https://github.com/yourusername/vehiclebert/network)
```

## Step 8: Update Links in Files

Replace `yourusername` with your actual GitHub username in:

- `README.md`
- `setup.py`
- `CONTRIBUTING.md`
- `QUICKSTART.md`

You can do this with a find-and-replace:

```bash
# On Linux/Mac
find . -type f -name "*.md" -o -name "*.py" | xargs sed -i 's/yourusername/ACTUAL_USERNAME/g'

# On Windows (using PowerShell)
Get-ChildItem -Recurse -Include *.md,*.py | ForEach-Object { 
    (Get-Content $_) -replace 'yourusername', 'ACTUAL_USERNAME' | Set-Content $_ 
}
```

Then commit and push:

```bash
git add .
git commit -m "Update GitHub username in documentation"
git push
```

## Common Git Commands

### Daily workflow:

```bash
# Check status
git status

# Add new/modified files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push

# Pull latest changes
git pull
```

### Creating branches:

```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Push branch to GitHub
git push -u origin feature/new-feature

# Switch back to main
git checkout main
```

## Troubleshooting

### Error: "Permission denied (publickey)"

Set up SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add the key to GitHub: Settings → SSH and GPG keys
```

Or use HTTPS instead:
```bash
git remote set-url origin https://github.com/yourusername/vehiclebert.git
```

### Error: "Repository not found"

Make sure:
1. Repository exists on GitHub
2. Username is correct in the URL
3. You have access to the repository

### Large file errors

Git has a 100MB file size limit. If you have large model files:
1. Add them to `.gitignore`
2. Use Git LFS (Large File Storage) for model files:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git lfs track "*.bin"
   ```

## Next Steps

1. ✓ Repository is live on GitHub
2. Add a nice profile README
3. Star your own repository
4. Share with others
5. Accept contributions

## Making Your First Release

When ready to release v1.0:

```bash
# Tag the release
git tag -a v1.0.0 -m "VehicleBERT v1.0.0 - Initial Release"

# Push tag to GitHub
git push origin v1.0.0
```

Then create a release on GitHub:
1. Go to Releases
2. Click "Create a new release"
3. Select your tag
4. Add release notes
5. Publish release

## Cloning Your Repository

Others can now clone your repository:

```bash
git clone https://github.com/yourusername/vehiclebert.git
cd vehiclebert
pip install -r requirements.txt
```

---

Congratulations! Your VehicleBERT project is now on GitHub! 🎉
