# Render.com Deployment Guide

## Quick Deployment Steps

### 1. Create Render Account
- Sign up at [render.com](https://render.com)
- Connect your GitHub repository

### 2. Create PostgreSQL Database
1. In Render dashboard, click "New +" → "PostgreSQL"
2. Choose a name (e.g., "youtube-transcript-db")
3. Select region closest to your users
4. Choose free tier or paid based on needs
5. Copy the "Internal Database URL" after creation

### 3. Deploy Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure settings:
   - **Name**: youtube-transcript-rag
   - **Region**: Same as database
   - **Branch**: main
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r render-requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT main:app`

### 4. Environment Variables
Add these in the Environment section:
```
DATABASE_URL=<your-postgresql-internal-url>
MISTRAL_API_KEY=<your-mistral-api-key>
SESSION_SECRET=<random-secret-string>
```

### 5. Deploy
- Click "Create Web Service"
- Wait for deployment (5-10 minutes)
- Your app will be live at `https://your-app-name.onrender.com`

## Environment Variables Explained

- **DATABASE_URL**: PostgreSQL connection string from Render
- **MISTRAL_API_KEY**: Get from [console.mistral.ai](https://console.mistral.ai)
- **SESSION_SECRET**: Any random string for Flask sessions

## Post-Deployment
- Database tables are created automatically on first run
- Test all features: upload videos, chat, premium features
- Monitor logs in Render dashboard for any issues