# 🚀 Deployment Guide

This guide covers various deployment options for the LangChain Agentic Dashboard.

## 📋 Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Python 3.8+ (for direct deployment)
- Ollama installed and running
- Required Ollama models pulled

## 🐳 Docker Deployment (Recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-agentic-dashboard.git
   cd langchain-agentic-dashboard
   ```

2. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   Open your browser to `http://localhost:8501`

### Manual Docker Build

1. **Build the image**
   ```bash
   docker build -t langchain-agentic-dashboard .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/index:/app/index \
     langchain-agentic-dashboard
   ```

### Docker Environment Variables

```bash
# Optional: Use external Ollama service
docker run -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://your-ollama-server:11434 \
  langchain-agentic-dashboard
```

## ☁️ Cloud Deployment

### AWS EC2

1. **Launch EC2 instance**
   - Instance type: t3.medium or larger
   - OS: Ubuntu 20.04 LTS
   - Storage: 20GB+ EBS volume

2. **Install dependencies**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy application**
   ```bash
   git clone https://github.com/yourusername/langchain-agentic-dashboard.git
   cd langchain-agentic-dashboard
   docker-compose up -d
   ```

4. **Configure security group**
   - Allow inbound traffic on port 8501
   - Restrict access to your IP if needed

### Google Cloud Platform

1. **Create Compute Engine instance**
   ```bash
   gcloud compute instances create langchain-dashboard \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --machine-type=e2-medium \
     --boot-disk-size=20GB
   ```

2. **Deploy using startup script**
   ```bash
   gcloud compute instances add-metadata langchain-dashboard \
     --metadata-from-file startup-script=deploy.sh
   ```

### Azure

1. **Create Virtual Machine**
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name langchain-dashboard \
     --image UbuntuLTS \
     --size Standard_B2s \
     --admin-username azureuser \
     --generate-ssh-keys
   ```

2. **Deploy application**
   ```bash
   az vm run-command invoke \
     --resource-group myResourceGroup \
     --name langchain-dashboard \
     --command-id RunShellScript \
     --scripts @deploy.sh
   ```

## 🏠 Local Development

### Direct Python Installation

1. **Setup environment**
   ```bash
   python -m venv mendyenv
   source mendyenv/bin/activate  # Linux/Mac
   # or
   mendyenv\Scripts\activate    # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_streamlit.txt
   ```

3. **Run application**
   ```bash
   streamlit run streamlit_app.py
   ```

### Using Setup Script

```bash
python setup.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
SLM_PARSE_MODEL=llama3.2:3b
LLM_REASONING_MODEL=gpt-oss:120b-cloud

# Application Settings
DATA_DIR=./data
INDEX_DIR=./index
LOG_LEVEL=INFO

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Production Settings

For production deployment:

```env
# Security
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false

# Performance
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# Logging
LOG_LEVEL=WARNING
```

## 📊 Monitoring

### Health Checks

The application includes health check endpoints:

- **Streamlit**: `http://localhost:8501/_stcore/health`
- **Ollama**: `http://localhost:11434/api/tags`

### Logging

Logs are stored in:
- **Application logs**: `app.log`
- **Database logs**: `logs.db`
- **User activity**: `user_profiles.db`

### Monitoring with Prometheus

Add to `docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔒 Security

### Production Security Checklist

- [ ] Use HTTPS with SSL certificates
- [ ] Implement authentication (OAuth, LDAP)
- [ ] Restrict file upload types and sizes
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Network isolation
- [ ] Backup strategy

### SSL/TLS Setup

Using Let's Encrypt with Nginx:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔄 Updates and Maintenance

### Updating the Application

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Rebuild containers**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/langchain-dashboard"

mkdir -p $BACKUP_DIR

# Backup databases
cp logs.db $BACKUP_DIR/logs_$DATE.db
cp user_profiles.db $BACKUP_DIR/user_profiles_$DATE.db
cp search_index.db $BACKUP_DIR/search_index_$DATE.db

# Backup uploaded files
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz uploads/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Automated Backups

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama connection failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart Ollama
   ollama serve
   ```

2. **Port already in use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   
   # Kill process
   kill -9 <PID>
   ```

3. **Permission denied**
   ```bash
   # Fix upload directory permissions
   chmod 755 uploads/
   chown -R $USER:$USER uploads/
   ```

### Log Analysis

```bash
# View application logs
tail -f app.log

# Check Docker logs
docker-compose logs -f langchain-dashboard

# Database queries
sqlite3 logs.db "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10;"
```

## 📈 Scaling

### Horizontal Scaling

Use load balancer with multiple instances:

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  langchain-dashboard:
    build: .
    ports:
      - "8501-8503:8501"
    deploy:
      replicas: 3
```

### Vertical Scaling

Increase resources:
- **CPU**: More cores for parallel processing
- **RAM**: Larger models and more concurrent users
- **Storage**: SSD for faster I/O

## 📞 Support

For deployment issues:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/langchain-agentic-dashboard/issues)
- **Documentation**: Check README.md and CONTRIBUTING.md
- **Community**: Join our Discord server

---

**Happy Deploying! 🚀**
