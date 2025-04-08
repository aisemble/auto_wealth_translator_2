# Deployment Guide for AutoWealthTranslate

This document provides instructions for deploying the AutoWealthTranslate application in various environments.

## Local Deployment

### Running with Streamlit

1. Clone the repository:
   ```bash
   git clone https://github.com/autowealthtranslate/auto-wealth-translate.git
   cd auto-wealth-translate
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

   Or use the provided script:
   ```bash
   ./run_streamlit.sh
   ```

4. Access the application in your browser at http://localhost:8501

### Environment Variables

The application uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required for GPT models)
- `STREAMLIT_SERVER_PORT`: Port to run Streamlit on (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Address to bind Streamlit to (default: localhost)

## Docker Deployment

### Building the Docker Image

1. Clone the repository:
   ```bash
   git clone https://github.com/autowealthtranslate/auto-wealth-translate.git
   cd auto-wealth-translate
   ```

2. Build the Docker image:
   ```bash
   docker build -t autowealthtranslate .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 autowealthtranslate
   ```

4. Access the application in your browser at http://localhost:8501

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  autowealthtranslate:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=your_api_key_here
```

Run with Docker Compose:
```bash
docker-compose up
```

## Cloud Deployment

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app by pointing to your repository and the `streamlit_app.py` file
4. Add your OpenAI API key in the Streamlit Cloud secrets management

### Heroku Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/autowealthtranslate/auto-wealth-translate.git
   cd auto-wealth-translate
   ```

2. Create a new Heroku app:
   ```bash
   heroku create
   ```

3. Add a `Procfile` with the following content:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

4. Set the OpenAI API key:
   ```bash
   heroku config:set OPENAI_API_KEY=your_api_key_here
   ```

5. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

### AWS Elastic Beanstalk

1. Create an `aws.config` file with the following content:
   ```
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: run.py
     aws:elasticbeanstalk:application:environment:
       OPENAI_API_KEY: your_api_key_here
   ```

2. Create a `run.py` file:
   ```python
   import os
   import subprocess
   import sys
   
   def application(environ, start_response):
       start_response('200 OK', [('Content-Type', 'text/plain')])
       subprocess.Popen([sys.executable, 'streamlit_app.py'])
       return [b"Starting Streamlit application. Please wait a moment and access the application at the root URL."]
   
   if __name__ == '__main__':
       subprocess.call([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"])
   ```

3. Package your application and deploy using the EB CLI:
   ```bash
   eb init -p python-3.8 autowealthtranslate
   eb create autowealthtranslate-env
   eb deploy
   ```

## Performance Considerations

- For optimal performance, ensure at least 2GB of RAM is available
- For processing large documents (>50MB), consider increasing the memory allocation
- If deploying to a server with limited resources, consider using the `gpt-3.5-turbo` model instead of `gpt-4` for faster processing

## Security Considerations

- Never hardcode the OpenAI API key in your application code
- Use environment variables or a secure secrets management system
- For production deployments, consider implementing user authentication
- When deploying to a public server, ensure HTTPS is enabled 