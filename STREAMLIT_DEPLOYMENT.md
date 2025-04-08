# Deploying AutoWealthTranslate to Streamlit Cloud

This guide provides step-by-step instructions for deploying AutoWealthTranslate to Streamlit Cloud.

## Prerequisites

Before deploying to Streamlit Cloud, make sure you have:

1. A GitHub account
2. A Streamlit Cloud account (sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud))
3. API keys for any translation services you want to use (OpenAI, DeepL, xAI)

## Step 1: Prepare Your Repository

1. Create a GitHub repository for your project (if you haven't already)
2. Push your code to the repository

Ensure your repository contains the following key files:
- `streamlit_cloud.py` (main entry point for Streamlit Cloud)
- `requirements-streamlit.txt` (dependencies for Streamlit)
- `.streamlit/config.toml` (Streamlit configuration)

## Step 2: Set Up Streamlit Cloud

1. Log in to Streamlit Cloud
2. Click "New app"
3. Connect your GitHub repository
4. Configure the app:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your preferred branch)
   - **Main file path**: `streamlit_cloud.py` (use this as the entry point)
   - **Python version**: Select Python 3.9 or higher (3.9, 3.10, 3.11, or 3.12)

## Step 3: Set Up Secrets

Streamlit Cloud allows you to securely store API keys and other sensitive information:

1. In your Streamlit Cloud app settings, find the "Secrets" section
2. Click "Edit Secrets"
3. Add your API keys using the format from `.streamlit/secrets_template.toml`:

```toml
[openai]
api_key = "your-openai-api-key"

[deepl]
api_key = "your-deepl-api-key"

[xai]
api_key = "your-xai-api-key"
```

## Step 4: Advanced Settings (Optional)

For larger documents or more complex processing, you may need to adjust the app resources:

1. In your Streamlit Cloud app settings, find "Advanced settings"
2. Increase the memory limit if needed (recommend at least 2GB)
3. Enable "Persistent file storage" if you need to save files between sessions

## Step 5: Deploy and Test

1. Click "Deploy" in Streamlit Cloud
2. Wait for the build and deployment process to complete
3. Test your application by:
   - Uploading small documents first
   - Testing all translation models and languages
   - Verifying formatting is preserved in translations

## Troubleshooting

If you encounter issues during deployment:

1. **Dependencies not installing**: Check `requirements-streamlit.txt` for compatibility issues
   - Some packages may need to be pinned to specific versions
   - Try using `opencv-python-headless` instead of `opencv-python`

2. **Memory errors**: Large documents may require more memory
   - Upgrade to a higher tier in Streamlit Cloud
   - Add pagination or file size limits in your app

3. **Missing API keys**: Verify your secrets are configured correctly
   - Check Streamlit Cloud logs for any key errors
   - Verify the secrets format matches what your code expects

4. **OCR issues**: Tesseract may need additional configuration
   - Provide explicit path configuration in your code
   - Use cloud OCR services as an alternative

## Performance Optimization

To improve performance on Streamlit Cloud:

1. Use the `@st.cache_data` decorator for expensive operations
2. Process documents in chunks to avoid memory issues
3. Store temporary files in the session state, not on disk
4. Use asynchronous processing for long-running translations
5. Implement a progress indicator for better user experience

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching) 