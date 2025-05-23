# ResumeGPT

ResumeGPT is an intelligent resume analysis and job matching platform that leverages AI to help users optimize their resumes and find relevant job opportunities. The application uses advanced natural language processing and machine learning techniques to provide personalized insights and recommendations.

## Features

- Resume Analysis and Optimization
- Job Matching and Recommendations
- LinkedIn Job Scraping
- Interactive Chat Interface
- PDF Resume Processing
- Vector Database Integration (ChromaDB)

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: 
  - OpenAI GPT Models
  - LangChain
  - FAISS (Vector Similarity Search)
- **Data Storage**: 
  - ChromaDB (Vector Database)
  - JSON for structured data
- **Cloud Services**:
  - Google Cloud Platform (Service Account Integration)

## Prerequisites

- Python 3.8 or higher
- OpenAI API Key
- Google Cloud Platform Account (for certain features)
- LinkedIn API Access (for job scraping features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ResumeGPT.git
cd ResumeGPT
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.streamlit/secrets.toml` file with the following structure:
```toml
OPENAI_API_KEY = "your-openai-api-key"
AUTHORIZATION = "your-authorization-token"
[google_creds]
# Your Google Cloud credentials
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Upload your resume or use the chat interface to interact with the AI assistant

## Project Structure

```
ResumeGPT/
├── .streamlit/           # Streamlit configuration
├── chroma_db/           # Vector database storage
├── .devcontainer/       # Development container configuration
├── app.py              # Main Streamlit application
├── resume_reader.py    # Resume processing module
├── scrape_linkedin.py  # LinkedIn job scraping module
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- API keys and sensitive credentials are stored in `.streamlit/secrets.toml`
- The `.gitignore` file is configured to prevent sensitive data from being committed
- Google Cloud credentials are managed through service accounts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Streamlit for the web application framework
- LangChain for the AI/ML framework
- ChromaDB for vector database capabilities 