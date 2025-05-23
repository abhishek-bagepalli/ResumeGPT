# ResumeGPT

## Project Description

ResumeGPT is an innovative AI-powered platform designed to revolutionize the way job seekers approach their career development. By combining cutting-edge artificial intelligence with practical career guidance, ResumeGPT offers a comprehensive solution for resume optimization and job matching.

### Core Functionality

The platform leverages advanced natural language processing and machine learning techniques to:

- **Analyze and Optimize Resumes**: Using GPT models to provide detailed feedback on resume content, structure, and formatting, helping users create more impactful resumes that stand out to employers.

- **Smart Job Matching**: Through vector similarity search and semantic analysis, the platform matches resumes with relevant job opportunities, considering both explicit qualifications and implicit skills.

- **LinkedIn Integration**: Automatically scrapes and analyzes job postings from LinkedIn, providing real-time insights into market demands and helping users tailor their applications accordingly.

- **Interactive AI Assistant**: Offers personalized career guidance through an intuitive chat interface, answering questions about resume writing, job search strategies, and career development.

### Technical Innovation

ResumeGPT stands out through its sophisticated technical implementation:

- **Vector Database Architecture**: Utilizes ChromaDB for efficient storage and retrieval of semantic embeddings, enabling fast and accurate job matching.

- **Advanced NLP Pipeline**: Implements a multi-stage processing pipeline that extracts, analyzes, and enhances resume content using state-of-the-art language models.

- **Real-time Processing**: Provides instant feedback and recommendations through Streamlit's interactive interface, making the platform both powerful and user-friendly.

### Target Users

- Job seekers looking to optimize their resumes
- Career changers seeking guidance on skill presentation
- HR professionals and recruiters for candidate evaluation
- Career counselors and advisors for client support

### Impact and Benefits

- **Time Efficiency**: Reduces the time spent on resume optimization and job searching
- **Quality Improvement**: Enhances resume quality through AI-powered suggestions
- **Market Alignment**: Helps align candidate profiles with current market demands
- **Career Growth**: Provides actionable insights for professional development

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