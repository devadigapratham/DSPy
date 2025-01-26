# DSPy Exploration Project 

A simple pair of systems built to explore DSPy's capabilities. This repo contains:

## 🎥 Movie Analysis
* Classifies movie genres from descriptions
* Detects key themes and sentiment
* Basic recommendation engine

## 📄 Resume Evaluator
* Identifies resume sections
* Scores content quality
* Gives improvement suggestions

## Why? 🤔
Built purely out of curiosity to:
* Experiment with DSPy's LM pipelines
* Test prompt optimization (MIPRO)
* Learn by building end-to-end systems

## Quick Start ▶️

1. **Install requirements**:
```bash
pip install dspy streamlit ollama  
```

2. **Run Ollama**:
```bash
ollama serve &  
ollama pull llama3.2:3b  
```

3. **Try the apps**:
```bash
# For movies  
streamlit run movie_analysis.py  

# For resumes  
streamlit run resume_evaluator.py  
```

## Notes 📌
* **Not production-ready** - just experimental code!
* Uses local Ollama models (llama3.2:3b)
* Code shows basic DSPy patterns I wanted to test