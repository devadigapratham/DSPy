import dspy
import streamlit as st
from typing import Dict, List
import re

# Configure DSPy with local Ollama model
lm = dspy.LM(
    model="ollama_chat/llama3.2:3b",
    api_base="http://localhost:11434",
    api_key=""
)
dspy.configure(lm=lm)

class ResumeAnalyzer(dspy.Module):    
    def __init__(self):
        super().__init__()
        
        self.section_identifier = dspy.ChainOfThought(
            dspy.Signature(
                "resume_text -> sections",
                "Identify key resume sections from the text. Return as comma-separated list."
            )
        )
        
        self.content_evaluator = dspy.ChainOfThought(
            dspy.Signature(
                "section, text -> analysis, score",
                "Analyze this resume section for clarity, relevance, and impact. "
                "Provide critical feedback and score 1-10."
            )
        )
        
        self.overall_assessor = dspy.ChainOfThought(
            dspy.Signature(
                "resume_text -> summary, strengths, weaknesses, recommendations",
                "Provide overall resume assessment with key strengths, weaknesses, "
                "and actionable improvement suggestions."
            )
        )

    def forward(self, resume_text: str) -> Dict[str, any]:
        """Process resume with hierarchical analysis"""
        try:
            # Identify sections
            sections = self.section_identifier(resume_text=resume_text).sections
            section_list = [s.strip() for s in sections.split(",") if s.strip()]
            
            # Analyze each section
            section_analyses = {}
            for section in section_list:
                result = self.content_evaluator(section=section, text=resume_text)
                section_analyses[section] = {
                    "analysis": result.analysis,
                    "score": self._parse_score(result.score)
                }
            
            # Get overall assessment
            overall = self.overall_assessor(resume_text=resume_text)
            
            return {
                "sections": section_list,
                "section_analyses": section_analyses,
                "overall_summary": overall.summary,
                "strengths": self._format_list(overall.strengths),
                "weaknesses": self._format_list(overall.weaknesses),
                "recommendations": self._format_list(overall.recommendations)
            }
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return {}

    def _parse_score(self, score_str: str) -> float:
        """Extract numerical score from text response"""
        match = re.search(r"(\d+\.?\d*)", score_str)
        return min(10, max(1, float(match.group(1))) if match else 5.0)

    def _format_list(self, items: str) -> List[str]:
        """Convert text list to bullet points"""
        return [f"- {item.strip()}" for item in items.split(";") if item.strip()]

# Streamlit UI Configuration
st.set_page_config(
    page_title="CareerCompass Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_RESUMES = {
    "Software Engineer": """
John Doe
San Francisco, CA | john@email.com | GitHub: johndoe

SUMMARY:
Full-stack developer with 5+ years experience building scalable web applications. 
Proficient in Python, JavaScript, and cloud technologies.

EXPERIENCE:
Senior Developer @ TechCorp (2020-Present)
- Led team of 5 developers in building SaaS platform
- Implemented CI/CD pipeline reducing deployment time by 40%
- Tech stack: React, Node.js, AWS

EDUCATION:
BS Computer Science - Stanford University (2016-2020)
GPA: 3.8

SKILLS:
Python, JavaScript, AWS, Docker, React, SQL
    """,
    "Marketing Manager": """
Jane Smith
New York, NY | jane@email.com | Portfolio: janesmith.com

PROFILE:
Data-driven marketing professional with 7 years experience in digital campaigns.

EXPERIENCE:
Marketing Director @ AdWorld (2018-Present)
- Managed $2M annual budget
- Increased ROAS by 150% through optimized targeting
- Led 10-person cross-functional team

SKILLS:
SEO, Google Analytics, Facebook Ads, Team Leadership

EDUCATION:
MBA - NYU Stern School of Business
    """
}

def main():
    """Main application interface"""
    st.title("📄 Resume Analysis System")
    
    with st.sidebar:
        st.header("Options")
        selected_sample = st.selectbox(
            "Load Sample Resume",
            options=list(SAMPLE_RESUMES.keys())
        )
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Quick Scan", "Detailed Evaluation"],
            index=1
        )
    
    resume_text = st.text_area(
        "Paste Resume/CV Text:",
        value=SAMPLE_RESUMES[selected_sample],
        height=400,
        help="Copy-paste your resume text for analysis"
    )
    
    if st.button("Analyze Resume", type="primary"):
        if len(resume_text.split()) < 50:
            st.warning("Please provide a more detailed resume (minimum 50 words)")
            return
        
        analyzer = ResumeAnalyzer()
        
        with st.spinner("🔍 Analyzing resume content..."):
            try:
                result = analyzer(resume_text)
                display_results(result, analysis_mode)
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

def display_results(result: Dict[str, any], mode: str) -> None:
    """Render analysis results in organized layout"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Resume Sections")
        for section in result.get("sections", []):
            st.write(f"- {section}")
        
        st.divider()
        
        st.subheader("📈 Section Scores")
        for section, data in result.get("section_analyses", {}).items():
            st.write(f"**{section}**")
            st.progress(data["score"]/10)
            if mode == "Detailed Evaluation":
                with st.expander("Analysis Details"):
                    st.write(data["analysis"])
    
    with col2:
        st.subheader("📊 Overall Assessment")
        st.write(result.get("overall_summary", ""))
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["Strengths", "Weaknesses", "Recommendations"])
        
        with tab1:
            st.write("\n".join(result.get("strengths", [])))
        
        with tab2:
            st.write("\n".join(result.get("weaknesses", [])))
        
        with tab3:
            st.write("\n".join(result.get("recommendations", [])))
        
        st.divider()
        st.write("🤖 Analysis powered by Llama3.2 3B via Ollama")

if __name__ == "__main__":
    main()