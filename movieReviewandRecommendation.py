import dspy 
from typing import Dict, Any
import streamlit as st
import re

lm = dspy.LM("ollama_chat/llama3.2:3b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

class AdvancedMovieReviewer(dspy.Module):
    """Comprehensive movie review analysis module with quality control"""
    
    def __init__(self):
        super().__init__()
        
        # Define detailed signatures for better output control
        self.analysis = dspy.ChainOfThought(
            dspy.Signature(
                "review -> plot_summary, character_analysis, directing_quality, "
                "cinematography, technical_aspects, cultural_impact, rating",
                "Analyze the movie review in depth. Provide detailed plot summary, "
                "character analysis, and technical evaluation. Rating should be 0-10."
            )
        )
        
        self.genre_classifier = dspy.ChainOfThought(
            dspy.Signature(
                "review -> genres",
                "Identify movie genres from the review. Return as comma-separated list."
            )
        )
        
        self.recommender = dspy.ChainOfThought(
            dspy.Signature(
                "review -> similar_movies, recommendations",
                "Suggest 3 similar movies and 3 recommendations based on review content."
            )
        )

    def forward(self, review: str) -> Dict[str, Any]:
        """Process review with quality checks and fallbacks"""
        try:
            analysis = self.analysis(review=review)
            genres = self.genre_classifier(review=review)
            comparisons = self.recommender(review=review)
            
            return {
                "plot_summary": analysis.plot_summary,
                "character_analysis": analysis.character_analysis,
                "technical_review": {
                    "directing": self._rate_quality(analysis.directing_quality),
                    "cinematography": self._rate_quality(analysis.cinematography),
                    "technical_aspects": self._rate_quality(analysis.technical_aspects)
                },
                "cultural_impact": analysis.cultural_impact,
                "rating": self._parse_rating(analysis.rating),
                "genres": self._format_genres(genres.genres),
                "similar_movies": self._format_list(comparisons.similar_movies),
                "recommendations": self._format_list(comparisons.recommendations)
            }
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return {}

    def _parse_rating(self, rating_str: str) -> float:
        match = re.search(r"(\d+\.?\d*)", rating_str)
        return min(10, max(0, float(match.group(1))) if match else 5.0)

    def _format_genres(self, genres: str) -> list:
        return [g.strip().title() for g in genres.split(",") if g.strip()]

    def _format_list(self, items: str) -> list:
        return [f"- {item.strip()}" for item in items.split(",") if item.strip()]

    def _rate_quality(self, text: str) -> str:
        text = text.lower()
        if "excellent" in text: return "★★★★★"
        if "good" in text: return "★★★★☆"
        if "average" in text: return "★★★☆☆"
        if "poor" in text: return "★★☆☆☆"
        return "★★★☆☆"

st.set_page_config(
    page_title="CineAnalytica Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_REVIEWS = {
    "Positive Sci-Fi": (
        "Christopher Nolan's Interstellar is a breathtaking cosmic odyssey that "
        "masterfully blends emotional depth with scientific accuracy. The stunning "
        "visual effects paired with Hans Zimmer's haunting score create an immersive "
        "experience. Matthew McConaughey delivers a career-best performance, "
        "anchoring the human drama amidst mind-bending theoretical physics concepts."
    ),
    "Critical Drama": (
        "While the performances were strong, the latest superhero movie suffers from "
        "a bloated runtime and excessive CGI. The plot feels recycled from previous "
        "installments, and character development takes a backseat to spectacle. "
        "Despite impressive action sequences, the film lacks the emotional core "
        "that made earlier entries in the franchise memorable."
    )
}

def main():
    """Main application interface"""
    st.title("🎬 CineAnalytica Pro - Advanced Movie Review Analysis")
    
    with st.sidebar:
        st.header("Configuration")
        selected_sample = st.selectbox(
            "Load Sample Review",
            options=list(SAMPLE_REVIEWS.keys()),
            index=0
        )
        analysis_depth = st.radio(
            "Analysis Depth",
            ["Quick Scan", "Comprehensive Analysis"],
            index=1
        )
    
    review = st.text_area(
        "Enter Movie Review:",
        value=SAMPLE_REVIEWS[selected_sample],
        height=250,
        help="Paste your movie review text here for analysis"
    )
    
    if st.button("Analyze Review", type="primary"):
        if not review.strip():
            st.warning("Please enter a review to analyze")
            return
        
        analyzer = AdvancedMovieReviewer()
        
        with st.spinner("🧠 Analyzing content..."):
            try:
                result = analyzer(review)
                display_results(result)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def display_results(result: Dict[str, Any]) -> None:
    """Render analysis results in organized layout"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚡ Quick Insights")
        st.metric("Overall Rating", f"{result['rating']}/10")
        st.write("**Genres Identified:**")
        st.write(", ".join(result["genres"]))
        
        st.divider()
        
        st.subheader("🎯 Recommendations")
        st.write("\n".join(result["recommendations"]))
        
        st.divider()
        
        st.subheader("🍿 Similar Movies")
        st.write("\n".join(result["similar_movies"]))
    
    with col2:
        st.subheader("📖 Detailed Analysis")
        
        with st.expander("Plot Summary", expanded=True):
            st.write(result["plot_summary"])
        
        with st.expander("Character Analysis"):
            st.write(result["character_analysis"])
        
        with st.expander("Technical Evaluation"):
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            tech_col1.metric("Directing", result["technical_review"]["directing"])
            tech_col2.metric("Cinematography", result["technical_review"]["cinematography"])
            tech_col3.metric("Technical Aspects", result["technical_review"]["technical_aspects"])
        
        with st.expander("Cultural Impact"):
            st.write(result["cultural_impact"])
        
        st.divider()
        st.write("🔍 Analysis powered by Llama3 3B via Ollama")

if __name__ == "__main__":
    main()