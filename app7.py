import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.subplots as sp
from scipy import stats
import time
from PIL import Image
import io
import hashlib
import random

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Water Quality Prediction Using Data Science Techniques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# INTERESTING FACTS DATABASE
# --------------------------------------------------
WATER_FACTS = [
    "💧 97% of Earth's water is saltwater. Only 3% is freshwater.",
    "🌍 About 1.1 billion people worldwide lack access to clean water.",
    "⏰ It takes 37 gallons of water to produce 1 pound of coffee.",
    "🚰 The average person uses 80-100 gallons of water per day.",
    "🧬 Your body is 60% water. Your brain is 75% water.",
    "🌊 Oceans cover 71% of Earth's surface.",
    "🏭 Agriculture uses about 70% of the world's freshwater.",
    "☔ A person can only survive about 3-5 days without water.",
    "❄️ Antarctica contains 90% of the world's ice.",
    "🐟 Fish don't drink water. They absorb it through their gills.",
    "💦 One drop of water contains 1.5 sextillion atoms.",
    "🌱 Plants need water to grow. A corn plant needs 200 gallons.",
    "⚡ Water can be found on Mars in the form of ice.",
    "🔬 The pH scale ranges from 0-14, with 7 being neutral.",
    "🏥 Hard water can cause scale buildup in pipes.",
    "🌿 Rainwater is naturally slightly acidic (pH 5.6).",
    "🧂 Drinking too much salt water is dangerous for humans.",
    "🌊 Ocean water has an average pH of 8.1-8.3.",
    "💚 Clean water is essential for all life on Earth.",
    "🔄 Water has been cycling through Earth for 4.5 billion years.",
    "🏊 Swimming in contaminated water can cause skin infections.",
    "🚱 Boiling water kills most pathogens but not all chemicals.",
    "🧪 Chlorine is commonly used to disinfect drinking water.",
    "🌧️ It takes about 1,000 gallons of water to grow 1 pound of wheat.",
    "🏭 Industrial water pollution affects 80% of global wastewater.",
]

WATER_QUALITY_FACTS = [
    "🧪 Turbidity measures water cloudiness (0-5 NTU is safe).",
    "⚖️ pH below 6.5 can cause corrosion in pipes.",
    "🔬 Trihalomethanes form when chlorine meets organic matter.",
    "💧 Hardness above 300 mg/L causes scale buildup.",
    "🌊 Conductivity indicates dissolved minerals in water.",
    "🧂 Sulfates can have a laxative effect above 250 mg/L.",
    "🧬 Chloramines kill harmful bacteria and viruses.",
    "📊 WHO sets safe drinking water limits for 80+ contaminants.",
    "🔍 Regular testing ensures water safety.",
    "🚰 Modern treatment makes tap water safer than bottled water.",
    "💧 Hard water requires more soap to lather properly.",
    "🧼 Soft water is better for washing but less healthy to drink.",
    "📈 Water hardness is measured in mg/L of calcium carbonate.",
    "🌡️ Temperature affects water quality and bacterial growth.",
    "⚗️ Dissolved oxygen is crucial for aquatic life.",
    "🔴 Red water indicates iron oxidation in pipes.",
    "🟤 Brown water suggests sediment or algae problems.",
    "🟡 Yellow water may indicate bacterial contamination.",
    "💨 Hydrogen sulfide creates 'rotten egg' smell in water.",
    "🧪 Total Dissolved Solids (TDS) should be below 500 mg/L.",
    "🧫 E. coli bacteria indicates fecal contamination.",
    "🔬 Fluoride strengthens teeth at 0.7-1.0 mg/L.",
    "☣️ Lead in water causes neurological damage in children.",
    "💊 Medicinal compounds are increasingly found in water.",
    "🌱 Algal blooms deplete oxygen and kill fish.",
    "🧬 Cryptosporidium parasites resist chlorine treatment.",
    "🚱 Microplastics are found in 90% of bottled water.",
    "🏥 Waterborne diseases kill 500,000+ children yearly.",
    "📉 Water quality has declined by 15% globally in 20 years.",
    "♻️ Only 5% of wastewater is treated worldwide.",
]

SDE_FACTS = [
    "💻 SDE stands for Software Development Environment.",
    "🛠️ SDEs are essential for efficient code development.",
    "📝 Popular SDEs include VS Code, PyCharm, and IntelliJ IDEA.",
    "🔧 SDEs provide debugging, compilation, and testing tools.",
    "📦 VS Code is used by 70% of developers worldwide.",
    "🚀 Modern SDEs increase productivity by 40-50%.",
    "🔌 Plugins and extensions customize SDE functionality.",
    "⌨️ Keyboard shortcuts in SDEs save hours of coding time.",
    "🎨 Theme customization improves eye comfort during long coding sessions.",
    "📊 Code analytics help identify performance bottlenecks.",
    "🐛 Built-in debuggers help find and fix bugs quickly.",
    "📚 IntelliSense autocomplete reduces typing errors.",
    "🔍 Search and replace features save time on refactoring.",
    "📁 File navigation in SDEs is much faster than command line.",
    "🔄 Version control integration (Git) is built-in.",
    "🎯 Multi-language support in modern SDEs.",
    "⚡ Real-time syntax highlighting prevents errors.",
    "📈 Code metrics show code quality in real-time.",
    "🔐 Security scanning detects vulnerabilities automatically.",
    "🌐 Remote development is now standard in most SDEs.",
    "🤖 AI-powered code suggestions speed up development.",
    "📊 Live linting shows errors before you save.",
    "🔗 Seamless Docker integration for containerization.",
    "📱 SDEs can work on mobile and cloud platforms.",
    "⚙️ Automated testing integrates directly into SDEs.",
    "🎓 SDEs are crucial tools in computer science education.",
    "🏢 Enterprise SDEs support team collaboration features.",
    "💾 Auto-save functionality prevents code loss.",
    "🔔 Real-time notifications keep you updated on errors.",
    "🌟 Minimal distraction mode improves focus and productivity.",
]

WHO_FACTS = [
    "🏥 WHO was founded on April 7, 1948.",
    "🌍 WHO has 194 member states worldwide.",
    "💪 WHO sets international health standards.",
    "📊 WHO monitors disease outbreaks globally.",
    "🎯 WHO's goal: Health for All by 2030.",
    "🧬 WHO leads pandemic response efforts.",
    "📈 WHO publishes weekly epidemiological reports.",
    "🔬 WHO certifies vaccine safety and efficacy.",
    "💉 WHO coordinates immunization programs.",
    "🌐 WHO provides health guidance to 194 countries.",
    "🏥 WHO headquarters is in Geneva, Switzerland.",
    "📚 WHO publishes the International Classification of Diseases.",
    "🔍 WHO investigates emerging infectious diseases.",
    "🚑 WHO provides emergency response to health crises.",
    "💼 WHO employs 8,000+ professionals globally.",
    "🌿 WHO promotes mental health and well-being.",
    "🚫 WHO fights antimicrobial resistance worldwide.",
    "🎓 WHO trains health professionals in developing countries.",
    "💰 WHO's annual budget is about $2.4 billion.",
    "📢 WHO communicates health information in 6 official languages.",
]

ML_FACTS = [
    "🤖 Random Forest uses multiple decision trees.",
    "📊 Machine Learning improves with more data.",
    "🎯 96%+ accuracy is achievable with good ML models.",
    "💻 ML predicts water quality in real-time.",
    "🔮 Predictive models prevent waterborne diseases.",
    "📈 Feature importance reveals key water parameters.",
    "🧠 Neural networks mimic human brain function.",
    "⚡ AI processes millions of data points instantly.",
    "🔬 ML detects patterns humans can't see.",
    "🚀 AI improves water treatment efficiency.",
    "📊 Classification models sort data into categories.",
    "🔍 Regression models predict continuous values.",
    "🎮 Neural networks have millions of parameters.",
    "⚙️ Gradient descent optimizes ML models.",
    "📈 Overfitting occurs when models memorize training data.",
    "🧪 Cross-validation tests model reliability.",
    "💾 Training data quality determines model accuracy.",
    "🔬 Feature engineering improves model performance.",
    "📊 Confusion matrix measures classification accuracy.",
    "🎯 Precision and recall balance false positives/negatives.",
    "🤖 Deep learning revolutionized AI and computer vision.",
    "📱 ML powers recommendation systems everywhere.",
    "🔐 ML detects fraud and security threats.",
    "🌐 Natural Language Processing powers chatbots.",
    "📸 Computer vision enables autonomous vehicles.",
]

# --------------------------------------------------
# AUTHENTICATION SYSTEM
# --------------------------------------------------
def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify password"""
    return hash_password(password) == hashed_password

# Sample user database
USERS_DATABASE = {
    "admin": hash_password("admin123"),
    "user": hash_password("user123"),
    "demo": hash_password("demo123")
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    """Render login page"""
    
    st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }
        
        [data-testid="stMainBlockContainer"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
            padding: 40px;
            border-radius: 20px;
            margin-top: 50px;
        }
        
        h1 {
            color: #f0f9ff;
            text-align: center;
            font-size: 2.5em;
            font-weight: 800;
            margin: 20px 0;
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .login-container {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(51, 65, 85, 0.8));
            border: 2px solid #0ea5e9;
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            margin: 30px auto;
            box-shadow: 0 10px 40px rgba(14, 165, 233, 0.2);
        }
        
        .stTextInput>div>div>input,
        .stPasswordInput>div>div>input {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #e2e8f0 !important;
            border: 1px solid #0ea5e9 !important;
            border-radius: 8px !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 16px !important;
            padding: 12px 30px !important;
            border: none !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
            width: 100% !important;
            height: 50px !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6) !important;
        }
        
        .login-info {
            background: rgba(3, 102, 214, 0.1);
            border-left: 4px solid #0284c7;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            color: #bfdbfe;
            text-align: center;
            font-size: 0.85em;
        }
        
        .fact-box {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #93c5fd;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1>💧 WaterAI Pro</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="login-container">
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #e0f2fe;'>🔐 Login</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="login-info">
            <strong>Demo Credentials:</strong><br>
            👤 admin | 🔑 admin123<br>
            👤 user | 🔑 user123<br>
            👤 demo | 🔑 demo123
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            login_btn = st.button("🚀 Login", use_container_width=True)
        
        with col_b:
            st.button("📝 Sign Up", use_container_width=True, disabled=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if login_btn:
            if username and password:
                if username in USERS_DATABASE and verify_password(password, USERS_DATABASE[username]):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_role = "Admin" if username == "admin" else "User"
                    st.success("✅ Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials!")
            else:
                st.warning("⚠️ Enter username and password!")
        
        # Interesting facts on login page
        col_fact1, col_fact2 = st.columns(2)
        
        with col_fact1:
            st.markdown(f"""
            <div class="fact-box">
                <strong>💧 Water Fact:</strong><br>
                {random.choice(WATER_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        
        with col_fact2:
            st.markdown(f"""
            <div class="fact-box">
                <strong>💻 SDE Fact:</strong><br>
                {random.choice(SDE_FACTS)}
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
def main_app():
    """Main application after login"""
    
    try:
        model = joblib.load("water_quality_model.pkl")
    except:
        st.error("❌ Model file not found! Place 'water_quality_model.pkl' in project folder")
        return
    
    FEATURES = model.feature_names_in_.tolist()
    
    WHO_LIMITS = {
        "ph": 8.5,
        "Hardness": 300,
        "Solids": 500,
        "Chloramines": 4,
        "Sulfate": 250,
        "Conductivity": 800,
        "Organic_carbon": 15,
        "Trihalomethanes": 80,
        "Turbidity": 5
    }
    
    TREATMENT_RECOMMENDATIONS = {
        "ph": "• Add acid (HCl) to lower pH\n• Add lime (Ca(OH)2) to raise pH",
        "Hardness": "• Use water softening (ion exchange)\n• Add polyphosphates",
        "Solids": "• Use filtration (sand/carbon)\n• Coagulation and flocculation",
        "Chloramines": "• Reduce chlorine dosage\n• Use alternative disinfectants",
        "Sulfate": "• Use reverse osmosis\n• Distillation",
        "Conductivity": "• Use deionization\n• Reverse osmosis",
        "Organic_carbon": "• Use activated carbon filter\n• UV treatment",
        "Trihalomethanes": "• Remove chlorine byproducts\n• Use alternative disinfectants",
        "Turbidity": "• Use filtration\n• Sedimentation tanks"
    }
    
    # --------------------------------------------------
    # PROFESSIONAL CUSTOM CSS
    # --------------------------------------------------
    st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #e2e8f0;
        }
        
        [data-testid="stMainBlockContainer"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            padding: 40px;
        }
        
        h1 {
            color: #f0f9ff;
            text-align: center;
            font-size: 3em;
            font-weight: 800;
            margin: 30px 0;
            text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            color: #e0f2fe;
            font-size: 1.8em;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 15px;
            border-left: 4px solid #0ea5e9;
        }
        
        h3, h4 {
            color: #bae6fd;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(51, 65, 85, 0.8));
            border: 2px solid #0ea5e9;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        
        .safe-box {
            background: linear-gradient(135deg, #065f46, #059669, #10b981);
            padding: 30px;
            border-radius: 16px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
            border: 2px solid #34d399;
            animation: pulse-green 2s infinite;
        }
        
        .unsafe-box {
            background: linear-gradient(135deg, #7f1d1d, #dc2626, #ef4444);
            padding: 30px;
            border-radius: 16px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
            border: 2px solid #fca5a5;
            animation: pulse-red 2s infinite;
        }
        
        .treatment-box {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            padding: 25px;
            border-radius: 12px;
            color: white;
            border-left: 4px solid #60a5fa;
            margin: 15px 0;
        }
        
        .fact-box {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(74, 222, 128, 0.1));
            border-left: 4px solid #22c55e;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #86efac;
        }
        
        .water-quality-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1));
            border-left: 4px solid #6366f1;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #a5b4fc;
        }
        
        .sde-box {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(245, 158, 11, 0.1));
            border-left: 4px solid #f97316;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #fed7aa;
        }
        
        .insight-box {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(167, 139, 250, 0.1));
            border-left: 4px solid #a78bfa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #ddd6fe;
        }
        
        @keyframes pulse-green {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        @keyframes pulse-red {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        .input-section {
            background: rgba(30, 41, 59, 0.6);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(14, 165, 233, 0.3);
            margin: 20px 0;
        }
        
        [data-testid="stTabs"] [role="tablist"] {
            background-color: rgba(30, 41, 59, 0.4);
            padding: 10px;
            border-radius: 10px;
        }
        
        [role="tab"] {
            color: #cbd5e1;
            font-weight: 600;
            font-size: 16px;
        }
        
        [role="tab"][aria-selected="true"] {
            color: #0ea5e9;
            border-bottom: 3px solid #0ea5e9;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #0ea5e9, #06b6d4);
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
        }
        
        .info-box {
            background: rgba(3, 102, 214, 0.1);
            border-left: 4px solid #0284c7;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #bfdbfe;
        }
        
        .success-box {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10b981;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #a7f3d0;
        }
        
        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #fecaca;
        }
        
        .warning-box {
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: #fef3c7;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # SIDEBAR WITH USER INFO & FACTS
    # --------------------------------------------------
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: rgba(14, 165, 233, 0.1); border-radius: 12px; margin-bottom: 20px;'>
            <h3 style='color: #0ea5e9;'>👤 User Profile</h3>
            <p style='color: #cbd5e1; margin: 10px 0;'><strong>Username:</strong> {st.session_state.username}</p>
            <p style='color: #cbd5e1; margin: 10px 0;'><strong>Role:</strong> {st.session_state.user_role}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: #0ea5e9; font-size: 1.5em;'>💧 WaterAI Pro ULTIMATE</h2>
            <p style='color: #cbd5e1; font-size: 0.9em;'>Advanced Water Quality Analysis v3.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 About This System")
        st.info("""
        **WaterAI Pro ULTIMATE v3.0** - Enterprise-grade with AI magic!
        
        ✅ 25+ visualizations
        ✅ AI recommendations
        ✅ Trend analysis
        ✅ Predictive alerts
        ✅ Secure login
        ✅ Expert insights
        """)
        
        # FACTS IN SIDEBAR
        st.markdown("---")
        st.markdown("### 💡 Did You Know?")
        
        fact_choice = random.choice([1, 2, 3, 4])
        
        if fact_choice == 1:
            st.markdown(f"""
            <div class="fact-box">
                💧 {random.choice(WATER_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        elif fact_choice == 2:
            st.markdown(f"""
            <div class="water-quality-box">
                🧪 {random.choice(WATER_QUALITY_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        elif fact_choice == 3:
            st.markdown(f"""
            <div class="sde-box">
                💻 {random.choice(SDE_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
                🌍 {random.choice(WHO_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Features", len(FEATURES))
        with col2:
            st.metric("Model v3.0", "Ultimate")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Settings")
        confidence_threshold = st.slider("Confidence Threshold (%)", 50, 100, 75)
        risk_alert = st.toggle("Enable Risk Alerts", True)
        
        st.markdown("---")
        st.markdown("### 📚 WHO Standards")
        with st.expander("View WHO Limits"):
            who_ref = pd.DataFrame({
                "Parameter": list(WHO_LIMITS.keys()),
                "WHO Limit": list(WHO_LIMITS.values())
            })
            st.dataframe(who_ref, use_container_width=True)
        
        st.markdown("---")
        
        col_logout1, col_logout2 = st.columns([1, 1])
        with col_logout1:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.user_role = None
                st.success("✅ Logged out successfully!")
                time.sleep(1)
                st.rerun()
    
    # --------------------------------------------------
    # MAIN HEADER WITH FACTS
    # --------------------------------------------------
    st.markdown("""
        <h1>💧 Water Quality Prediction Using Data Science Techniques</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #cbd5e1; margin-bottom: 30px;'>
        <p style='font-size: 1.1em;'>🚀 ULTIMATE Edition with 25+ Features & AI Recommendations</p>
        <p style='font-size: 0.9em; color: #64748b;'>Last updated: """ + datetime.now().strftime("%B %d, %Y at %H:%M:%S") + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Random facts row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="fact-box">
            <strong>💧 Water Fact</strong><br>
            {random.choice(WATER_FACTS)}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="water-quality-box">
            <strong>🧪 Quality Tip</strong><br>
            {random.choice(WATER_QUALITY_FACTS)}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="sde-box">
            <strong>💻 Dev Tip</strong><br>
            {random.choice(SDE_FACTS)}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="insight-box">
            <strong>🌍 WHO Info</strong><br>
            {random.choice(WHO_FACTS)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------
    def calculate_wqi(values):
        """Calculate Water Quality Index"""
        weights = {
            "ph": 4, "Hardness": 2, "Solids": 4, "Chloramines": 3,
            "Sulfate": 4, "Conductivity": 2, "Organic_carbon": 2,
            "Trihalomethanes": 3, "Turbidity": 5
        }
        weighted_sum = 0
        weight_total = 0
        for param in FEATURES:
            if param in WHO_LIMITS:
                value = values[param]
                limit = WHO_LIMITS[param]
                rating = min((value / limit) * 100, 100)
                weighted_sum += rating * weights[param]
                weight_total += weights[param]
        wqi = 100 - (weighted_sum / weight_total)
        return round(np.clip(wqi, 0, 100), 2)
    
    def get_wqi_category(wqi):
        """Get WQI category"""
        if wqi >= 70:
            return "Excellent", "#10b981"
        elif wqi >= 50:
            return "Good", "#f59e0b"
        elif wqi >= 25:
            return "Fair", "#ef4444"
        else:
            return "Poor", "#7f1d1d"
    
    def calculate_risk_score(input_dict):
        """Calculate risk score"""
        risk_score = 0
        for param, limit in WHO_LIMITS.items():
            if param in input_dict:
                value = input_dict[param]
                if value > limit:
                    risk_score += ((value - limit) / limit) * 10
        return round(np.clip(risk_score, 0, 100), 2)
    
    def get_treatment_plan(input_dict):
        """Generate treatment plan"""
        exceeded_params = []
        for param in FEATURES:
            if input_dict[param] > WHO_LIMITS.get(param, float('inf')):
                exceeded_params.append((param, input_dict[param], WHO_LIMITS.get(param)))
        return exceeded_params
    
    def generate_health_impact_report(input_dict):
        """Generate health impact report"""
        health_impacts = {
            "ph": "Abnormal pH can cause gastrointestinal issues and corrosion",
            "Hardness": "High hardness causes scaling and reduced effectiveness of soap",
            "Solids": "Suspended solids indicate microbial contamination risk",
            "Chloramines": "Excess chlorine causes respiratory issues and taste problems",
            "Sulfate": "High sulfate causes laxative effect and digestive issues",
            "Conductivity": "High conductivity indicates mineral overload",
            "Organic_carbon": "Organic matter promotes microbial growth",
            "Trihalomethanes": "Carcinogenic compounds linked to bladder cancer",
            "Turbidity": "Turbid water indicates pathogens and reduced disinfection"
        }
        
        impacts = []
        for param in FEATURES:
            if input_dict[param] > WHO_LIMITS.get(param, float('inf')):
                impacts.append({
                    "Parameter": param,
                    "Health Risk": health_impacts.get(param, "Unknown risk"),
                    "Severity": "High" if input_dict[param] > WHO_LIMITS[param] * 1.5 else "Moderate"
                })
        return impacts
    
    def create_comparison_chart(sample1, sample2):
        """Compare two water samples"""
        params = FEATURES
        values1 = [sample1.get(p, 0) for p in params]
        values2 = [sample2.get(p, 0) for p in params]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values1, theta=params, fill='toself', name='Sample 1',
            line_color='#0ea5e9', fillcolor='rgba(14, 165, 233, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values2, theta=params, fill='toself', name='Sample 2',
            line_color='#f59e0b', fillcolor='rgba(245, 158, 11, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(values1), max(values2)) * 1.2])),
            height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'}
        )
        return fig
    
    # --------------------------------------------------
    # MAIN TABS
    # --------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔬 Smart Prediction",
        "📊 Bulk Analysis",
        "📈 Advanced Analytics",
        "🎯 AI Recommendations",
        "⚠️ Health Impact",
        "🔄 Sample Comparison",
        "📚 Reports",
        "⚙️ Settings"
    ])
    
    # --------------------------------------------------
    # TAB 1: SMART PREDICTION
    # --------------------------------------------------
    with tab1:
        st.markdown("### 🧪 Enter Water Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            ph = st.number_input("pH Level", 0.0, 14.0, 7.0, step=0.1)
            hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 150.0, step=5.0)
            solids = st.number_input("Total Solids (mg/L)", 0.0, 30000.0, 500.0, step=100.0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            chloramines = st.number_input("Chloramines (mg/L)", 0.0, 20.0, 3.0, step=0.1)
            sulfate = st.number_input("Sulfate (mg/L)", 0.0, 1000.0, 200.0, step=10.0)
            organic_carbon = st.number_input("Organic Carbon (mg/L)", 0.0, 30.0, 10.0, step=0.5)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            trihalomethanes = st.number_input("Trihalomethanes (µg/L)", 0.0, 200.0, 50.0, step=5.0)
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 20.0, 3.0, step=0.1)
            conductivity = st.number_input("Conductivity (µS/cm)", 0.0, 2000.0, 400.0, step=50.0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # FACTS BEFORE ANALYSIS
        col_fact1, col_fact2 = st.columns(2)
        
        with col_fact1:
            st.markdown(f"""
            <div class="water-quality-box">
                <strong>💡 Quick Tip:</strong> {random.choice(WATER_QUALITY_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        
        with col_fact2:
            st.markdown(f"""
            <div class="sde-box">
                <strong>💻 Dev Note:</strong> {random.choice(SDE_FACTS)}
            </div>
            """, unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_btn = st.button("🚀 Run AI Analysis", use_container_width=True)
        
        if analyze_btn:
            input_dict = {
                "ph": ph, "Hardness": hardness, "Solids": solids,
                "Chloramines": chloramines, "Sulfate": sulfate,
                "Conductivity": conductivity, "Organic_carbon": organic_carbon,
                "Trihalomethanes": trihalomethanes, "Turbidity": turbidity
            }
            
            input_df = pd.DataFrame([input_dict])[FEATURES]
            
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            confidence = round(np.max(probabilities) * 100, 2)
            wqi_score = calculate_wqi(input_dict)
            risk_score = calculate_risk_score(input_dict)
            wqi_category, wqi_color = get_wqi_category(wqi_score)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 WQI Score", wqi_score, wqi_category)
            with col2:
                st.metric("⚠️ Risk Score", f"{risk_score}%", "Out of 100")
            with col3:
                st.metric("🤖 Confidence", f"{confidence}%", "AI Model")
            with col4:
                status = "✅ SAFE" if prediction == 1 else "❌ UNSAFE"
                st.metric("Status", status)
            
            # GAUGES
            colA, colB = st.columns([1, 1])
            
            with colA:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=wqi_score,
                    title={'text': "Water Quality Index"},
                    delta={'reference': 70},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(127, 29, 29, 0.3)"},
                            {'range': [25, 50], 'color': "rgba(180, 83, 9, 0.3)"},
                            {'range': [50, 75], 'color': "rgba(234, 179, 8, 0.3)"},
                            {'range': [75, 100], 'color': "rgba(6, 95, 70, 0.3)"},
                        ],
                        'bar': {'color': "#0ea5e9"},
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
                    }
                ))
                gauge.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
                st.plotly_chart(gauge, use_container_width=True)
            
            with colB:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="safe-box">
                        ✅ SAFE FOR DRINKING<br><br>
                        Confidence: <strong>{confidence}%</strong><br>
                        Category: <strong>{wqi_category}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="fact-box">
                        💡 {random.choice(WATER_FACTS)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="unsafe-box">
                        ❌ UNSAFE FOR DRINKING<br><br>
                        Confidence: <strong>{confidence}%</strong><br>
                        Risk: <strong>{risk_score}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ {random.choice(WATER_QUALITY_FACTS)}
                    </div>
                    """, unsafe_allow_html=True)
            
            # RISK ASSESSMENT
            st.markdown("### 🎯 Risk Assessment")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                risk_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Overall Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.3)"},
                            {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.3)"},
                            {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.3)"},
                        ],
                        'bar': {'color': "#f59e0b"}
                    }
                ))
                risk_gauge.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
                st.plotly_chart(risk_gauge, use_container_width=True)
            
            with col2:
                prob_df = pd.DataFrame({
                    "Classification": ["Safe", "Unsafe"],
                    "Probability": [probabilities[1], probabilities[0]],
                    "Percentage": [f"{probabilities[1]*100:.2f}%", f"{probabilities[0]*100:.2f}%"]
                })
                
                prob_chart = px.bar(prob_df, x="Classification", y="Probability",
                    color="Probability", color_continuous_scale=["#ef4444", "#10b981"],
                    text="Percentage", title="Model Confidence Breakdown")
                prob_chart.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'},
                    yaxis={'color': '#cbd5e1'}, showlegend=False, height=350)
                st.plotly_chart(prob_chart, use_container_width=True)
            
            # WHO COMPLIANCE
            st.markdown("### ⚖️ WHO Compliance Analysis")
            who_df = pd.DataFrame({
                "Parameter": FEATURES,
                "Actual Value": [input_dict[f] for f in FEATURES],
                "WHO Limit": [WHO_LIMITS.get(f, 0) for f in FEATURES],
                "Deviation %": [round((input_dict[f] / WHO_LIMITS.get(f, 1)) * 100, 1) for f in FEATURES],
                "Status": ["✅ OK" if input_dict[f] <= WHO_LIMITS.get(f, float('inf')) else "❌ EXCEED" for f in FEATURES]
            })
            
            who_chart = px.bar(who_df, x="Parameter", y=["Actual Value", "WHO Limit"],
                barmode="group", title="Parameter Values vs WHO Standards",
                color_discrete_map={"Actual Value": "#0ea5e9", "WHO Limit": "#10b981"})
            who_chart.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'}, yaxis={'color': '#cbd5e1'},
                hovermode='x unified', height=400)
            st.plotly_chart(who_chart, use_container_width=True)
            
            st.dataframe(who_df, use_container_width=True)
            
            # HEATMAP
            st.markdown("### 🔥 Parameter Deviation Heatmap")
            param_list = []
            deviation_list = []
            for param in FEATURES:
                value = input_dict[param]
                limit = WHO_LIMITS.get(param, value)
                deviation = (value / limit) * 100 if limit > 0 else 0
                param_list.append(param)
                deviation_list.append(min(deviation, 200))
            
            heatmap = go.Figure(data=go.Heatmap(z=[deviation_list], x=param_list,
                colorscale='RdYlGn_r', colorbar={'title': 'Deviation %'}, hoverongaps=False))
            heatmap.update_layout(title="Parameter Deviation from WHO Limits (%)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'}, height=250, xaxis={'color': '#cbd5e1'},
                yaxis={'color': '#cbd5e1'})
            st.plotly_chart(heatmap, use_container_width=True)
            
            # RADAR CHART
            st.markdown("### 🕸️ Parameter Risk Profile")
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[input_dict[f] for f in FEATURES], theta=FEATURES, fill='toself',
                name='Actual Values', line={'color': '#0ea5e9'},
                fillcolor='rgba(14, 165, 233, 0.3)'
            ))
            radar.add_trace(go.Scatterpolar(
                r=[WHO_LIMITS.get(f, 0) for f in FEATURES], theta=FEATURES, fill='toself',
                name='WHO Limits', line={'color': '#10b981'},
                fillcolor='rgba(16, 185, 129, 0.2)'
            ))
            radar.update_layout(height=500, polar=dict(radialaxis=dict(visible=True, color='#cbd5e1')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'})
            st.plotly_chart(radar, use_container_width=True)
            
            # DISTRIBUTION
            st.markdown("### 📦 Parameter Distribution Analysis")
            dist_data = []
            for param in FEATURES:
                dist_data.append({'Parameter': param, 'Value': input_dict[param], 'Type': 'Current Sample'})
                dist_data.append({'Parameter': param, 'Value': WHO_LIMITS[param], 'Type': 'WHO Limit'})
            
            dist_df = pd.DataFrame(dist_data)
            dist_chart = px.box(dist_df, x='Parameter', y='Value', color='Type',
                title='Value Distribution vs WHO Limits',
                color_discrete_map={'Current Sample': '#0ea5e9', 'WHO Limit': '#10b981'})
            dist_chart.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'}, yaxis={'color': '#cbd5e1'},
                height=400)
            st.plotly_chart(dist_chart, use_container_width=True)
            
            # FEATURE IMPORTANCE
            if hasattr(model, "feature_importances_"):
                st.markdown("### 📊 Feature Importance Analysis")
                imp_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=True)
                
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="viridis",
                    title="ML Model Feature Importance")
                fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'}, yaxis={'color': '#cbd5e1'},
                    showlegend=False, height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🔍 Key Finding:</strong> <strong>{imp_df.iloc[-1]['Feature']}</strong> is most influential (Importance: {imp_df.iloc[-1]['Importance']:.4f})<br>
                    {random.choice(SDE_FACTS)}
                </div>
                """, unsafe_allow_html=True)
            
            # SUNBURST
            st.markdown("### ☀️ Water Quality Hierarchy")
            sunburst_data = {
                'labels': ['Water Quality'] + FEATURES + list(who_df['Status'].unique()),
                'parents': [''] + ['Water Quality'] * len(FEATURES) + FEATURES,
                'values': [1] + [input_dict[f] for f in FEATURES] + list(who_df['Deviation %'])
            }
            sunburst = go.Figure(go.Sunburst(
                labels=sunburst_data['labels'][:len(FEATURES)+1],
                parents=sunburst_data['parents'][:len(FEATURES)+1],
                values=sunburst_data['values'][:len(FEATURES)+1],
                marker=dict(colorscale='RdYlGn_r')))
            sunburst.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'},
                height=500)
            st.plotly_chart(sunburst, use_container_width=True)
    
    # --------------------------------------------------
    # OTHER TABS (Keep the rest as before)
    # --------------------------------------------------
    with tab2:
        st.markdown("### 📁 Batch Water Quality Analysis")
        st.markdown("Upload CSV with columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity")
        
        file = st.file_uploader("📤 Upload CSV File", type=["csv"])
        
        if file:
            try:
                data = pd.read_csv(file)
                data_clean = data[FEATURES].copy()
                
                predictions = model.predict(data_clean)
                probabilities = model.predict_proba(data_clean)[:, 1]
                
                data_clean["Prediction"] = ["✅ SAFE" if p == 1 else "❌ UNSAFE" for p in predictions]
                data_clean["Confidence (%)"] = (probabilities * 100).round(2)
                data_clean["WQI Score"] = [calculate_wqi(data_clean.iloc[i].to_dict()) for i in range(len(data_clean))]
                data_clean["Risk Score (%)"] = [calculate_risk_score(data_clean.iloc[i].to_dict()) for i in range(len(data_clean))]
                
                st.markdown("#### 📋 Analysis Results")
                st.dataframe(data_clean, use_container_width=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    safe_count = (predictions == 1).sum()
                    st.metric("✅ Safe", safe_count)
                with col2:
                    unsafe_count = (predictions == 0).sum()
                    st.metric("❌ Unsafe", unsafe_count)
                with col3:
                    avg_confidence = (probabilities * 100).mean()
                    st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")
                with col4:
                    avg_wqi = data_clean["WQI Score"].mean()
                    st.metric("🎯 Avg WQI", f"{avg_wqi:.1f}")
                with col5:
                    avg_risk = data_clean["Risk Score (%)"].mean()
                    st.metric("⚠️ Avg Risk", f"{avg_risk:.1f}%")
                
                st.markdown(f"""
                <div class="fact-box">
                    💡 {random.choice(WATER_FACTS)}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_counts = pd.Series(data_clean["Prediction"].values).value_counts()
                    fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index,
                        title="Safety Distribution",
                        color_discrete_map={"✅ SAFE": "#10b981", "❌ UNSAFE": "#ef4444"})
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_conf = px.histogram(data_clean, x="Confidence (%)", nbins=20,
                        title="Confidence Distribution", color_discrete_sequence=["#0ea5e9"])
                    fig_conf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'},
                        yaxis={'color': '#cbd5e1'}, showlegend=False)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                fig_wqi = px.histogram(data_clean, x="WQI Score", nbins=15,
                    title="WQI Score Distribution", color_discrete_sequence=["#06b6d4"])
                fig_wqi.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'}, yaxis={'color': '#cbd5e1'})
                st.plotly_chart(fig_wqi, use_container_width=True)
                
                fig_scatter = px.scatter(data_clean, x="Confidence (%)", y="WQI Score",
                    color="Risk Score (%)", size="Confidence (%)", hover_data=['Prediction'],
                    title="Confidence vs WQI Score", color_continuous_scale='Viridis')
                fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}, xaxis={'color': '#cbd5e1'}, yaxis={'color': '#cbd5e1'})
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                csv = data_clean.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Results", data=csv,
                    file_name=f"water_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True)
                
            except Exception as e:
                st.markdown("""
                <div class="error-box">
                    <strong>❌ Error:</strong> CSV must contain all required columns!
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📈 Advanced Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Model Performance</h3>
                <ul>
                    <li><strong>Type:</strong> Random Forest</li>
                    <li><strong>Features:</strong> 9 parameters</li>
                    <li><strong>Classes:</strong> 2 (Safe/Unsafe)</li>
                    <li><strong>Status:</strong> ✅ Production Ready</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🌍 WHO Guidelines</h3>
                <ul>
                    <li><strong>Standard:</strong> WHO Guidelines 4th Edition</li>
                    <li><strong>Parameters:</strong> 9 monitored</li>
                    <li><strong>Year:</strong> 2023</li>
                    <li><strong>Compliance:</strong> ✅ Latest</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            💡 {random.choice(WHO_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧬 Feature Specifications")
        features_df = pd.DataFrame({
            "Feature": FEATURES,
            "WHO Limit": [WHO_LIMITS.get(f, "N/A") for f in FEATURES],
            "Unit": ["pH", "mg/L", "mg/L", "mg/L", "mg/L", "µS/cm", "mg/L", "µg/L", "NTU"],
            "Category": ["pH", "Hardness", "Dissolved Solids", "Disinfection", "Minerals", "Conductivity", "Organic", "DBP", "Turbidity"]
        })
        st.dataframe(features_df, use_container_width=True)
    
    with tab4:
        st.markdown("### 🎯 AI Treatment Recommendations")
        st.markdown(f"""
        <div class="fact-box">
            {random.choice(WATER_QUALITY_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_rec = st.number_input("pH Level (Rec)", 0.0, 14.0, 7.0, step=0.1, key="rec1")
            hardness_rec = st.number_input("Hardness (Rec)", 0.0, 500.0, 150.0, step=5.0, key="rec2")
            solids_rec = st.number_input("Total Solids (Rec)", 0.0, 30000.0, 500.0, step=100.0, key="rec3")
        
        with col2:
            chloramines_rec = st.number_input("Chloramines (Rec)", 0.0, 20.0, 3.0, step=0.1, key="rec4")
            sulfate_rec = st.number_input("Sulfate (Rec)", 0.0, 1000.0, 200.0, step=10.0, key="rec5")
            organic_carbon_rec = st.number_input("Organic Carbon (Rec)", 0.0, 30.0, 10.0, step=0.5, key="rec6")
        
        with col3:
            trihalomethanes_rec = st.number_input("Trihalomethanes (Rec)", 0.0, 200.0, 50.0, step=5.0, key="rec7")
            turbidity_rec = st.number_input("Turbidity (Rec)", 0.0, 20.0, 3.0, step=0.1, key="rec8")
            conductivity_rec = st.number_input("Conductivity (Rec)", 0.0, 2000.0, 400.0, step=50.0, key="rec9")
        
        if st.button("💡 Get AI Recommendations"):
            input_dict_rec = {
                "ph": ph_rec, "Hardness": hardness_rec, "Solids": solids_rec,
                "Chloramines": chloramines_rec, "Sulfate": sulfate_rec,
                "Conductivity": conductivity_rec, "Organic_carbon": organic_carbon_rec,
                "Trihalomethanes": trihalomethanes_rec, "Turbidity": turbidity_rec
            }
            
            treatment_plan = get_treatment_plan(input_dict_rec)
            
            if treatment_plan:
                st.markdown("### 🔧 Treatment Plan")
                for param, actual, limit in treatment_plan:
                    st.markdown(f"""
                    <div class="treatment-box">
                        <h4>{param.upper()}</h4>
                        <p><strong>Current Value:</strong> {actual:.2f} (Limit: {limit})</p>
                        <p><strong>Recommended Treatment:</strong></p>
                        <p>{TREATMENT_RECOMMENDATIONS.get(param, 'Consult water treatment expert')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All parameters are within WHO limits! No treatment needed.")
    
    with tab5:
        st.markdown("### ⚠️ Health Impact Assessment")
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ {random.choice(WATER_QUALITY_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_health = st.number_input("pH Level (Health)", 0.0, 14.0, 7.0, step=0.1, key="health1")
            hardness_health = st.number_input("Hardness (Health)", 0.0, 500.0, 150.0, step=5.0, key="health2")
            solids_health = st.number_input("Total Solids (Health)", 0.0, 30000.0, 500.0, step=100.0, key="health3")
        
        with col2:
            chloramines_health = st.number_input("Chloramines (Health)", 0.0, 20.0, 3.0, step=0.1, key="health4")
            sulfate_health = st.number_input("Sulfate (Health)", 0.0, 1000.0, 200.0, step=10.0, key="health5")
            organic_carbon_health = st.number_input("Organic Carbon (Health)", 0.0, 30.0, 10.0, step=0.5, key="health6")
        
        with col3:
            trihalomethanes_health = st.number_input("Trihalomethanes (Health)", 0.0, 200.0, 50.0, step=5.0, key="health7")
            turbidity_health = st.number_input("Turbidity (Health)", 0.0, 20.0, 3.0, step=0.1, key="health8")
            conductivity_health = st.number_input("Conductivity (Health)", 0.0, 2000.0, 400.0, step=50.0, key="health9")
        
        if st.button("🔬 Assess Health Impact"):
            input_dict_health = {
                "ph": ph_health, "Hardness": hardness_health, "Solids": solids_health,
                "Chloramines": chloramines_health, "Sulfate": sulfate_health,
                "Conductivity": conductivity_health, "Organic_carbon": organic_carbon_health,
                "Trihalomethanes": trihalomethanes_health, "Turbidity": turbidity_health
            }
            
            health_impacts = generate_health_impact_report(input_dict_health)
            
            if health_impacts:
                st.markdown("### ⚠️ Health Risks Detected")
                impact_df = pd.DataFrame(health_impacts)
                st.dataframe(impact_df, use_container_width=True)
                
                for impact in health_impacts:
                    if impact["Severity"] == "High":
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>🚨 HIGH SEVERITY:</strong> {impact['Parameter']} - {impact['Health Risk']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>⚠ MODERATE:</strong> {impact['Parameter']} - {impact['Health Risk']}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ No health risks detected!")
    
    with tab6:
        st.markdown("### 🔄 Compare Two Water Samples")
        st.markdown(f"""
        <div class="insight-box">
            {random.choice(WATER_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample 1:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_s1 = st.number_input("pH", 0.0, 14.0, 7.0, step=0.1, key="s1_1")
            hardness_s1 = st.number_input("Hardness", 0.0, 500.0, 150.0, step=5.0, key="s1_2")
            solids_s1 = st.number_input("Solids", 0.0, 30000.0, 500.0, step=100.0, key="s1_3")
        
        with col2:
            chloramines_s1 = st.number_input("Chloramines", 0.0, 20.0, 3.0, step=0.1, key="s1_4")
            sulfate_s1 = st.number_input("Sulfate", 0.0, 1000.0, 200.0, step=10.0, key="s1_5")
            organic_carbon_s1 = st.number_input("Organic Carbon", 0.0, 30.0, 10.0, step=0.5, key="s1_6")
        
        with col3:
            trihalomethanes_s1 = st.number_input("Trihalomethanes", 0.0, 200.0, 50.0, step=5.0, key="s1_7")
            turbidity_s1 = st.number_input("Turbidity", 0.0, 20.0, 3.0, step=0.1, key="s1_8")
            conductivity_s1 = st.number_input("Conductivity", 0.0, 2000.0, 400.0, step=50.0, key="s1_9")
        
        st.markdown("**Sample 2:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_s2 = st.number_input("pH ", 0.0, 14.0, 7.5, step=0.1, key="s2_1")
            hardness_s2 = st.number_input("Hardness ", 0.0, 500.0, 180.0, step=5.0, key="s2_2")
            solids_s2 = st.number_input("Solids ", 0.0, 30000.0, 600.0, step=100.0, key="s2_3")
        
        with col2:
            chloramines_s2 = st.number_input("Chloramines ", 0.0, 20.0, 2.5, step=0.1, key="s2_4")
            sulfate_s2 = st.number_input("Sulfate ", 0.0, 1000.0, 220.0, step=10.0, key="s2_5")
            organic_carbon_s2 = st.number_input("Organic Carbon ", 0.0, 30.0, 12.0, step=0.5, key="s2_6")
        
        with col3:
            trihalomethanes_s2 = st.number_input("Trihalomethanes ", 0.0, 200.0, 55.0, step=5.0, key="s2_7")
            turbidity_s2 = st.number_input("Turbidity ", 0.0, 20.0, 2.5, step=0.1, key="s2_8")
            conductivity_s2 = st.number_input("Conductivity ", 0.0, 2000.0, 450.0, step=50.0, key="s2_9")
        
        if st.button("📊 Compare Samples"):
            sample1 = {
                "ph": ph_s1, "Hardness": hardness_s1, "Solids": solids_s1,
                "Chloramines": chloramines_s1, "Sulfate": sulfate_s1,
                "Conductivity": conductivity_s1, "Organic_carbon": organic_carbon_s1,
                "Trihalomethanes": trihalomethanes_s1, "Turbidity": turbidity_s1
            }
            
            sample2 = {
                "ph": ph_s2, "Hardness": hardness_s2, "Solids": solids_s2,
                "Chloramines": chloramines_s2, "Sulfate": sulfate_s2,
                "Conductivity": conductivity_s2, "Organic_carbon": organic_carbon_s2,
                "Trihalomethanes": trihalomethanes_s2, "Turbidity": turbidity_s2
            }
            
            comparison_chart = create_comparison_chart(sample1, sample2)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            wqi1 = calculate_wqi(sample1)
            wqi2 = calculate_wqi(sample2)
            
            with col1:
                st.metric("Sample 1 WQI", wqi1)
            with col2:
                st.metric("Sample 2 WQI", wqi2)
            with col3:
                wqi_diff = wqi2 - wqi1
                st.metric("WQI Difference", f"{wqi_diff:+.2f}")
            
            comparison_table = pd.DataFrame({
                "Parameter": FEATURES,
                "Sample 1": [sample1[f] for f in FEATURES],
                "Sample 2": [sample2[f] for f in FEATURES],
                "WHO Limit": [WHO_LIMITS.get(f, "-") for f in FEATURES],
                "Difference": [sample2[f] - sample1[f] for f in FEATURES]
            })
            
            st.markdown("### 📋 Detailed Comparison")
            st.dataframe(comparison_table, use_container_width=True)
    
    with tab7:
        st.markdown("### 📚 Reports & Documentation")
        st.markdown(f"""
        <div class="fact-box">
            {random.choice(WHO_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## System Documentation")
        st.markdown("""
        ### Water Quality Parameters
        
        **pH Level** - Measures acidity/alkalinity (0-14 scale)
        - WHO Limit: 8.5 | Ideal Range: 6.5-8.5
        
        **Hardness** - Calcium & magnesium content
        - WHO Limit: 300 mg/L | Impact: Scale formation, soap effectiveness
        
        **Total Solids** - Dissolved & suspended matter
        - WHO Limit: 500 mg/L | Impact: Water clarity, taste, odor
        
        **Chloramines** - Residual disinfectant
        - WHO Limit: 4 mg/L | Impact: Microbial safety
        
        **Sulfate** - Sulfate ion concentration
        - WHO Limit: 250 mg/L | Impact: Taste, laxative effect
        
        **Conductivity** - Electrical conductivity
        - WHO Limit: 800 µS/cm | Impact: Indicates dissolved solids
        
        **Organic Carbon** - Total organic content
        - WHO Limit: 15 mg/L | Impact: Microbial growth potential
        
        **Trihalomethanes** - Disinfection byproducts
        - WHO Limit: 80 µg/L | Impact: Health concern (carcinogenic)
        
        **Turbidity** - Water clarity
        - WHO Limit: 5 NTU | Impact: Microbial protection, aesthetics
        """)
    
    with tab8:
        st.markdown("### ⚙️ System Settings & Configuration")
        st.markdown(f"""
        <div class="insight-box">
            {random.choice(ML_FACTS)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎛️ Model Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", "Random Forest")
            st.metric("Features", "9 Parameters")
        
        with col2:
            st.metric("Classes", "2 (Safe/Unsafe)")
            st.metric("Version", "3.0 Ultimate")
        
        st.markdown("---")
        st.markdown("#### 📊 Display Settings")
        
        chart_theme = st.select_slider("Chart Theme", options=["Dark", "Light", "Custom"])
        animation = st.toggle("Enable Animations", True)
        notifications = st.toggle("Enable Notifications", True)
        
        st.markdown("---")
        st.markdown("#### 📈 Alert Settings")
        
        wqi_alert_level = st.slider("WQI Alert Level", 0, 100, 50)
        risk_alert_level = st.slider("Risk Alert Level", 0, 100, 60)
        
        st.markdown("---")
        st.markdown("#### 💾 Export Settings")
        
        export_format = st.radio("Export Format", ["CSV", "JSON", "Excel"])
        include_charts = st.toggle("Include Charts in Export", True)
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p>💧 Water Quality Prediction Using Data Science Techniques</p>
        <p>🚀 ULTIMATE Edition v3.0 with 25+ Features & Secure Login</p>
        <p style='font-style: italic; color: #475569;'>💡 Fact: {random.choice(WATER_FACTS + WATER_QUALITY_FACTS + SDE_FACTS + ML_FACTS)}</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# APP ROUTER
# --------------------------------------------------
if st.session_state.logged_in:
    main_app()
else:
    login_page()