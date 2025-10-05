# Urdu-Roman-Neural-Machine-Translation-NMT-
An AI-powered web application that translates Urdu text to Roman Urdu using a hybrid neural-rule-based approach. Built with PyTorch and Streamlit, featuring a modern gradient UI and real-time translation. 
✨ Key Features  
Bidirectional LSTM encoder-decoder architecture
Rule-based fallback for improved accuracy
Modern, responsive web interface
Real-time translation with example phrases
4,000 token vocabulary for both Urdu and Roman scripts

🚀 Quick Start
bashpip install torch streamlit
streamlit run deploy.py
📊 Model Architecture

Encoder: 2-layer BiLSTM (128-dim embeddings)
Decoder: 4-layer LSTM (20-dim hidden states)
Hybrid: Neural + rule-based translation

💻 Tech Stack
PyTorch | Streamlit | Python 3.8+
