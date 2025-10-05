🌟 Urdu-Roman Neural Machine Translation (NMT)
An AI-powered web application for translating Urdu text to Roman Urdu using a hybrid approach combining Neural Machine Translation with rule-based systems.
🚀 Features

Neural Machine Translation: Bidirectional LSTM encoder-decoder architecture
Rule-Based Fallback: Comprehensive word and character mapping for improved accuracy
Modern UI: Beautiful gradient-based interface built with Streamlit
Real-time Translation: Instant conversion from Urdu to Roman script
Example Phrases: Quick-access sidebar with common Urdu phrases
Emoji Filtering: Automatic cleanup of emojis and special characters

🏗️ Architecture
Model Components

Encoder: 2-layer Bidirectional LSTM with 128-dimensional embeddings
Decoder: 4-layer LSTM with attention mechanism
Vocabulary: 4,000 tokens each for Urdu and Roman scripts
Hidden Units: 20-dimensional hidden states

Translation Pipeline

Input preprocessing and tokenization
Neural model inference
Rule-based post-processing
Character-level fallback for unknown words

📋 Prerequisites
bashPython 3.8+
PyTorch
Streamlit
🔧 Installation
bash# Clone the repository
git clone [https://github.com/yourusername/urdu-roman-nmt.git](https://github.com/Umaimtahir/Urdu-Roman-Neural-Machine-Translation-NMT-)
cd urdu-roman-nmt

# Install dependencies
pip install -r requirements.txt
📦 Required Files
Place these files in your project directory:

nmt_model.pth - Trained model weights
model_config.pkl - Model configuration
deploy.py - Main application file

🎯 Usage
bashstreamlit run deploy.py
Then open your browser to http://localhost:8501
🛠️ Technical Details

Framework: PyTorch, Streamlit
Model Type: Sequence-to-Sequence with LSTM
Training: Bidirectional encoding with teacher forcing
Inference: Greedy decoding with maximum length of 50 tokens

🎨 UI Features

Gradient backgrounds with modern color schemes
Responsive two-column layout
Interactive sidebar with example phrases
Real-time translation status
Clean and minimalist design

📝 Model Training
The model was trained on Urdu-Roman parallel corpus with:

Embedding dimension: 128
Hidden dimension: 20
Dropout: 0.1
Batch processing with packed sequences

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Your Name - Your GitHub Profile
🙏 Acknowledgments

Urdu linguistic resources
PyTorch community
Streamlit framework
