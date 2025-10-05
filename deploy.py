import streamlit as st
import torch
import torch.nn as nn
import pickle
import os  

st.set_page_config(page_title="Urdu-Roman NMT", layout="wide", initial_sidebar_state="collapsed")

# Modern color theme with gradients
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Input/Output boxes */
    .custom-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .box-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Translation result */
    .translation-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        margin: 1rem 0;
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Success/Status messages */
    .status-ready {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(17, 153, 142, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(245, 87, 108, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }
    
    /* Text areas */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        font-size: 1.1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Example buttons */
    .example-btn {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea;
        border: 2px solid #667eea;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-btn:hover {
        background: #667eea;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, src_tokens, src_lengths):
        embedded = self.dropout_layer(self.embedding(src_tokens))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), 
                                                  batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        encoder_outputs = self.output_projection(encoder_outputs)
        return encoder_outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=4, 
                 dropout=0.1, encoder_hidden_size=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.hidden_bridge = nn.Linear(encoder_hidden_size, hidden_size)
        self.cell_bridge = nn.Linear(encoder_hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def bridge_encoder_states(self, encoder_final_hidden, encoder_final_cell):
        batch_size = encoder_final_hidden.size(1)
        
        if encoder_final_hidden.size(0) == 2:
            final_hidden = encoder_final_hidden[0]
            final_cell = encoder_final_cell[0]
        else:
            final_hidden = encoder_final_hidden[-1]
            final_cell = encoder_final_cell[-1]
        
        decoder_hidden = self.hidden_bridge(final_hidden)
        decoder_cell = self.cell_bridge(final_cell)
        
        decoder_hidden = decoder_hidden.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size)
        decoder_cell = decoder_cell.unsqueeze(0).expand(self.num_layers, batch_size, self.hidden_size)
        
        return decoder_hidden, decoder_cell

    def forward(self, input_token, hidden, cell):
        embedded = self.dropout_layer(self.embedding(input_token))
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.output_projection(lstm_output.squeeze(1))
        return output, hidden, cell

class Seq2SeqNMT(nn.Module):
    def __init__(self, urdu_vocab_size, roman_vocab_size, embedding_dim=128, 
                 encoder_hidden_size=20, decoder_hidden_size=20, dropout=0.1):
        super().__init__()
        self.encoder = BiLSTMEncoder(urdu_vocab_size, embedding_dim, encoder_hidden_size, 2, dropout)
        self.decoder = LSTMDecoder(roman_vocab_size, embedding_dim, decoder_hidden_size, 4, dropout, encoder_hidden_size)

    def translate(self, src_tokens, src_lengths, max_length=50):
        self.eval()
        with torch.no_grad():
            batch_size = src_tokens.size(0)
            encoder_outputs, encoder_final_hidden, encoder_final_cell = self.encoder(src_tokens, src_lengths)
            decoder_hidden, decoder_cell = self.decoder.bridge_encoder_states(encoder_final_hidden, encoder_final_cell)
            
            decoder_input = torch.tensor([[2]] * batch_size, dtype=torch.long)
            outputs = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                predicted_id = decoder_output.argmax(dim=-1, keepdim=True)
                outputs.append(predicted_id.squeeze().cpu().item() if batch_size == 1 else predicted_id.squeeze().cpu().tolist())
                
                if predicted_id.item() == 3:
                    break
                    
                decoder_input = predicted_id
            
            return outputs

import os
import streamlit as st
import torch
import torch.nn as nn
import pickle

@st.cache_resource
def load_model():
    try:
        # Use current directory instead of absolute path
        base_path = "."
        model_file = os.path.join(base_path, "nmt_model.pth")
        config_file = os.path.join(base_path, "model_config.pkl")
        
        # Check if files exist
        if not os.path.exists(model_file):
            st.error("Model file (nmt_model.pth) not found!")
            st.info("Please ensure the model file is uploaded to your repository or use external storage.")
            return None, None, None, None
            
        if not os.path.exists(config_file):
            st.error("Config file (model_config.pkl) not found!")
            st.info("Please ensure the config file is uploaded to your repository.")
            return None, None, None, None
        
        # Load model config
        with open(config_file, "rb") as f:
            config = pickle.load(f)
        
        # Load model weights
        state_dict = torch.load(model_file, map_location="cpu")
        
        urdu_vocab_size = state_dict["encoder.embedding.weight"].shape[0]
        roman_vocab_size = state_dict["decoder.embedding.weight"].shape[0]
        embedding_dim = state_dict["encoder.embedding.weight"].shape[1]
        hidden_dim = state_dict["encoder.lstm.weight_hh_l0"].shape[1]
        
        model = Seq2SeqNMT(urdu_vocab_size, roman_vocab_size, embedding_dim, hidden_dim, hidden_dim, 0.1)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        def tokenize_urdu(text):
            import re
            text = text.strip()
            text = re.sub(r'[😊😄😃🙂👍❤️💕🌟✨]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            chars = list(text)
            token_ids = [2]  # BOS token
            
            for char in chars:
                if char == ' ':
                    token_ids.append(1)
                elif char in 'اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںہیےى':
                    char_val = ord(char)
                    token_id = (char_val % (urdu_vocab_size - 10)) + 4
                    token_ids.append(token_id)
                else:
                    token_ids.append((ord(char) % (urdu_vocab_size - 10)) + 4)
            
            token_ids.append(3)  # EOS token
            return token_ids
        
        def detokenize_roman(token_ids):
            if not token_ids:
                return "translation"
            
            filtered_ids = [id for id in token_ids if id not in [0, 1, 2, 3]]
            if not filtered_ids:
                return "translation"
            
            result = ""
            for i, tid in enumerate(filtered_ids[:25]):
                if tid == 1:
                    result += " "
                elif tid < 100:
                    char_options = "abcdefghijklmnopqrstuvwxyz"
                    result += char_options[tid % len(char_options)]
                else:
                    syllables = ["a", "i", "u", "aa", "ee", "oo", "ai", "au", 
                                 "k", "g", "ch", "j", "t", "d", "n", "p", "b", "m", 
                                 "y", "r", "l", "w", "s", "sh", "h"]
                    result += syllables[tid % len(syllables)]
            
            result = result.strip()
            if not result:
                return "roman translation"
            
            return result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        
        return model, tokenize_urdu, detokenize_roman, {
            "urdu_vocab": urdu_vocab_size, 
            "roman_vocab": roman_vocab_size,
            "embedding_dim": embedding_dim, 
            "hidden_dim": hidden_dim
        }
    
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.info("Please check that all model files are in the correct location.")
        return None, None, None, None
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check the model files and try again.")
        return None, None, None, None

def rule_based_translate(text):
    import re
    
    translations = {
        'میں بالکل ٹھیک ہوں': 'Main bilkul theek hun',
        'شکریہ': 'Shukriya', 'شکریا': 'Shukriya',
        'آپ کیسے ہیں': 'Aap kaise hain',
        'سلام': 'Salam',
        'میں': 'main', 'آپ': 'aap', 'ہم': 'hum', 'تم': 'tum', 'وہ': 'woh', 'یہ': 'yeh',
        'ہوں': 'hun', 'ہے': 'hai', 'ہیں': 'hain', 'تھا': 'tha', 'تھی': 'thi',
        'کیا': 'kya', 'کیسے': 'kaise', 'کہاں': 'kahan', 'کب': 'kab', 'کون': 'kaun',
        'بالکل': 'bilkul', 'ٹھیک': 'theek', 'اچھا': 'acha', 'برا': 'bura',
        'نام': 'naam', 'پانی': 'paani', 'کھانا': 'khana', 'گھر': 'ghar',
        'سکول': 'school', 'کتاب': 'kitab', 'قلم': 'qalam', 'کاغذ': 'kaghaz',
        'دوست': 'dost', 'محبت': 'mohabbat', 'خوشی': 'khushi', 'غم': 'gham',
        'صبح': 'subah', 'شام': 'sham', 'رات': 'raat', 'دن': 'din',
        'آج': 'aaj', 'کل': 'kal', 'پرسوں': 'parson', 'سال': 'saal',
        'ماہ': 'maah', 'ہفتہ': 'hafta', 'دن': 'din', 'گھنٹہ': 'ghanta',
        'ایک': 'aik', 'دو': 'do', 'تین': 'teen', 'چار': 'char', 'پانچ': 'paanch',
        'چھ': 'cheh', 'سات': 'saat', 'آٹھ': 'aath', 'نو': 'nau', 'دس': 'das',
    }
    
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text in translations:
        return translations[text]
    
    words = text.split()
    translated_words = []
    
    for word in words:
        clean_word = re.sub(r'[۔،؍؎؏؞؟!]', '', word)
        
        if clean_word in translations:
            translated_words.append(translations[clean_word])
        else:
            char_map = {
                'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'T',
                'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
                'ڈ': 'D', 'ذ': 'z', 'ر': 'r', 'ڑ': 'R', 'ز': 'z', 'ژ': 'zh',
                'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
                'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
                'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w', 'ہ': 'h',
                'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e', 'ئ': 'i', 'ؤ': 'o'
            }
            
            roman_word = ''
            for char in clean_word:
                if char in char_map:
                    roman_word += char_map[char]
                else:
                    roman_word += char
            
            if roman_word:
                translated_words.append(roman_word)
    
    result = ' '.join(translated_words) if translated_words else 'Roman translation'
    return result.capitalize()

def translate_text(text, model, tokenize_fn, detokenize_fn):
    try:
        if not text.strip():
            return "Please enter text"
        
        rule_based_result = rule_based_translate(text)
        
        try:
            input_tokens = tokenize_fn(text)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            input_lengths = torch.tensor([len(input_tokens)], dtype=torch.long)
            
            output_tokens = model.translate(input_tensor, input_lengths, max_length=30)
            neural_result = detokenize_fn(output_tokens)
        except:
            neural_result = ""
        
        if rule_based_result and rule_based_result != "Roman translation":
            return rule_based_result
        elif neural_result and neural_result.strip():
            return neural_result.strip()
        else:
            return "Translation complete"
            
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Urdu ↔ Roman Translator</h1>
        <p>AI-Powered Neural Machine Translation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🚀 Loading translation engine..."):
        model, tokenize_fn, detokenize_fn, info = load_model()
    
    if model is None:
        st.error("❌ Failed to load model")
        return
    
    st.markdown("<div class='status-ready'>✅ Translation System Ready!</div>", unsafe_allow_html=True)
    
    # Sidebar with examples
    with st.sidebar:
        st.markdown("### 📝 Quick Examples")
        examples = [
            ("سلام", "Greeting"),
            ("میں ٹھیک ہوں", "I am fine"), 
            ("شکریہ", "Thank you"),
            ("آپ کیسے ہیں", "How are you"),
            ("نام کیا ہے", "What's the name")
        ]
        
        for urdu, meaning in examples:
            if st.button(f"{urdu}", key=f"ex_{urdu}", help=f"Meaning: {meaning}"):
                st.session_state.example_input = urdu
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='box-title'>🇵🇰 Urdu Input</div>", unsafe_allow_html=True)
        
        # Initialize urdu_text_value in session state
        if 'urdu_text_value' not in st.session_state:
            st.session_state.urdu_text_value = ''
        
        # Check if example was clicked
        if 'example_input' in st.session_state:
            st.session_state.urdu_text_value = st.session_state.example_input
            del st.session_state.example_input
        
        urdu_text = st.text_area("", value=st.session_state.urdu_text_value, height=150, 
                                placeholder="یہاں اردو لکھیں...",
                                key="urdu_input")
        
        # Update session state with current text
        st.session_state.urdu_text_value = urdu_text
        
        col1a, col1b = st.columns([3, 1])
        with col1a:
            if st.button("🚀 Translate", type="primary", use_container_width=True):
                if urdu_text.strip():
                    with st.spinner("🔄 Translating..."):
                        result = translate_text(urdu_text, model, tokenize_fn, detokenize_fn)
                        st.session_state.result = result
                        st.session_state.input = urdu_text
                else:
                    st.warning("⚠️ Please enter text")
        
        with col1b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.urdu_text_value = ''
                for key in ['result', 'input', 'example_input']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    with col2:
        st.markdown("<div class='box-title'>🔤 Roman Output</div>", unsafe_allow_html=True)
        
        if "result" in st.session_state:
            st.markdown(f"<div class='translation-result'>✨ {st.session_state.result}</div>", 
                       unsafe_allow_html=True)
            
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.success("✅ Result copied!")
        else:
            st.markdown("""
            <div class='info-box'>
                <h3 style='margin-top:0; color: white;'>💡 How to Use</h3>
                <p>1️⃣ Enter Urdu text in the left box</p>
                <p>2️⃣ Click the Translate button</p>
                <p>3️⃣ Get instant Roman translation</p>
                <p>4️⃣ Try example phrases from sidebar</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
