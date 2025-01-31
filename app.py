import streamlit as st
import torch
import spacy
import numpy as np
from gensim.models import Word2Vec
import joblib
import os

# Load spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load('en_core_web_sm')

# Load all components
@st.cache_resource
def load_model_components(model_dir='model_files'):
    # Load model configuration
    model_config = joblib.load(os.path.join(model_dir, 'model_config.joblib'))
    
    # Initialize model with saved configuration
    model = ImprovedTextClassifier(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_classes=model_config['num_classes']
    )
    
    # Load model state
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'model.pth'),
        map_location=torch.device('cpu'),
        weights_only=True
    ))
    model.eval()
    
    # Load Word2Vec model
    word2vec = Word2Vec.load(os.path.join(model_dir, 'word2vec.model'))
    
    # Load label encoder
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    
    return model, word2vec, label_encoder

class ImprovedTextClassifier(torch.nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_classes=2):
        super(ImprovedTextClassifier, self).__init__()
        
        # Add dropout for regularization
        self.dropout = torch.nn.Dropout(0.3)
        
        # Bi-directional LSTM layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softmax(dim=1)
        )
        
        # Output layers with batch normalization
        self.bn = torch.nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, lengths):
        # Apply initial dropout
        x = self.dropout(x)
        
        # Pack padded sequence for LSTM
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention
        attention_weights = self.attention(output)
        context = torch.sum(attention_weights * output, dim=1)
        
        # Apply batch normalization and final layers
        context = self.bn(context)
        x = torch.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def predict_text(text, model, word2vec_model, nlp, label_encoder):
    # Process text with spaCy
    doc = nlp(text)
    token_list = [token.text for token in doc if not token.is_space]
    
    # Vectorize each word
    vectors = []
    for token in token_list:
        if token in word2vec_model.wv:
            vectors.append(word2vec_model.wv[token])
        else:
            vectors.append(word2vec_model.wv.vectors.mean(axis=0))
    
    if not vectors:
        vectors.append(np.zeros(word2vec_model.vector_size))
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(vectors).unsqueeze(0)
    sequence_length = [len(vectors)]
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor, sequence_length)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_genre = label_encoder.inverse_transform(predicted.numpy())
        confidence_score = confidence.item()
    
    return predicted_genre[0], confidence_score

# Main Streamlit app
st.title('Story Genre Predictor')
st.write('Enter a story description to predict if it\'s a Horror or Romance story!')

# Load components
try:
    model, word2vec_model, label_encoder = load_model_components()
    nlp = load_spacy()
    
    # Text input
    text_input = st.text_area("Enter your story description:", height=200)
    
    if st.button('Predict Genre'):
        if text_input.strip():
            # Make prediction
            genre, confidence = predict_text(
                text_input, model, word2vec_model, nlp, label_encoder
            )
            
            # Display results
            st.write('---')
            st.write(f'### Predicted Genre: {genre}')
            st.write(f'Confidence: {confidence:.2%}')
            
            # Add color coding based on confidence
            if confidence > 0.8:
                st.success('High confidence prediction!')
            elif confidence > 0.6:
                st.warning('Moderate confidence prediction')
            else:
                st.error('Low confidence prediction')
        else:
            st.error('Please enter some text to analyze!')
            
except Exception as e:
    st.error(f'Error loading model components: {str(e)}')
    st.write('Please ensure all model files are present in the model_files directory.')