#!/usr/bin/env python3
"""
Optimized AI Chatbot Evaluation Flask API - BERT Memory Fix
Supports: ethical_alignment, inclusivity, complexity, sentiment
"""

import os
import hashlib
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# REMOVED: transformers and torch imports to save memory
import traceback
from flask import Flask, request, jsonify
import logging
import re

# Disable transformers warnings for faster loading
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Global Variables for Caching
# ------------------------------
_cmu_dict = None
_emotion_model = None
_ethical_alignment_cache = {}
_tokenizer_cache = {}

# ------------------------------
# NLTK Setup (Optimized)
# ------------------------------
def setup_nltk():
    """Optimized NLTK setup with minimal downloads"""
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    # Only download what we absolutely need
    required = ['punkt', 'cmudict']
    for resource in required:
        try:
            if resource == 'cmudict':
                nltk.data.find(f'corpora/{resource}')
            else:
                nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)
            except:
                pass

# Initialize NLTK once at startup
setup_nltk()

# ------------------------------
# LIGHTWEIGHT Ethical Alignment Scorer (BERT Replacement)
# ------------------------------
class LightweightEthicalAlignmentScorer:
    def __init__(self):
        """Initialize lightweight scorer without heavy models"""
        # Ethical alignment keywords and patterns
        self.ethical_positive_terms = {
            'respectful', 'supportive', 'understanding', 'empathetic', 'helpful',
            'professional', 'compassionate', 'safe', 'confidential', 'appropriate',
            'therapeutic', 'healing', 'wellness', 'recovery', 'growth', 'strength',
            'resilience', 'hope', 'positive', 'healthy', 'beneficial', 'care',
            'listen', 'validate', 'support', 'help', 'therapy', 'counseling',
            'treatment', 'resources', 'emotions', 'valid', 'normal', 'difficult',
            'challenging', 'important', 'understand'
        }
        
        self.ethical_negative_terms = {
            'crazy', 'insane', 'nuts', 'psycho', 'weird', 'abnormal', 'wrong',
            'stupid', 'ridiculous', 'overreacting', 'dramatic', 'attention',
            'manipulative', 'toxic', 'harmful', 'dangerous', 'inappropriate'
        }
        
        # Professional patterns
        self.professional_patterns = [
            r'\b(I understand|I hear you|I recognize)\b',
            r'\b(therapy|counseling|treatment|support)\b',
            r'\b(professional help|mental health|wellbeing)\b',
            r'\b(safe space|confidential|privacy)\b',
            r'\b(how are you feeling|what would help)\b'
        ]
    
    def ethical_alignment_score(self, text):
        """Calculate rule-based ethical alignment score"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in _ethical_alignment_cache:
            return _ethical_alignment_cache[text_hash]
        
        if not text.strip():
            result = 0.0
        else:
            try:
                # Tokenize text
                words = nltk.word_tokenize(text.lower()) if nltk else text.lower().split()
                
                # Count positive and negative terms
                positive_count = sum(1 for word in words if word in self.ethical_positive_terms)
                negative_count = sum(1 for word in words if word in self.ethical_negative_terms)
                
                # Check for professional patterns
                pattern_score = 0
                for pattern in self.professional_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        pattern_score += 1
                
                # Calculate base score
                word_count = len(words)
                if word_count == 0:
                    result = 0.0
                else:
                    # Weighted scoring
                    positive_weight = positive_count * 2.0
                    negative_weight = negative_count * 1.0
                    pattern_weight = pattern_score * 3.0
                    
                    # Base ethical score
                    base_score = (positive_weight - negative_weight + pattern_weight) / word_count
                    
                    # Length normalization (longer responses get slight bonus)
                    length_factor = min(1.2, 1.0 + (word_count / 100))
                    
                    # Final score with bounds
                    result = max(0.0, min(1.0, base_score * length_factor + 0.3))
                
            except Exception as e:
                print(f"Ethical scoring error: {e}")
                result = 0.0
        
        # Cache result
        if len(_ethical_alignment_cache) < 100:
            _ethical_alignment_cache[text_hash] = result
        
        return round(result, 2)

# Removed get_ethical_scorer function - using direct instantiation

# ------------------------------
# Model Initialization (Lazy Loading) - UNCHANGED
# ------------------------------
def get_emotion_model():
    """Lazy load emotion model only when needed"""
    global _emotion_model
    if _emotion_model is None:
        try:
            from transformers import pipeline
            _emotion_model = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                top_k=None,
                device=-1,  # Force CPU to avoid GPU overhead
                model_kwargs={"torch_dtype": "auto"}
            )
        except Exception as e:
            print(f"Warning: Could not load emotion model: {e}")
            _emotion_model = False  # Mark as failed to avoid retrying
    return _emotion_model if _emotion_model is not False else None

def get_cmu_dict():
    """Cached CMU dictionary loading"""
    global _cmu_dict
    if _cmu_dict is None:
        try:
            _cmu_dict = nltk.corpus.cmudict.dict()
        except:
            _cmu_dict = {}
    return _cmu_dict


# ------------------------------
# Constants (Same as before but organized for faster access) - UNCHANGED
# ------------------------------
LGBTQ_AFFIRMING_TERMS = frozenset([
    'sexual orientation', 'gender identity', 'lgbtq', 'transgender', 'non-binary',
    'gender nonconforming', 'coming out', 'transition', 'affirming', 'identity acceptance',
    'discrimination', 'microaggressions', 'minority stress', 'internalized', 'authentic self',
    'chosen family', 'community', 'belonging', 'pride', 'visibility'
])

SOCIAL_WORK_PROFESSIONAL_TERMS = frozenset([
    'strengths-based', 'person-centered', 'trauma-informed', 'culturally competent',
    'self-determination', 'empowerment', 'advocacy', 'social justice', 'systemic',
    'intersectionality', 'resilience', 'protective factors', 'risk factors',
    'assessment', 'intervention', 'case management', 'referral', 'collaboration'
])

CRISIS_ASSESSMENT_TERMS = frozenset([
    'suicidal', 'suicide', 'self-harm', 'harm', 'hurt', 'safety', 'plan', 'means',
    'access', 'intent', 'attempt', 'thoughts', 'feelings', 'crisis', 'emergency',
    'immediate', 'urgent', 'risk', 'protective', 'coping'
])

SUPPORTIVE_TERMS = frozenset([
    'support', 'help', 'understand', 'listen', 'care', 'confidential',
    'therapy', 'counseling', 'treatment', 'resources', 'professional',
    'emotions', 'valid', 'normal', 'difficult', 'challenging', 'important'
])

ETHICAL_NEGATIVE_TERMS = frozenset([
    'crazy', 'insane', 'nuts', 'psycho', 'weird', 'abnormal', 'wrong',
    'stupid', 'ridiculous', 'overreacting', 'dramatic', 'attention'
])

CORE_TERMS = frozenset([
    'gender identity', 'sexual orientation', 'lgbtq+', 'identity acceptance',
    'safe space', 'allyship', 'inclusive language', 'authentic self'
])

SECONDARY_TERMS = frozenset([
    'resilience', 'culturally appropriate', 'psychological safety',
    'connected to community', 'trusted person', 'inclusive provider'
])

GENERAL_INCLUSIVE_TERMS = frozenset([
    'inclusive', 'diverse', 'equitable', 'accessibility', 'non-binary',
    'gender nonconforming', 'lgbtq+ support', 'discrimination',
    'affirmation', 'gender-affirming', 'support system', 'self-worth', 
    'healing-centered', 'mental health advocate', 'compassionate support'
])

PENALTY_TERMS = frozenset(['crazy', 'normal', 'weak', 'abnormal', 'insane', 'burden', 'failure'])
SEVERE_PENALTY_TERMS = frozenset(['psychotic', 'schizo', 'delusional', 'mental case'])

# Combined sets for faster lookup
INCLUSIVITY_LEXICON = CORE_TERMS | SECONDARY_TERMS | GENERAL_INCLUSIVE_TERMS
ALL_PENALTY_TERMS = PENALTY_TERMS | SEVERE_PENALTY_TERMS

# Pre-compiled question patterns for faster matching
QUESTION_PATTERNS = [
    'how often', 'tell me about', 'describe', 'what has been', 
    'have you experienced', 'how do you feel', 'what would help', 
    'who in your life', 'what support'
]

RELEVANT_EMOTIONS = [
    'empathy', 'compassion', 'validation', 'understanding', 'trust', 'support',
    'safety', 'reassurance', 'joy', 'love', 'optimism', 'hope', 'relief', 'calm',
    'gratitude', 'caring', 'confident', 'sadness', 'fear', 'anxiety', 'anger',
    'shame', 'guilt', 'loneliness', 'isolation', 'confusion', 'neutral',
    'surprise', 'curiosity'
]

EMOTION_WEIGHTS = {
    'empathy': 2.5, 'compassion': 2.5, 'validation': 2.2, 'understanding': 2.0,
    'trust': 2.0, 'support': 1.8, 'safety': 1.8, 'reassurance': 1.6,
    'joy': 1.4, 'love': 1.6, 'optimism': 1.5, 'hope': 1.6,
    'relief': 1.3, 'calm': 1.2, 'gratitude': 1.2, 'caring': 1.5, 'confident': 1.3,
    'sadness': 0.9, 'fear': 0.8, 'anxiety': 0.8, 'anger': 0.6, 'shame': 0.5,
    'guilt': 0.5, 'loneliness': 0.6, 'isolation': 0.6, 'confusion': 0.6,
    'neutral': 0.4, 'surprise': 0.5, 'curiosity': 0.6
}
READABILITY_CONSTANTS = {
    'READABILITY_FK_CONSTANT': 206.835,
    'READABILITY_FK_SENTENCE_WEIGHT': 1.1,
    'READABILITY_FK_SYLLABLE_WEIGHT': 70.0,
    'SENTENCE_COMPLEXITY_WEIGHT': 1.2
}

# ------------------------------
# Optimized Helper Functions - UNCHANGED
# ------------------------------
def fast_tokenize(text, cache_key=None):
    """Fast tokenization with caching"""
    if cache_key and cache_key in _tokenizer_cache:
        return _tokenizer_cache[cache_key]
    
    try:
        tokens = nltk.word_tokenize(text.lower())
    except:
        tokens = text.lower().split()
    
    if cache_key and len(_tokenizer_cache) < 1000:  # Limit cache size
        _tokenizer_cache[cache_key] = tokens
    
    return tokens

def count_syllables(word):
    """Optimized syllable counting"""
    cmudict = get_cmu_dict()
    phonemes_list = cmudict.get(word.lower(), [[0]])
    return max(1, sum(1 for phoneme in phonemes_list[0] if isinstance(phoneme, str) and phoneme[-1].isdigit()))

# ------------------------------
# Evaluation Functions - ONLY ETHICAL ALIGNMENT CHANGED
# ------------------------------
def evaluate_ethical_alignment(generated_text):
    """Lightweight ethical alignment evaluation (NO BERT)"""
    scorer = LightweightEthicalAlignmentScorer()
    return scorer.ethical_alignment_score(generated_text)

# ALL OTHER FUNCTIONS REMAIN EXACTLY THE SAME
def evaluate_inclusivity_score(generated_text):
    """Optimized inclusivity evaluation"""
    words = fast_tokenize(generated_text)
    if not words:
        return 0.0

    # Optimized counting using set operations
    word_set = set(words)
    
    # Count inclusive terms with weights
    inclusive_count = 0
    for word in word_set:
        if word in CORE_TERMS:
            inclusive_count += words.count(word) * 4
        elif word in SECONDARY_TERMS:
            inclusive_count += words.count(word) * 2.5
        elif word in GENERAL_INCLUSIVE_TERMS:
            inclusive_count += words.count(word) * 2

    # Count penalty terms
    penalty_count = 0
    for word in word_set:
        if word in SEVERE_PENALTY_TERMS:
            penalty_count += words.count(word) * 1.0
        elif word in PENALTY_TERMS:
            penalty_count += words.count(word) * 0.5

    # Calculate score
    total_words = len(words)
    score = min(1.0,max(0, (inclusive_count - penalty_count) / total_words + inclusive_count / 15)) if total_words > 0 else 0
    return float(round(score, 2))

def evaluate_complexity_score(text):
    sentences = nltk.sent_tokenize(text)
    words = [w for s in sentences for w in nltk.word_tokenize(s)]
    total_words = len(words)
    total_sentences = len(sentences) if sentences else 1
    syllables = sum(count_syllables(w) for w in words)
    avg_sentence_length = total_words / total_sentences
    avg_syllables_per_word = syllables / total_words
    fk_score = 206.835 - 1.015*avg_sentence_length - 84.6*avg_syllables_per_word
    return float(round(fk_score, 2))

def evaluate_sentiment_distribution(reference_text, generated_text):
    """Optimized sentiment evaluation"""
    def get_vector_fast(text):
        emotion_model = get_emotion_model()
        if emotion_model is None:
            # Fast fallback
            words = fast_tokenize(text)
            word_set = set(words)
            scores = {}
            for emotion in RELEVANT_EMOTIONS:
                scores[emotion] = len([w for w in word_set if emotion in w]) / len(words) if words else 0
        else:
            try:
                # Truncate long texts for faster processing
                if len(text) > 500:
                    text = text[:500]
                raw = emotion_model(text)[0]
                scores = {e['label'].lower(): e['score'] for e in raw}
            except:
                words = fast_tokenize(text)
                word_set = set(words)
                scores = {}
                for emotion in RELEVANT_EMOTIONS:
                    scores[emotion] = len([w for w in word_set if emotion in w]) / len(words) if words else 0
        
        return np.array([
            scores.get(emotion, 0.0) * EMOTION_WEIGHTS.get(emotion, 1.0)
            for emotion in RELEVANT_EMOTIONS
        ]).reshape(1, -1)

    try:
        ref_vec = get_vector_fast(reference_text)
        gen_vec = get_vector_fast(generated_text)
        similarity = cosine_similarity(ref_vec, gen_vec)[0][0]
        if np.isnan(similarity) or np.isinf(similarity):
            similarity = 0.0
    except:
        similarity = 0.0
    
    return round(float(similarity), 2)

# ------------------------------
# Main Evaluation Function - UNCHANGED
# ------------------------------
def evaluate_chatbot_response(formula_name, chatbot_text, human_text=""):
    """Fast evaluation dispatcher"""
    formula_name = formula_name.strip().lower()
    if not chatbot_text.strip():
        return 0.0
    
    try:
        if formula_name == "ethical_alignment":
            return evaluate_ethical_alignment(chatbot_text)
        elif formula_name == "inclusivity":
            return evaluate_inclusivity_score(chatbot_text)
        elif formula_name == "complexity":
            return evaluate_complexity_score(chatbot_text)
        elif formula_name == "sentiment":
            if not human_text.strip():
                raise ValueError("Sentiment requires human_text")
            return evaluate_sentiment_distribution(human_text, chatbot_text)
        else:
            raise ValueError(f"Unknown formula: {formula_name}")
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

def evaluate_all_fast(chatbot_text, human_text=""):
    results = {
        "ethical_alignment": float(evaluate_ethical_alignment(chatbot_text)),
        "inclusivity": float(evaluate_inclusivity_score(chatbot_text)),
        "complexity": float(evaluate_complexity_score(chatbot_text)),
    }
    
    if human_text:
        results["sentiment"] = float(evaluate_sentiment_distribution(human_text, chatbot_text))
    else:
        results["sentiment"] = None
    
    return results


# ------------------------------
# Flask API (Optimized) - UNCHANGED
# ------------------------------
app = Flask(__name__)

# Disable Flask debug logging for better performance
if not app.debug:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.route("/evaluate", methods=["POST"])
def api_evaluate():
    """
    Optimized API endpoint
    POST JSON: {
        "formula": "ethical_alignment" or "all",
        "chatbot_text": "...",
        "human_text": "..." (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        formula = data.get("formula", "").strip().lower()
        chatbot_text = data.get("chatbot_text", "").strip()
        human_text = data.get("human_text", "").strip()

        if not chatbot_text:
            return jsonify({"error": "chatbot_text required"}), 400

        if formula == "all":
            scores = evaluate_all_fast(chatbot_text, human_text)
            formatted_scores = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in scores.items()}
            return jsonify(formatted_scores)
        else:
            score = evaluate_chatbot_response(formula, chatbot_text, human_text)
            return jsonify({formula: round(score, 2)})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Quick health check endpoint"""
    return jsonify({"status": "healthy"}), 200

# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    
    # Production-optimized settings
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False,
        threaded=True,
        use_reloader=False
    )