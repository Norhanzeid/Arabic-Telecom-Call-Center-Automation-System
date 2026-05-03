"""
Local Analyzer - Uses Sentence Transformers for Category & Sentiment Classification
No OpenAI API required - runs completely offline
"""

import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model = None


def get_embedding_model():
    """Load Sentence Transformer model (cached)"""
    global _model
    
    if _model is None:
        logger.info("🔄 Loading Sentence Transformers model...")
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("✅ Sentence Transformers model loaded!")
    
    return _model


class LocalAnalyzer:
    """
    Analyzes Arabic call center text using Sentence Transformers
    Classifies category and sentiment without any API calls
    """
    
    def __init__(self):
        self.model = get_embedding_model()
        
        # Category definitions with example phrases
        self.categories = {
            "مشكلة نت": [
                "الإنترنت فصل",
                "النت فصل",
                "مفيش إنترنت",
                "مفيش نت",
                "الإنترنت بطيء",
                "النت بطيء جداً",
                "الإنترنت مش شغال",
                "انقطاع الإنترنت",
                "مشكلة في الإنترنت",
                "الراوتر مش شغال",
                "الواي فاي فصل",
                "سرعة النت ضعيفة",
                "النت واقع",
                "عطل في الإنترنت",
                "الخط فاصل",
                "مفيش اتصال بالنت"
            ],
            "مشكلة فواتير": [
                "الفاتورة غلط",
                "الفاتورة عالية",
                "مبلغ الفاتورة كبير",
                "رسوم إضافية",
                "خصم من الرصيد",
                "مشكلة في الدفع",
                "دفعت الفاتورة ومش متسجلة",
                "فاتورة مكررة",
                "حساب غلط"
            ],
            "طلب خدمة": [
                "عايز أشترك في باقة جديدة",
                "عايز أعرف العروض",
                "ترقية الباقة",
                "تغيير الباقة",
                "اشتراك جديد",
                "استفسار عن الخدمات",
                "عايز أفعل خدمة"
            ]
        }
        
        # Sentiment phrases
        self.sentiment_phrases = {
            "سلبي": [
                "مشكلة كبيرة",
                "زهقت",
                "مستاء جداً",
                "غضبان",
                "الموضوع مش منطقي",
                "محتاج حل فوري",
                "فجأة فصل",
                "انقطع",
                "مش راضي",
                "تعبت من المشكلة",
                "الخدمة وحشة"
            ],
            "إيجابي": [
                "شكراً جزيلاً",
                "خدمة ممتازة",
                "المشكلة اتحلت",
                "راضي",
                "تمام",
                "شكراً على المساعدة"
            ]
        }
        
        # Negative keywords for quick detection
        self.negative_keywords = [
            "فصل", "قطع", "مشكلة", "عطل", "مفيش", "مافيش",
            "بطيء", "تأخير", "بشتكي", "فجأة", "فجاة", "انقطاع",
            "مش شغال", "متوقف", "معطل", "واقع", "ضعيف"
        ]
        
        # Internet problem keywords for quick detection
        self.internet_keywords = [
            "الإنترنت فصل", "النت فصل", "مفيش إنترنت", "مفيش نت",
            "الإنترنت مش شغال", "النت مش شغال", "انقطاع الإنترنت",
            "بطيء", "الراوتر", "واي فاي"
        ]
        
        # Pre-compute embeddings for categories
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all category and sentiment phrases"""
        logger.info("🔄 Pre-computing category embeddings...")
        
        self.category_embeddings = {}
        for category, phrases in self.categories.items():
            embeddings = self.model.encode(phrases)
            self.category_embeddings[category] = embeddings
        
        self.sentiment_embeddings = {}
        for sentiment, phrases in self.sentiment_phrases.items():
            embeddings = self.model.encode(phrases)
            self.sentiment_embeddings[sentiment] = embeddings
        
        logger.info("✅ Embeddings pre-computed!")
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _classify_category(self, text: str) -> str:
        """Classify text into category using semantic similarity"""
        
        # Quick keyword check for internet problems
        if any(keyword in text for keyword in self.internet_keywords):
            return "مشكلة نت"
        
        # Get text embedding
        text_embedding = self.model.encode([text])
        
        best_category = "مشكلة نت"  # Default
        best_score = -1
        
        for category, embeddings in self.category_embeddings.items():
            # Calculate similarity with all phrases in category
            similarities = cosine_similarity(text_embedding, embeddings)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_category = category
        
        logger.info(f"📊 Category: {best_category} (score: {best_score:.3f})")
        return best_category
    
    def _classify_sentiment(self, text: str) -> str:
        """Classify sentiment using semantic similarity and keywords"""
        
        # Quick keyword check
        is_negative = any(keyword in text for keyword in self.negative_keywords)
        
        if is_negative:
            return "سلبي"
        
        # Get text embedding
        text_embedding = self.model.encode([text])
        
        best_sentiment = "سلبي"  # Default for complaints
        best_score = -1
        
        for sentiment, embeddings in self.sentiment_embeddings.items():
            similarities = cosine_similarity(text_embedding, embeddings)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_sentiment = sentiment
        
        return best_sentiment
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze text for category and sentiment
        
        Args:
            text: Arabic text to analyze
        
        Returns:
            Dictionary with category and sentiment labels
        """
        text = self.clean_text(text)
        
        if not text:
            return {
                'category': {'label': 'غير محدد'},
                'sentiment': {'label': 'سلبي'}
            }
        
        # Force internet problem for clear cases
        internet_problem_phrases = [
            "الإنترنت فصل", "النت فصل", "مفيش إنترنت", "مفيش نت",
            "الإنترنت مش شغال", "النت مش شغال", "مشكلة في الإنترنت",
            "انقطاع الإنترنت", "عطل في النت", "الإنترنت بطيء"
        ]
        
        if any(phrase in text for phrase in internet_problem_phrases):
            return {
                'category': {'label': 'مشكلة نت'},
                'sentiment': {'label': 'سلبي'}
            }
        
        category = self._classify_category(text)
        sentiment = self._classify_sentiment(text)
        
        return {
            'category': {'label': category},
            'sentiment': {'label': sentiment}
        }
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze multiple texts"""
        return [self.analyze_text(text) for text in texts]


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing Local Analyzer (Sentence Transformers)")
    print("=" * 60)
    
    analyzer = LocalAnalyzer()
    
    # Test with the actual call center text
    test_text = """مساء الخير يا فندم، شكراً لاتصالك WE للاتصالات. أحمد في خدمتك. 
    الحقيقة أنا عايزة أقول لك أن أنا بالضبط على بعد 30 دقيقة من هتفصل من شغلي. 
    أنا كنت بشتغل والإنترنت شغال بشكل طبيعي، وبعد ما حضرت ملفات الإكسل والإيميلات 
    فجأة الإنترنت فصل وعلامة الإنترنت على جهاز الكمبيوتر بتقول أني مفيش إنترنت. 
    والموضوع مش منطقي خالص، لأني دفعت الفاتورة في معادها."""
    
    result = analyzer.analyze_text(test_text)
    
    print(f"Category: {result['category']['label']}")
    print(f"Sentiment: {result['sentiment']['label']}")
    print("=" * 60)
    
    # Verify
    if result['category']['label'] == "مشكلة نت":
        print("✅ Category classification: CORRECT")
    else:
        print(f"❌ Category classification: WRONG (expected: مشكلة نت)")
