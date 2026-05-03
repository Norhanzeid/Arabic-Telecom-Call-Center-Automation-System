"""
LLM Analyzer - Uses OpenAI API for Category & Sentiment Classification
No PyTorch/sentence_transformers required - lightweight solution
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import logging
from openai import OpenAI
from prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMAnalyzer:
    def __init__(self):
        # Problem categories
        self.categories = {
            "مشكلة نت": "مشاكل الإنترنت والاتصال",
            "مشكلة فواتير": "مشاكل الفواتير والدفع",
            "طلب خدمة": "طلب خدمة جديدة أو ترقية"
        }
        
        # Negative keywords for quick sentiment detection
        self.negative_keywords = [
            "فصل", "قطع", "مشكلة", "عطل", "مفيش", "مافيش", 
            "بطيء", "تأخير", "بشتكي", "فجأة", "فجاة", "انقطاع",
            "مش شغال", "متوقف", "معطل", "خرب", "باظ", "تعب",
            "زهقان", "مضايق", "غضبان", "مستاء", "مبسوط مش",
            "النت فصل", "الانترنت فصل", "مفيش نت", "مافيش نت"
        ]

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def analyze_text(self, text):
        """
        Analyze text using OpenAI API for category and sentiment
        """
        text = self.clean_text(text)
        if not text:
            return {
                'category': {'label': 'غير محدد'},
                'sentiment': {'label': 'سلبي'}
            }

        # Quick sentiment check using keywords
        is_negative = any(word in text for word in self.negative_keywords)
        
        # Force classification for clear internet problems
        internet_problem_phrases = [
            "الإنترنت فصل", "النت فصل", "مفيش إنترنت", "مفيش نت",
            "الإنترنت مش شغال", "النت مش شغال", "مشكلة في الإنترنت",
            "انقطاع الإنترنت", "عطل في النت", "الإنترنت بطيء"
        ]
        
        # If it's clearly an internet problem, don't even ask GPT
        if any(phrase in text for phrase in internet_problem_phrases):
            return {
                'category': {'label': 'مشكلة نت'},
                'sentiment': {'label': 'سلبي'}
            }
        
        try:
            # Use OpenAI for classification
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("llm_analyzer_system")
                    },
                    {
                        "role": "user",
                        "content": text[:500]  # Limit text length
                    }
                ],
                temperature=0,
                max_tokens=50
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            # Extract category and sentiment from response
            try:
                result = json.loads(result_text)
                category = result.get('category', 'غير محدد')
                sentiment = result.get('sentiment', 'سلبي')
            except json.JSONDecodeError:
                # Fallback parsing
                category = self._keyword_classify(text)
                sentiment = "سلبي" if is_negative else "إيجابي"
            
            # Force internet problem classification if obvious patterns found
            if any(phrase in text for phrase in internet_problem_phrases):
                category = "مشكلة نت"
            
            # Override sentiment if negative keywords found
            if is_negative:
                sentiment = "سلبي"
                
            return {
                'category': {'label': category},
                'sentiment': {'label': sentiment}
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis error: {str(e)}")
            # Fallback to keyword-based classification
            category = self._keyword_classify(text)
            sentiment = "سلبي" if is_negative else "إيجابي"
            
            return {
                'category': {'label': category},
                'sentiment': {'label': sentiment}
            }

    def _keyword_classify(self, text):
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        
        # Internet problem keywords (prioritized)
        net_problem_keywords = [
            "نت", "انترنت", "راوتر", "واي فاي", "سرعة", "اتصال", 
            "فصل", "قطع", "انقطاع", "بطيء", "بطء", "عطل", "مشكلة",
            "مفيش نت", "مافيش نت", "الانترنت فصل", "النت فصل",
            "مش شغال", "متوقف", "معطل"
        ]
        
        # Check for internet problems first (highest priority)
        if any(word in text_lower for word in net_problem_keywords):
            # Additional check: if it contains internet + problem words = definitely net issue
            internet_words = ["نت", "انترنت", "راوتر", "واي فاي"]
            problem_words = ["فصل", "قطع", "مشكلة", "عطل", "بطيء", "مفيش", "مش شغال"]
            
            has_internet = any(word in text_lower for word in internet_words)
            has_problem = any(word in text_lower for word in problem_words)
            
            if has_internet and has_problem:
                return "مشكلة نت"
            elif has_internet:
                return "مشكلة نت"  # Default for internet-related content
        
        # Billing keywords (second priority)
        bill_keywords = ["فاتورة", "دفع", "رصيد", "فلوس", "حساب", "اشتراك", "مبلغ"]
        if any(word in text_lower for word in bill_keywords):
            # But if it also mentions internet problems, it's still a net issue
            if any(word in text_lower for word in net_problem_keywords):
                return "مشكلة نت"
            return "مشكلة فواتير"
        
        # Service request keywords (lowest priority)
        service_keywords = ["طلب", "عايز", "محتاج", "ترقية", "عرض", "باقة جديدة"]
        if any(word in text_lower for word in service_keywords):
            # But if it mentions problems, it's not a service request
            if any(word in text_lower for word in net_problem_keywords):
                return "مشكلة نت"
            return "طلب خدمة"
        
        # Default: if nothing specific found, assume it's internet issue
        # (since most call center calls are about problems)
        return "مشكلة نت"

    def analyze_batch(self, texts):
        """Analyze multiple texts"""
        return [self.analyze_text(text) for text in texts]


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    analyzer = LLMAnalyzer()
    
    # Test the exact text from the user's example
    test_text = """مساء الخير يا فندم، شكراً لاتصالك WE للاتصالات. أحمد في خدمتك، ممكن أتشرف باسم حضرتك؟ أهلاً سيد أحمد، أنا اسمي هبه. والحقيقة أنا أنا موجودة على خدمة الانتظار من فترة طويلة جداً. أنا أعتذر جداً يا فندم، الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جداً. آسفين جداً على وقت حضرتك. ممكن أحضرك تقوليلي برقم المتاح عليه الخدمة؟ طبعاً الرقم هو 01274714823. تمام يا فندم، هل الخط مسجل باسم حضرتك؟ مسجل باسمي. طيب ممكن توضحيلي الاسم بالكامل يا فندم؟ هبه عاطف الناغي. تمام، شكراً جداً يا سيدة هبه. أقدر أساعد حضرتك إزاي؟ الحقيقة سيدة أحمد، أنا عايزة أقول لك أن أنا بالضبط على بعد 30 دقيقة من هتفصل من شغلي. أنا كنت بشتغل والإنترنت شغال بشكل طبيعي، وبعد ما حضرت ملفات الإكسل والإيميلات اللي كنت محتاجة أبعتها قبل ما ضغط على الإرسال، فجأة الإنترنت فصل وعلامة الإنترنت على جهاز الكمبيوتر بتقول أني مفيش إنترنت. والموضوع مش منطقي خالص، لأني دفعت الفاتورة في معادها الساعة 9 وأنا محتاجة أبعت الشغل بتاعي الساعة 10 على الأكتر. فمن فضلك سيدة أحمد، أنا محتاجة حل لأن الوقت لو عدى من غير ما أبعت الإيميلات دي، هيبقى فيه مشكلة كبيرة في شغلي."""
    
    result = analyzer.analyze_text(test_text)
    
    print("="*60)
    print("Testing LLM Analyzer")
    print("="*60)
    print(f"Category: {result['category']['label']}")
    print(f"Sentiment: {result['sentiment']['label']}")
    print("="*60)
    
    # Should be: مشكلة نت, سلبي
    expected_category = "مشكلة نت"
    if result['category']['label'] == expected_category:
        print("✅ Category classification: CORRECT")
    else:
        print(f"❌ Category classification: WRONG (expected: {expected_category})")
    
    print()
    print("Key phrases found in text:")
    key_phrases = ["فجأة الإنترنت فصل", "مفيش إنترنت", "دفعت الفاتورة"]
    for phrase in key_phrases:
        if phrase in test_text:
            print(f"✅ Found: '{phrase}'")
        else:
            print(f"❌ Not found: '{phrase}'")

