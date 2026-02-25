from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

def test_scrubber():
    print("Initializing Bank-Grade PII Scrubber...")
    
    # 1. Initialize engines
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # 2. Simulated User Input (Highly Sensitive)
    raw_text = "My name is John Doe. My phone number is 212-555-1234, and I need to transfer $500."
    print(f"\n[RAW INPUT]   {raw_text}")
    
    # 3. Analyze for PII
    results = analyzer.analyze(text=raw_text, entities=["PERSON", "PHONE_NUMBER"], language='en')
    
    # 4. Define tokenization rules (Replace with <ENTITY_TYPE>)
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON_TOKEN>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_TOKEN>"})
    }
    
    # 5. Anonymize
    anonymized_result = anonymizer.anonymize(text=raw_text, analyzer_results=results, operators=operators)
    
    print(f"\n[SCRUBBED]    {anonymized_result.text}")
    print("\nVerification Complete. If you see tokens instead of PII, you are cleared to proceed.")

if __name__ == "__main__":
    test_scrubber()