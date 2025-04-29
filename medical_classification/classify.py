from typing import Literal
import dspy

class SpecialtyClassifier(dspy.Signature):
    """Classify a wall of text to an appropriate medical specialty and confidence score. If nothing matches (confidence is less than 0.5), return '' """

    wall_of_text: str = dspy.InputField()
   

    specialty: Literal[ 'dermatology',
        'orthopedics',
        'cardiology',
        'neurology',
        'psychiatry',
        'pediatrics',
        'internal medicine',
        'family medicine',
        'emergency medicine',
        'radiology'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

specialty_classify = dspy.ChainOfThought(SpecialtyClassifier)