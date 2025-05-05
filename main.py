import dspy
import os
from dotenv import load_dotenv
from medical_classification.classify import specialty_classify
from fastapi import FastAPI
from pydantic import BaseModel


load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1,
    cache=True,
)
dspy.configure(lm=lm)
# dspy.settings.configure(track_usage=True)

app = FastAPI()

class ClassificationRequestBody(BaseModel):
    wall_of_text: str


@app.post("/classify")
def classify_handler( body: ClassificationRequestBody):
    wall_of_text = body.wall_of_text  
    print(f"Classifying wall of text: {wall_of_text}")
    classification_resp = specialty_classify(
        wall_of_text=wall_of_text,
    )
    print(f"response: {classification_resp}")
    dspy.inspect_history(n=1)
    print(f'cost: {lm.history[-1]["cost"]}')
    print(f'cost: {lm.history[-1]["usage"]}')
    return {
        "response": classification_resp,
    }
