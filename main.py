import dspy
import os
from dotenv import load_dotenv
from medical_classification.classify import specialty_classify
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1, 
    max_output_tokens=256,
    cache=True,
)
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)

app = FastAPI()

class ClassificationRequestBody(BaseModel):
    wall_of_text: str = Field(title="Wall of text to classify",
                              description="The wall of text to classify. This should be a string containing the text to be classified.",
                              default=None,
                              example="Patient presents with a rash on the arm.")


@app.post("/classify", status_code=status.HTTP_200_OK)
def classify_handler( body: ClassificationRequestBody):
    wall_of_text = body.wall_of_text 
    if not wall_of_text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "wall_of_text is required."},
        )
    
    #  Added some basic query validation
    if len(wall_of_text) > 2000 or len(wall_of_text) < 10:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "wall_of_text is too long. Maximum length is 2000 characters."},
        )
    classification_resp = specialty_classify(
        wall_of_text=wall_of_text,
    )
    print('*' * 50)
    print(f"response: {classification_resp}")
    dspy.inspect_history(n=1)
    print('*' * 50)
    # print(classification_resp)
    print(f'cost of single-turn invocation: {lm.history[-1]["cost"]}')
    print('*' * 50)
    
    return {
        "response": classification_resp,
    }
