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

app = FastAPI(title="Medical Specialty Classification API",
              description="This API classifies a wall of text to an appropriate medical specialty and confidence score.",
              version="1.0.0",
            )

class ClassificationRequestBody(BaseModel):
    wall_of_text: str = Field(title="Wall of text to classify",
                              description="The wall of text to classify. This should be a string containing the text to be classified.",
                              default=None,
                              example="Patient presents with a rash on the arm.",
                              min_length=10, max_length=2000)


@app.post("/classify", status_code=status.HTTP_200_OK)
def classify_handler( body: ClassificationRequestBody):

    if not body.wall_of_text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "wall_of_text is required"},
        )
    wall_of_text = body.wall_of_text 
    
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
