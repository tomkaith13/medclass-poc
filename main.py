import dspy
import os
from dotenv import load_dotenv
from medical_classification.classify import specialty_classify
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager
import asyncio


load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1, 
    max_output_tokens=256,
    cache=True,
)
dspy.configure(lm=lm, track_usage=True, async_max_workers=8)
# dspy.settings.configure(track_usage=True,async_max_workers=8 )



class ClassificationRequestBody(BaseModel):
    wall_of_text: str = Field(title="Wall of text to classify",
                              description="The wall of text to classify. This should be a string containing the text to be classified.",
                              default=None,
                              example="Patient presents with a rash on the arm.",
                              min_length=10, max_length=2000)

# init MCP client
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="/opt/homebrew/bin/npx",  # Executable
    args=["-y", "@modelcontextprotocol/server-google-maps"],
    env={"GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY")},
)


async def fetch_home_address_coordinates():
    """Fetch home address coordinates as the last resort if location based tools fail. """

    await asyncio.sleep(1)  # Simulate async work

    return "12.34, -43.21"

def fetch_work_address_coordinates():
    """Fetch work address coordinates. If this fails, use home address coordinates using fetch_home_address_coordinates tool. """

    # Uncomment the following line to simulate a failure
    # raise Exception("Work address coordinates not found. Using home address coordinates instead.")
    return "56.78, -87.65"

async def query_run(query: str):
    print("Starting MCP client...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))
            
            reactAgent = dspy.ReAct(UserQueryToLocationCoordinates,
                                    tools=dspy_tools + [fetch_home_address_coordinates, fetch_work_address_coordinates])
            
            result = await reactAgent.acall(query=query)
            # print(result)
            return result



app = FastAPI(title="Medical Specialty Classification API",
              description="This API classifies a wall of text to an appropriate medical specialty and confidence score.",
              version="1.0.0"
              )

class UserQueryToLocationCoordinates(dspy.Signature):
    """Convert a user query to location coordinates. Use only values from the tools and NO OTHER.
      If location is not found, call the fetch_work_coordinates tool and return that address.
        If that fails as well, default to tool fetch_home_coordinates."""

    query: str = dspy.InputField()
    location_coordinates: str = dspy.OutputField(
        desc=("Extract any possible address from the query. Use that address location to get coordinates in the format '(latitude, longitude)'."),
        example="37.7749, -122.4194",
    )

     
@app.post("/classify", status_code=status.HTTP_200_OK)
async def classify_handler( body: ClassificationRequestBody):

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

    result = await query_run(query=wall_of_text)
    dspy.inspect_history(n=1)
    print(f'cost of single-turn invocation: {lm.history[-1]["cost"]}')
    # print("results::", result)
    
    
    return {
        "response": classification_resp,
        "location_coordinates": result.location_coordinates,
    }
