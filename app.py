from fastapi import FastAPI

# Initialize the FastAPI app
app = FastAPI()

# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}
