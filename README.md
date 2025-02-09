# Movie Success Predictor
Predict if a movie will be a hit or flop based on features like budget, runtime, and profit margin.

## Tech Stack
- Python
- Scikit-learn
- FastAPI

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Movie-Success-Prediction.git
   cd app

2. Install dependencies:
   pip install -r requirements.txt

3. Run the FastAPI app:
   uvicorn main:app --reload

4. The API will be available at:
   http://127.0.0.1:8000

5. You can test the API using Swagger UI:
   http://127.0.0.1:8000/docs

**Endpoint**: `/predict/`

**Method**: POST

**Sample Input**:
```json
{
    "budget": 50000000,
    "runtime": 120,
    "profit_margin": 0.5,
    "release_month": 6,
    "Action": 1,
    "Comedy": 0,
    ...
}

 **Sample Response**:
{
    "prediction": 1,
    "description": "Hit"
}



