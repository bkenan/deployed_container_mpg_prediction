#!/usr/bin/env bash

PORT=8080
echo "Port: $PORT"

# POST method predict
curl -d '{  
   "cylinders":{  
      "0":8
   },
   "displacement":{  
      "0":307.0
   },
   "horsepower":{  
      "0":130
   },
   "weight":{  
      "0":3504
   },
   "acceleration":{  
      "0":12.0
   },
   "year":{  
      "0":70
   }
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict
