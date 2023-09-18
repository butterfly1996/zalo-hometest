# Ex 1
## Usage
To active api, run
```
python3 api.py
```
or run using docker
```
docker build -t ex1 .
docker run -p 8000:8000 ex1
```
then run by curl
```
curl -X GET -H "Content-Type: application/json" http://localhost:8000/v1/generate?text=chó%20và%20mèo%20là%202%20con%20vật&max_length=50
```
# Ex 2
## Usage
To run the model, run a_test_zalo.py
```
python3 a_test_zalo.py --model_name "NlpHUST/gpt2-vietnamese" --no_repeat_ngram_size 0 --generate_type greedy_search --num_beam 1 --max_new_tokens 128 --prompt "chó và mèo là 2 con vật"
```



