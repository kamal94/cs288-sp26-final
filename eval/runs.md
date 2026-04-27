I’ll give you the commands I ran for each (just plug in the API key and replace dataset with the subset of TruthfulQA questions): 
```
python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gpt-3.5-turbo-instruct.csv \
--provider llama \
--llama_base_url https://openrouter.ai/api/v1 \
--model openai/gpt-3.5-turbo-instruct \
--temperature 0.0 \
--num_turns 7 --max_workers 4 

python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gpt-3.5-turbo.csv \
--provider llama \
--llama_base_url https://openrouter.ai/api/v1 \
--model openai/gpt-3.5-turbo \
--temperature 0.0 \
--num_turns 7 --max_workers 4 

python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gemini-2.0-flash.csv \
--provider llama \
--llama_base_url https://openrouter.ai/api/v1 \
--model google/gemini-2.0-flash-001 \
--temperature 0.0 \
--num_turns 7 --max_workers 4 

python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_llama-3.1-70b.csv \
--provider llama \
--llama_base_url https://openrouter.ai/api/v1 \
--model meta-llama/llama-3.1-70b-instruct \
--temperature 0.0 \
--num_turns 7 --max_workers 4 


python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gemma-4-e2b.csv \
--provider llama \
--llama_base_url http://localhost:11434/v1 \
--model gemma4:e2b \
--temperature 0.0 \
--num_turns 7 --max_workers 1 --max_tokens 1048576


python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gemma-4-e4b.csv \
--provider llama \
--llama_base_url http://localhost:11434/v1 \
--model gemma4:e4b \
--temperature 0.0 \
--num_turns 7 --max_workers 1 --max_tokens 1048576

python -m eval.static_eval \
--input_csv data/questions.csv \
--output_csv outputs/truthfulqa_100_static_eval_gemma-4-e4b.csv \
--provider llama \
--llama_base_url http://localhost:11434/v1 \
--model gemma4:e4b \
--temperature 0.0 \
--num_turns 7 --max_workers 1 --max_tokens 1048576

```