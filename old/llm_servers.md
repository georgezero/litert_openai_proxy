[paui - rpi5 - litert-lm - gemma-4-e2b-it]

http://paui:8000/v1
http://192.168.86.30:8000/v1

model: gemma-4-e2b-it

curl -s http://192.168.86.30:8000/v1/chat/completions \          :(
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e2b-it","messages":[{"role":"user","content":"hello from lan"}]}'
