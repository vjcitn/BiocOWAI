## open weight AI support with pi harness and Bioc 3.23, R 4.6, Rstudio

custom container image: vjcitn/biocowai:0.0.3

see https://github.com/vjcitn/BiocOWAI

when rstudio cloud env (should be 8 core, 52GB RAM) is running, in terminal: 

```
t4-coder install  # downloads persistent weights file
t4-coder start  # will return with information on environment
```

Once the server is running you can use `pi` to do agentic coding in your folders on AnVIL.

### How it starts and tunes to your configuration

![startup snapshot](https://raw.githubusercontent.com/vjcitn/BiocOWAI/main/zoomstart.jpg)

### Working with the pi harness in terminal

`pi` is on your path.

![pi snapshot](https://raw.githubusercontent.com/vjcitn/BiocOWAI/main/startpi.jpg)

### Working with pi in console

See [here](https://github.com/vjcitn/BiocOWAI/tree/main/using_pi)

### Working with ellmer in console

```
library(ellmer)
Sys.setenv(OPENAI_API_KEY = "local")   # llama-server ignores it; ellmer just wants something set

chat <- chat_openai_compatible(
  base_url = "http://127.0.0.1:8181/v1",
  model    = "qwen3.6-35b-a3b",          # the alias t4-coder gives the server
  system_prompt = "You are a concise R/Bioconductor assistant."
)
chat$chat("Write a function that returns the median absolute deviation, ignoring NAs.")
```

