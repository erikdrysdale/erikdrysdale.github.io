---
title: "Fine tuning GPT3 to soud like podcast host"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

## Executive summary

Words go here


## Background

People are [very excited](https://techmonitor.ai/technology/ai-and-automation/chatgpt-openai-chatbot) about [ChatGPT](https://chat.openai.com/chat).[[^1]] ChatGPT is the 3rd generation of large language model ([LLM](https://en.wikipedia.org/wiki/Wikipedia:Large_language_models)) from [OpenAI](https://openai.com/) that uses many multi-headed attention layers (a type of architecture [first proposed](https://arxiv.org/abs/1706.03762) by Google researchers in 2017) to carry out a variety of natural language tasks. The model is very large,[[^2]] needing 8 high-end GPUs to run a single query, and was training of a large corpus of mainly English using a very simple training framework: predict the next token given a context. What's fascinating is how such a simple [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) training process yields such impressive results on a variety of tasks from language translation to summarization to logical reasoning. 


Talk about Reinforcement Learning from Human Feedback ([RLHF]((https://openai.com/blog/instruction-following/))) ... 


[ChatGPT](https://chat.openai.com/chat) is currently free to use. 


Here is a example of impressive ability to write poetry





## Data and processing


As of January 4th, the episode transcripts range from: 2022-12-20 to 2012-01-11


      ntokens   nwords    nchars
mean      122       98       548
sum   7606445  6086970  34037331


         Cost per epoch   Usage
Model                          
Ada                2.41    9.48
Babbage            3.59   14.19
Curie             17.72   70.71
Davinci          176.68  706.55



Given the size of the corpus and wanting to run the model for at least 4 epochs for all for models to maximum cost of $10USD per model I had to...

<br>

## (2) Model training

TBD


<br>

## (3) Inference and testing


<img src="/figures/playground.png">
<img src="/figures/tomato_poem.png">



<br>

***

### Footnotes

[^1]: Here are some fairly emblematic headlines that have come out on social and traditional media platforms: "[Google is done. Here’s why OpenAI’s ChatGPT Will Be a Game Changer](https://medium.com/@lucapetriconi/google-is-done-heres-why-openai-s-chatgpt-will-be-a-game-changer-98ae591ad747)", "[ChatGPT is a game changer for artificial intelligence](https://www.ctvnews.ca/video?clipId=2585069)", "[ChatGPT is a GAME CHANGER!](https://www.reddit.com/r/algotrading/comments/zkdi3e/chatgpt_is_a_game_changer/)".

[^2]: The model has ~170B parameters and was trained on about 300 billion tokens (where 5 tokens averages about 4 words).
