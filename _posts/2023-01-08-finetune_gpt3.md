---
title: "Fine tuning GPT3 to soud like podcast host"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

## Executive summary

I developed four fine-tuned versions of OpenAI's [GPT-3 models](https://beta.openai.com/docs/models/gpt-3) to sound like [Russ Roberts](https://russroberts.info/) who is the host of [EconTalk](https://www.econlib.org/econtalk/) using more than 10 years of episode transcripts. I calibrated my experiments for a $40 USD budget and tested the fine-tuned model of 12 quintessential Russ/EconTalk questions.

This process led to four key insights:

1. The Davinci model (which powers [ChatGPT](https://chat.openai.com/)) provides decent zero-shot accuracy (see [baseline results](#baseline-results)), and even better results with fine tuning (see [results](#finetuning-results)).
2. All models require prompt engineering and repeated experimentation.
3. The Curie model provides poor zero-shot accuracy but does okay after fine-tuning, specifically for "memorization".
4. Even with fine-tuning the Ada and Babbage models give nonsensical answers to main prompts.
5. Overall, it is clear that with my resources (i.e. money) a fine-tuned model of the Davinci model (it was only trained on <2% of the corpus) would be able to provide a good impersonation of the answers and "style" of public figures with a large corpus of conversational text.

The rest of this post provides more details on the development of this model and the experimental results. Interested readers can create their only versions of [`EconChatR`](https://github.com/ErikinBC/EconChattR) by cloning this repo and running the main pipeline. 

<br>

## Background

People are [very excited](https://techmonitor.ai/technology/ai-and-automation/chatgpt-openai-chatbot) about [ChatGPT](https://chat.openai.com/chat).[[^1]] ChatGPT is the 3rd generation of large language models ([LLMs](https://en.wikipedia.org/wiki/Wikipedia:Large_language_models)) from [OpenAI](https://openai.com/), GPT 3.5, that has been able to produce human-like performance for a variety of natural language tasks. LLMs use many multi-headed attention layers (a type of architecture [first proposed](https://arxiv.org/abs/1706.03762) by Google researchers in 2017) which are more amenable to GPU speedups. The `Davinci` version of GPT-3.5 (which powers ChatGPT) is very large,[[^2]] and would be prohibitive to run on a local machine.[[^3]] 

From what OpenAI has [published](https://arxiv.org/abs/2005.14165), GPT-3 was trained of a large corpus of mainly English sources using a very simple training framework: predict the next token given a context. What's fascinating is how such a simple [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) training process yields such impressive results on a variety of tasks from language translation to summarization to logical reasoning. If you wanted to [re-create ChatGPT](https://github.com/lucidrains/PaLM-rlhf-pytorch) from scratch you would [likely need]((https://lifearchitect.ai/chatgpt/)) a supercomputer cluster with ~10,000 GPUs and ~285,000 CPU cores (~$1 billion USD to rent), spend months training the model, and then spend months/year labeling the output of the model to further hone its abilities. One of OpenAI's "secret ingredients" is an unspecified amount of resources spent doing Reinforcement Learning from Human Feedback ([RLHF]((https://openai.com/blog/instruction-following/))). Basically this means having humans interact with the ChatBot, and label output as either correct/incorrect or rank-order multiple generations from the same prompt. 

ChatGPT has shown amazing performance in its ability to [write code](https://twitter.com/gabe_ragland/status/1598068207994429441), [write academic papers](https://icml.cc/Conferences/2023/llm-policy), [answer most homework questions](https://www.nbcnews.com/tech/tech-news/new-york-city-public-schools-ban-chatgpt-devices-networks-rcna64446), [write entire syllabi](https://oneusefulthing.substack.com/p/the-mechanical-professor), and [come up decent rhyming poems](https://lil.law.harvard.edu/blog/2022/12/20/chatgpt-poems-and-secrets/). 


<p align="center"><img src="/figures/tomato_poem.png" width="25%"></p>
<p><i>ChatGPT's cherry tomato poem is pretty good!</i></p>


## Introduction to fine-tuning

Even with unprecedented scale, training, and RLHF fine-tuning, ChatGPT still shows a tendency to "[hallucinate](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))", giving nonsensical answers, made-up facts, and repeated sentences. One way to address this issue is to fine-tune a version of GPT-3 on a custom dataset that updates the models weights in a way that makes more likely to give answers consistent with the task you have in mind. For example, we may want the model to give medically specific and accurate answers to questions [questions related to depression](https://betterprogramming.pub/how-to-finetune-gpt-3-finetuning-our-virtual-mental-health-assistant-641c1f3b1ef3). This can be done with a new collection of prompt/completion data points:

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

This process can be generalized to having the model give answers which "sound" like a person if there are enough examples ("completion" points) of that person responding to different questions. My goal with [`EconChatR`](https://github.com/ErikinBC/EconChattR) was to use OpenAI's native [fine-tuning](https://beta.openai.com/docs/guides/fine-tuning) functionality to give answers that would sound more like Russ Roberts, the host of EconTalk. For example, below are two prompt-completion examples where the prompt is a (truncated) statement from an EconTalk guest (with "Russ Roberts responds:" added as the suffix) followed by Russ' subsequent comment as a completion. 

```json
{"prompt":"... So, here I have one for the city of Shanghai, where they claim that they have eliminated 1, 367, 440 sparrows. You think, 'Really, was there somebody counting these sparrows? I doubt that very much. Russ Roberts responds:",
"completion":"Yeah. Well, it was probably 443 at the end. But, yeah, there's a terrible rounding and inaccuracy. I'm sure long-time listeners will remember my joke about macroeconomics: 'How do you know an economist has a sense of humor? They use decimal points. So, here's the same kind of ridiculous--tragic, really. There's a comic element to it. But it's tragicomic."}
{"prompt":"... It's like, 'Why are things done in a certain way? It's like, 'Well, try doing them some other way and you'll quickly find out why they're done in that way. This is a very classic phenomenon that people run into. Russ Roberts responds:",
"completion":"It's the Chesterton fence argument, and you think you know why the fence is there, but it evolved for some reason that you don't know of and you should start--your default should be, 'There's a reason it's there and I don't understand it.\n"}
```

After we've created a fine-tuned model we should be able to ask questions like: "*You often say that economists have a sense of humor because they use decimal points, why is that? Russ Roberts responds:*", with the prompt suffix "reminding" our model the sort of dialogue we expect to receive back. 


<a id="baseline-results"></a>
### Baseline results

Before spending money fine-tuning model, let's evaluate how well ChatGPT does "out of the box" (i.e. zero-shot) on 12 quintessential Russ'isms and EconTalk facts (inspired by the [EconTalk drinking game](http://www.salanwoodbine.com/econtalkdrinkinggame/)):

1. Does Russ Roberts, the host of EconTalk, mention the "Bootlegger and Baptist" theory often?
2. As a proponent of free markets, is Russ Roberts attracted to the "Bootlegger and Baptist" theory?
3. How skeptical is Russ Roberts, the host of EconTalk, of empirical research published in academic journals?
4. Does Russ Roberts, the host of EconTalk, believe that "data can speak for itself" as a general principle?
5. Finish the following sentence: "Man desires not only to be loved
6. Finish the following sentence: "The curious task of economics
7. Why is "skin in the game" a deep insight?
8. In what way does understanding the challenge around "creating a prairie" align with F A Hayek's famous maxim about the curious task of economics?
9. Why is it harder to "create a prairie" then it seems?
10. As a free market economist, why is Russ Roberts drawn to the idea of the "Chesterton Fence"?
11. Russ Roberts often says that "Macroeconomists have a sense of humor" why is that?
12. Who are some guests from EconTalk from the year 2014?


As Table 1 shows below....

<p align="center"> Table 1: Completions to baseline prompts </p>
<table>
  <tr>
    <td><img src="/figures/q1_econtalk.jpg" alt="q1" width="20%"></td>
    <td><img src="/figures/q2_econtalk.jpg" alt="q2" width="20%"></td>
  </tr>
  <tr>
    <td><img src="/figures/q3_econtalk.jpg" alt="q3" width="20%"></td>
    <td><img src="/figures/q4_econtalk.jpg" alt="q4" width="20%"></td>
  </tr>
  <tr>
    <td><img src="/figures/q5_econtalk.jpg" alt="q5" width="20%"></td>
    <td><img src="/figures/q6_econtalk.jpg" alt="q6" width="20%"></td>
  </tr>
  <tr>
    <td><img src="/figures/q7_econtalk.jpg" alt="q7" width="20%"></td>
    <td><img src="/figures/q8_econtalk.jpg" alt="q8" width="20%"></td>
  </tr>
  <tr>
    <td><img src="/figures/q9_econtalk.jpg" alt="q9" width="20%"></td>
    <td><img src="/figures/q10_econtalk.jpg" alt="q10" width="20%"></td>
  </tr>
  <tr>
    <td><img src="/figures/q11_econtalk.jpg" alt="q11" width="20%"></td>
    <td><img src="/figures/q12_econtalk.jpg" alt="q12" width="20%"></td>
  </tr>
</table>



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

<a id="finetuning-results"></a>
### Fine-tuning results


<img src="/figures/playground.png">



<br>

***

### Footnotes

[^1]: Here are some fairly emblematic headlines that have come out on social and traditional media platforms: "[Google is done. Here’s why OpenAI’s ChatGPT Will Be a Game Changer](https://medium.com/@lucapetriconi/google-is-done-heres-why-openai-s-chatgpt-will-be-a-game-changer-98ae591ad747)", "[ChatGPT is a game changer for artificial intelligence](https://www.ctvnews.ca/video?clipId=2585069)", "[ChatGPT is a GAME CHANGER!](https://www.reddit.com/r/algotrading/comments/zkdi3e/chatgpt_is_a_game_changer/)".

[^2]: The model has ~170B parameters and was trained on about 300 billion tokens (where 5 tokens averages about 4 words).

[^3]: It would [likely take](https://twitter.com/tomgoldsteincs/status/1600196981955100694?lang=en) 8 high-end GPUs to run a model of this size on a local machine (each GPU would cost ~$15K). Furthermore, it has been [estimated](https://lambdalabs.com/blog/demystifying-gpt-3) that GPT-3 would have cost $5 million USD  to train and One [popular estimate](https://twitter.com/tomgoldsteincs/status/1600196995389366274) has pegged the cost of running ChatGPT at $3 million USD a month.
