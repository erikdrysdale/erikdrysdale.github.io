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

1. The Davinci model (which powers [ChatGPT](https://chat.openai.com/)) provides decent zero-shot accuracy (~50% accuracy see [baseline results](#baseline-results)), and even better results with fine tuning (~85% accuracy see [results](#finetuning-results)), including a qualitative improvement in Russ-like diction.
2. All models require prompt engineering and repeated experimentation.
3. The Curie model provides poor zero-shot accuracy but does okay after fine-tuning, specifically for "memorization".
4. Even with fine-tuning the Ada and Babbage models give nonsensical answers to most prompts.
5. Overall, a fine-tuned Davinci model with more resources (the one used in this experiment was trained on <2% of the corpus) would be able to provide a good impersonation of the answers and "style" of public figures with a large corpus of conversational text.

The rest of this post provides more details on the development of this model and the experimental results. Interested readers can create their only versions of [`EconChatR`](https://github.com/ErikinBC/EconChattR) by cloning this repo and running the main pipeline. 

<br>

## Background

People are [very excited](https://techmonitor.ai/technology/ai-and-automation/chatgpt-openai-chatbot) about [ChatGPT](https://chat.openai.com/chat).[[^1]] ChatGPT is the 3rd generation of large language models ([LLMs](https://en.wikipedia.org/wiki/Wikipedia:Large_language_models)) from [OpenAI](https://openai.com/), GPT 3.5, that has been able to produce human-like performance for a variety of natural language tasks. LLMs use many multi-headed attention layers (a type of architecture [first proposed](https://arxiv.org/abs/1706.03762) by Google researchers in 2017) which are more amenable to GPU speedups. The `Davinci` version of GPT-3.5 (which powers ChatGPT) is very large,[[^2]] and would be prohibitive to run on a local machine.[[^3]] 

From what OpenAI has [published](https://arxiv.org/abs/2005.14165), GPT-3 was trained of a large corpus of mainly English sources using a very simple training framework: predict the next token given a context. What's fascinating is how such a simple [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) training process yields such impressive results on a variety of tasks from language translation to summarization to logical reasoning. If you wanted to [re-create ChatGPT](https://github.com/lucidrains/PaLM-rlhf-pytorch) from scratch you would [likely need]((https://lifearchitect.ai/chatgpt/)) a supercomputer cluster with ~10,000 GPUs and ~285,000 CPU cores (~$1 billion USD to rent), spend months training the model, and then spend months/year labeling the output of the model to further hone its abilities. One of OpenAI's "secret ingredients" is an unspecified amount of resources spent doing Reinforcement Learning from Human Feedback ([RLHF]((https://openai.com/blog/instruction-following/))). Basically this means having humans interact with the ChatBot, and label output as either correct/incorrect or rank-order multiple generations from the same prompt. 

ChatGPT has shown amazing performance in its ability to [write code](https://twitter.com/gabe_ragland/status/1598068207994429441), [write academic papers](https://icml.cc/Conferences/2023/llm-policy), [answer most homework questions](https://www.nbcnews.com/tech/tech-news/new-york-city-public-schools-ban-chatgpt-devices-networks-rcna64446), [write entire syllabi](https://oneusefulthing.substack.com/p/the-mechanical-professor), and [come up decent rhyming poems](https://lil.law.harvard.edu/blog/2022/12/20/chatgpt-poems-and-secrets/). 


<p align="center"><img src="/figures/tomato_poem.png" width="25%"></p>
<p align="center"><i>ChatGPT's cherry tomato poem is pretty good!</i></p>


## Introduction to fine-tuning

Even with unprecedented scale, training, and RLHF fine-tuning, ChatGPT still shows a tendency to "[hallucinate](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))", giving nonsensical answers, made-up facts, and repeated sentences. One way to address this issue is to fine-tune a version of GPT-3 on a custom dataset that updates the models weights in a way that makes more likely to give answers consistent with the task you have in mind. For example, we may want the model to give medically specific and accurate answers to questions [questions related to depression](https://betterprogramming.pub/how-to-finetune-gpt-3-finetuning-our-virtual-mental-health-assistant-641c1f3b1ef3). This can be done with a new collection of prompt/completion data points:

```json
{"prompt": "<prompt text #1>", "completion": "<ideal generated text #1>"}
{"prompt": "<prompt text #2>", "completion": "<ideal generated text #2>"}
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


As Table 1 shows below that out of the box ChatGPT does well for half of the questions 1, 2, 6, 7, 8, and 10. Questions 5 and 11 shows that the model is hesitant to answers to specific people, and surprisingly has a hard time remembering the famous quote by Adam Smith. The model also provides inaccurate answers for question 12, providing some real EconTalk guests, but not necessarily those who were there in 2014 (for example Ben Bernanke has not been a guest), and strangely adding that Russ was a guest on his own podcast. All this suggests that there is room for improvement with a fine-tuned model to give more accurate and Russ-like answers. 

<br>

<p align="center"> Table 1: Completions to baseline prompts </p>
<table align="center">
  <tr>
    <td><img src="/figures/q1_econtalk.png" alt="q1" width=360px></td>
    <td><img src="/figures/q2_econtalk.png" alt="q2" width=360px></td>
  </tr>
  <tr>
    <td><img src="/figures/q3_econtalk.png" alt="q3" width=360px></td>
    <td><img src="/figures/q4_econtalk.png" alt="q4" width=360px></td>
  </tr>
  <tr>
    <td><img src="/figures/q5_econtalk.png" alt="q5" width=360px></td>
    <td><img src="/figures/q6_econtalk.png" alt="q6" width=360px></td>
  </tr>
  <tr>
    <td><img src="/figures/q7_econtalk.png" alt="q7" width=360px></td>
    <td><img src="/figures/q8_econtalk.png" alt="q8" width=360px></td>
  </tr>
  <tr>
    <td><img src="/figures/q9_econtalk.png" alt="q9" width=360px></td>
    <td><img src="/figures/q10_econtalk.png" alt="q10" width=360px></td>
  </tr>
  <tr>
    <td><img src="/figures/q11_econtalk.png" alt="q11" width=360px></td>
    <td><img src="/figures/q12_econtalk.png" alt="q12" width=360px></td>
  </tr>
</table>


<br>

## Data and processing

The first step in any data science project is gather and clean a dataset and then properly format it. A full list of EconTalk episodes was obtained from the [xml file](http://www.econlib.org/library/EconTalk.xml) hosted on EconLib. A [simple web-scraper](https://github.com/ErikinBC/EconChattR/blob/main/1_scrape.R) was then used to extract the XPath as identified by the [SelectorGadget](https://chrome.google.com/webstore/detail/selectorgadget/mhjhnkcfbdhnjickkkdbjoemdmbfginb?hl=en) tool available as a chrome plugin (see example below).

<br>
<p align="center"><img src="/figures/selectorgadget.png" width="40%"></p>
<br>

After the raw transcripts were downloaded (~1 hour), it become clear the earlier episodes did not have the full conversational transcripts that I was expecting and what is needed for the "conversational" flow I was expecting to train the fine-tuned models on. The data was therefore subset to the oldest episode that had a full episode transcript which was from early 2022. A total of 573 episode transcripts remained which ranges from [Jan 11th 2012](https://www.econtalk.org/david-rose-on-the-moral-foundations-of-economic-behavior/) to [December 20th 2022](https://www.econtalk.org/hannah-ritchie-on-eating-local/) as of January 4th 2023. 

The second [step of the pipeline](https://github.com/ErikinBC/EconChattR/blob/main/2_process_transcripts.py) was to ensure that the dialogue was aggregated to alternate text segments between Russ and his guest (see example below), which would later form the basis of the prompt/completion.

<br>
<p align="center"><img src="/figures/russ_guest.png" width="50%"></p>
<br>

The [next step](https://github.com/ErikinBC/EconChattR/blob/main/3_prepare_training.py) of the pipeline carried out some basic string cleaning procedures including:

1. Removing repeated words
2. Making sure there was right number of spaces around punctuation
3. Removed transcript artifacts like "\[? \]" and timestamps (e.g. "12:03")
4. Removed excess white space
5. Removed special characters

At this point there was a total of 31046 unique Guest/Russ dialogue pairs. However not all ~31K prompt/completion values were informative. For example, many dialogue exchanges amounted to filler words that naturally get produced in a human conversation:

```json
{"prompt":"Okay.","completion":"But, Mike might be untouchable. We'll see."}
{"prompt":"Yeah, I always hated that.","completion":"Which part?"}
```

I imposed the following token requirements:

1. Russ' statements have to have at least 20 tokens
2. The guest's statements have to have at least 60 tokens
3. The combined token length of Russ and the guest must be less than 2000 tokens

This led to a reduced dataset of 10807 prompt/completions.


<br>

## Model training and cost

### Training set to match budget 

OpenAI has [model-specific costs](https://openai.com/api/pricing/) for both training and inference. For example the simplest model (`text-ada-001`) costs $0.0004 per 1000 tokens whilst the most sophisticated (`text-davinci-003`) costs 75x more at $0.03 per 1000 tokens. While this seems like a small price to pay, consider that the ~10K cleaned prompt/completions have a total of 4,818,101 tokens. This means that single epoch on all data prompt/completion pairs would cost $145 USD. While this would be a trivial amount for a commercial endeavor, `EconChatR` is a research project on a shoe string. 

Instead, for each model I calculated the number of tokens that would support a $10 USD training cost for 4 epochs. This amounted to a data reduction of 15%, 83%, and 98% for the Babbage, Curie, and Davinci models (with the Ada model being cheap enough to train for <$10 without a data reduction). For example, as Table 2 shows below, the Davinci model could only be trained on 83K tokens.

<p align="center"> <i> Table 2: Data reduction needed for $10 cost </i> </p>
<p align="center"> <img src="/figures/data_reduction.png" width="25%"> </p>

### Curated data choices

Because I knew the prompts I would be giving in the experiment phase, I wanted to look for examples that had key words I thought would most closely align with Russ-like responses. I came up with 12 different string matches to find:

1. bootlegger
2. baptist
3. empirical
4. skeptical
5. prairie
6. deep
7. Chesterton fence
8. decimal point
9. macroeconomist
10. macroeconomic
11. regression
12. speak/data

And then ordered each of them by the number of token from smallest largest. Then for each model, I selected the maximum number of prompt/completion pairings that had the largest number of each of the 12 string match types (see example below).

<p align="center"> <i> Training samples were chosen to maximize coverage for 12 string match types </i> </p>
<p align="center"> <img src="/figures/ordering_prompt_completions.png" width="25%"> </p>

Overall, this left a total of 10807 (Ada), 9106 (Babbage), 1514 (Curie), and 204 (Davinci) prompt/completion pairs for each of the models to be trained on for 4 epochs.

### Fine-tuning API

Training the custom GPT-3 model was [fairly simple](https://github.com/ErikinBC/EconChattR/blob/main/4_tune_models.py):

1. First, `openai.File.create`
2. Second, `openai.FineTune.create`


<br>

<a id="finetuning-results"></a>
## Fine-tuning results

After the four models were done training, we could submit questions to our custom models in the OpenAI playground by selecting the associated model. Two things quickly became clear:

1. The prompt had to be re-engineered from the baseline format to the style in which we trained it. For example "*Why is "skin in the game" a deep insight?*" became "*Russ, what do you think Nassim Taleb's idea of skin in the game? Russ Roberts responds:*". Sometimes it took a while to get the exact wording right
2. Using a temperature of 0.6 and max tokens of 250, it was clear that there was substantial 

<p align="center"> <img src="/figures/playground.png" width="25%"> </p>

For the remainder of this section I will review the model-specific answers to the questions. A csv-version of this can be [found here](https://github.com/ErikinBC/EconChattR/blob/main/output/results_finetune.csv).


### Davinci

We start with the most powerful model


<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">model</th>
<th title="Field #2">prompt</th>
<th title="Field #3">completion</th>
</tr></thead>
<tbody><tr>
<td>davinci</td>
<td>Do you mention the &quot;Bootlegger and Baptist&quot; theory often? Russ Roberts responds:</td>
<td>I don&#39;t, no. It&#39;s an--I think it&#39;s an interesting idea. It&#39;s a way of trying to understand why certain things appeal to people who may not be directly interested in them. So, you might be in favor of banning liquor sales on Sunday because you are a good Methodist and you don&#39;t like the idea of other people drinking liquor. But you realize that that&#39;s not a very good argument. So you have to come up with some other reason. And so you say, &#39;Well, it&#39;s a safety issue. It&#39;s a safety issue for workers. But you don&#39;t really care about workers&#39; safety. But you can use that as your justification. And so it&#39;s a way of allowing your pure self-interest to be disguised by some other motive that is more publicly acceptable. So, I think it&#39;s a really interesting idea. I don&#39;t know how much truth there is to it. Certainly, it&#39;s true that people sometimes vote for things that they know will benefit them personally, even though it&#39;s not in their self-interest to admit that.</td>
</tr>
<tr>
<td>davinci</td>
<td>As a proponent of free markets, are you attracted to the &quot;Bootlegger and Baptist&quot; theory? Russ Roberts responds:</td>
<td>Yeah, that&#39;s a great way to think about it. I think that&#39;s very helpful. So, the basic idea of the Bootlegger and Baptist theory is that--it&#39;s a way of thinking about the political process. So, a Bootlegger is somebody who wants to profit from some kind of regulation. So, a Bootlegger is a nightclub owner who wants to profit from liquor prohibition. So, Bootlegger is a kind of a code word for somebody who wants to profit from some kind of regulation. A Baptist is somebody who wants to advocate for that regulation for some other reason. So, a Baptist might be somebody who wants to advocate for liquor prohibition because they want people to be virtuous. So, the Bootleggers and the Baptists both want prohibition. The Bootlegger wants it for self-interested reasons; the Baptist wants it for other-interested reasons. And the idea is that Bootlegger and Baptist are both active in politics. And Bootleggers tend to be more active because they have a direct incentive to influence the political process. And Baptists tend to be less active because it&#39;s not clear that they have an immediate, self-interested reason to be active in the political process.</td>
</tr>
<tr>
<td>davinci</td>
<td>Russ, how skeptical are you of empirical research published in academic journals? Russ Roberts responds</td>
<td>Well, I&#39;m very skeptical. I think--I&#39;m going to get myself in trouble here. I think the incentives are all wrong. I think it&#39;s very difficult to do good empirical work. I think there are a lot of mistakes that are made over and over again. And I&#39;m very skeptical of a lot of what gets published in academic journals. I think there&#39;s a lot of--the incentives are toward getting published and getting tenure and promotion. And I think that leads to a lot of bad research. It&#39;s not necessarily fraudulent--although there&#39;s some of that--but it&#39;s--a lot of times it&#39;s just--it&#39;s the wrong answers to the wrong questions. It&#39;s--I&#39;m a fan of [John] Ioannou&#39;s work on this. It&#39;s--you know, the--if you want to make sure that someone is a good researcher, take them out of the university and put them on a desert island with nothing to do but research. And if they come back with a result, you&#39;ll know they&#39;re a fraud. You know, they&#39;ve had nothing to do but think about this one thing. The--I think a lot of the problems in economics are that people with tenure and promotion</td>
</tr>
<tr>
<td>davinci</td>
<td>Russ, are you skeptical that &quot;data can speak for itself&quot; as a general principle? Russ Roberts responds:</td>
<td>I&#39;m very skeptical. I think it&#39;s a lovely idea; it&#39;s very appealing. It&#39;s a very American idea, that the truth will prevail and the truth is the data. And it&#39;s very appealing to everyone because when the data goes your way, you can say, &#39;Look, the data showed it, so you can&#39;t deny it. And when the data doesn&#39;t go your way, you can say, look, the data is wrong. So, I&#39;m deeply skeptical of that. I think that&#39;s a lovely phrase, but I don&#39;t think it&#39;s a very good guide. In particular, I think it&#39;s very--I have a phrase: &#39;Bad data, wrong conclusions. Because the data can be wrong. And the conclusions can be wrong. And it&#39;s very hard to know which is which.</td>
</tr>
<tr>
<td>davinci</td>
<td>How would Russ Roberts finish the following sentence from Adam Smith: &quot;Man desires not only to be loved</td>
<td>, but to be lovely. . . . &#39;&#39;</td>
</tr>
<tr>
<td>davinci</td>
<td>Finish the following sentence by FA Hayek: &quot;The curious task of economics</td>
<td>is to demonstrate to men how little they really know about what they imagine they can design&quot;.</td>
</tr>
<tr>
<td>davinci</td>
<td>Russ, what do you think Nassim Taleb&#39;s idea of skin in the game? Russ Roberts responds:</td>
<td>Well, I think it&#39;s a very deep idea. I think it&#39;s a very deep idea for a lot of reasons. One of which is that it&#39;s very related to our conversation today about the value of information. Because what he is saying is that if you are in a position of giving advice or making decisions, you should have some of the consequences of those decisions. So, if you are an investment banker giving advice to a company about, say, an acquisition, you should have some of your own money at stake in that decision. If you are a policymaker making decisions about the future of the country, you should have some of your own wealth at stake. And so, I think that&#39;s a very deep idea. I think it&#39;s true. I think it&#39;s very--it&#39;s a very challenging idea. And I think it&#39;s very--it&#39;s a very powerful idea. Because, of course, it means that a lot of people who are making decisions don&#39;t have skin in the game. And that&#39;s very--that&#39;s a very bad thing.</td>
</tr>
<tr>
<td>davinci</td>
<td>In what way does understanding the challenge around &quot;creating a prairie&quot; align with F A Hayek&#39;s famous maxim about the curious task of economics? Russ Roberts responds:</td>
<td>Well, Hayek&#39;s maxim is that &quot;the curious task of economics is to demonstrate to man how little he really knows about what he imagines he can design. And I think that is true of policy generally. Policy is a process of trying to orchestrate things that we don&#39;t understand very well. And prairie restoration is a good example of that. We don&#39;t understand how prairies come about. We don&#39;t understand the processes that created them in the past, so we don&#39;t really know how to recreate them. And so it&#39;s interesting to do it as an experiment. It&#39;s interesting to try to understand what&#39;s going on. But it&#39;s a very, very difficult challenge. And I think that&#39;s true of most policy. It&#39;s certainly true of macroeconomic policy. It&#39;s true of international policy. It&#39;s very, very hard to orchestrate things that we don&#39;t understand very well.</td>
</tr>
<tr>
<td>davinci</td>
<td>Why is it harder to &quot;create a prairie&quot; then it seems? Russ Roberts responds:</td>
<td>Well, I think the biggest challenge is that it&#39;s a complex system. And, you know, we&#39;ve talked a lot on this program about how economists tend to overuse the word &#39;system&#39; and they tend to overuse the word &#39;complex. But, it&#39;s really hard to create a prairie, because prairies are very highly adapted to the climate and the soil and the particular species that are there. And, you can&#39;t just put the seeds in the ground and expect to get a prairie. They are a complex interaction of many different species of plants and animals that have evolved together. And, prairies are a classic example of what we now call an &#39;emergent system. And, an emergent system is one where the whole is greater than the sum of its parts. And, prairie is a classic example of that. And, prairie is also a classic example of what we now call a &#39;complex adaptive system. And, complex adaptive systems, we&#39;ve talked about on the program before with Nassim Taleb and others. But, complex adaptive systems are systems where the initial conditions are so important that you can&#39;t just start over. You can&#39;t just reset the system to the starting point and expect</td>
</tr>
<tr>
<td>davinci</td>
<td>As a free market economist, why are you drawn to the idea of the &quot;Chesterton Fence&quot;? Russ Roberts responds:</td>
<td>Well, the Chesterton fence is a fence that doesn&#39;t do what it was intended to do, but it has an unintended consequence that&#39;s good. And the idea is that Chesterton is talking about someone who wants to fence off their field. And Chesterton says, &#39;Well, why do you want to do that? It&#39;s a good idea. It&#39;s a fence; it keeps the cows in. But what about the cows that are on the other side of the fence? They can&#39;t get in. So, Chesterton&#39;s point is that we need to sometimes look at the unintended consequences of our actions. And, the problem I have with most regulations is that the people who are harmed by those regulations are not the same as the people who are helped. And, it&#39;s very hard to see the benefits. If you look at a regulation, like the Dodd-Frank legislation that was passed in the United States, there are people who are helped by that legislation. There are people who are working in the industry who are benefiting from that legislation. But, there are a lot of other people who are hurt by it. And, it&#39;s hard to see the benefits. And, so, I&#39;m a big fan of the idea that</td>
</tr>
<tr>
<td>davinci</td>
<td>You often say that economists have a sense of humor because they use decimal points, why is that? Russ Roberts responds:</td>
<td>Because we are deeply inadequate in our ability to understand the real world. And the decimal point gives us a sense that we are being precise when we really have no idea what we are talking about. The decimal point allows us to create the illusion of knowledge where there is only ignorance. And so, whenever I--we economists are very prone to say something like, &#39;Well, we have a sense of humor because we use decimal points. I don&#39;t have a sense of humor because I use decimal points. I have a sense of humor because I&#39;m in a profession that is deeply inadequate to describe the world as it is. And so that&#39;s the joke. And I think the joke is on me. And I think the joke is on my profession. And I think that&#39;s a good thing. I think it&#39;s healthy to acknowledge that we are in some sense in the dark. We are groping. We are stumbling. We are making things up. We are using our intuition. We are using our experience. But we are not doing--it&#39;s not a science. It&#39;s not like physics. And I think it&#39;s healthy to be honest about that.</td>
</tr>
<tr>
<td>davinci</td>
<td>Who are some guests you interviewed in 2014? Russ Roberts responds:</td>
<td>Well, we interviewed Arnold Kling, who is a regular. He&#39;s a frequent commentator on the EconTalk podcast. We interviewed Laura Tyson, who is a very smart economist. She was the Chair of the Council of Economic Advisers in the Clinton Administration. We interviewed Gary Becker, who is an old friend of mine and a Nobel Prize winner in economics. We interviewed Barry Lynn, who is the head of the New America Foundation, which is a think tank here in Washington. And we interviewed David Henderson, who is a regular, and a very good economist.</td>
</tr>
</tbody></table>

<br>
<br>

***

<br>

### Footnotes

[^1]: Here are some fairly emblematic headlines that have come out on social and traditional media platforms: "[Google is done. Here’s why OpenAI’s ChatGPT Will Be a Game Changer](https://medium.com/@lucapetriconi/google-is-done-heres-why-openai-s-chatgpt-will-be-a-game-changer-98ae591ad747)", "[ChatGPT is a game changer for artificial intelligence](https://www.ctvnews.ca/video?clipId=2585069)", "[ChatGPT is a GAME CHANGER!](https://www.reddit.com/r/algotrading/comments/zkdi3e/chatgpt_is_a_game_changer/)".

[^2]: The model has ~170B parameters and was trained on about 300 billion tokens (where 5 tokens averages about 4 words).

[^3]: It would [likely take](https://twitter.com/tomgoldsteincs/status/1600196981955100694?lang=en) 8 high-end GPUs to run a model of this size on a local machine (each GPU would cost ~$15K). Furthermore, it has been [estimated](https://lambdalabs.com/blog/demystifying-gpt-3) that GPT-3 would have cost $5 million USD  to train and One [popular estimate](https://twitter.com/tomgoldsteincs/status/1600196995389366274) has pegged the cost of running ChatGPT at $3 million USD a month.
