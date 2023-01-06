---
title: "(DRAFT) Fine tuning GPT3 to soud like podcast host"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

## Executive summary

I developed four fine-tuned versions of OpenAI's [GPT-3 models](https://beta.openai.com/docs/models/gpt-3) to sound like [Russ Roberts](https://russroberts.info/) who is the host of [EconTalk](https://www.econlib.org/econtalk/) using more than 10 years of episode transcripts. I calibrated my experiments for a $40 USD budget ($10 per model) and tested the fine-tuned models on 12 quintessential Russ/EconTalk questions.

This process led to four key insights:

1. [ChatGPT](https://chat.openai.com/) gives decent answers to EconTalk questions without any fine-tuning ([~50% accuracy](#baseline-results)).
2. A fine-tuned `text-davinci-003` model gives **even better results** (80% accuracy), and answers questions with impressive **Russ-like diction** when trained on only 83K tokens (see [examples](#davinci-examples)).
3. However, all models require prompt engineering and repeated querying to get the "best" answer.
4. The `curie` model has poor poor zero-shot accuracy, and doesn't do much better after fine-tuning, although its way of speaking has a podcast-like feel.
5. The `ada` and `babbage` models give nonsensical answers to most prompts even with training on the entire corpus (4-5 million tokens).
6. With OpenAI's pricing, the scaling laws clearly favor model size over data (e.g. the `curie` model fine-tuned on 10x data but with 1/10th of the parameter size does much worse).
7. Overall, a fine-tuned `davinci` model with more resources (the one used in this experiment was trained on <2% of the corpus for cost reasons) would likely provide a good impersonation of the style and type of answers a podcast host would give. 

The rest of this post provides more details on the development of this model and the experimental results. Interested readers can create their own versions of [`EconChatR`](https://github.com/ErikinBC/EconChattR) by cloning this repo and running the main pipeline. 

<br>

## Background

People are [very excited](https://techmonitor.ai/technology/ai-and-automation/chatgpt-openai-chatbot) about [ChatGPT](https://chat.openai.com/chat).[[^1]] ChatGPT is the 3rd generation of large language models ([LLMs](https://en.wikipedia.org/wiki/Wikipedia:Large_language_models)) from [OpenAI](https://openai.com/), GPT 3.5, that has been able to produce human-like performance for a variety of natural language tasks. LLMs use many multi-headed attention layers (a type of architecture [first proposed](https://arxiv.org/abs/1706.03762) by Google researchers in 2017) which are more amenable to parallelization and hence GPU acceleration. The `davinci` version of GPT-3.5 (which powers ChatGPT) is very large,[[^2]] and would be prohibitive to run on a local machine.[[^3]] 

From what OpenAI has [published](https://arxiv.org/abs/2005.14165), GPT-3 was trained on a large corpus of mainly English sources using a very simple training framework: predict the next token given a context. What's fascinating is how such a simple [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) training process yields such impressive results on a variety of tasks from language translation to summarization to logical reasoning. If you wanted to [re-create ChatGPT](https://github.com/lucidrains/PaLM-rlhf-pytorch) from scratch you would [likely need](https://lifearchitect.ai/chatgpt/) a supercomputer cluster with ~10,000 GPUs and ~285,000 CPU cores (~$1 billion USD to rent), spend months training the model, and then spend months/year labeling the output of the model to further hone its abilities. One of OpenAI's "secret ingredients" is an unspecified amount of resources spent doing Reinforcement Learning from Human Feedback ([RLHF]((https://openai.com/blog/instruction-following/))). Basically this means having humans interact with the ChatBot, and label output as either correct/incorrect or rank-order multiple generations from the same prompt. The model models are then further refined using this supervised learning framework.

ChatGPT has shown amazing performance in its ability to [write code](https://twitter.com/gabe_ragland/status/1598068207994429441), [write academic papers](https://icml.cc/Conferences/2023/llm-policy), [answer most homework questions](https://www.nbcnews.com/tech/tech-news/new-york-city-public-schools-ban-chatgpt-devices-networks-rcna64446), [write entire syllabi](https://oneusefulthing.substack.com/p/the-mechanical-professor), and [come up decent rhyming poems](https://lil.law.harvard.edu/blog/2022/12/20/chatgpt-poems-and-secrets/). 


<p align="center"><img src="/figures/tomato_poem.png" width="25%"></p>
<p align="center"><i>ChatGPT's cherry tomato poem is pretty good!</i></p>


## Introduction to fine-tuning

Even with unprecedented scale, training, and RLHF fine-tuning, ChatGPT still shows a tendency to "[hallucinate](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))", giving nonsensical answers, made-up facts, and repeated sentences. One way to address this issue is to fine-tune a version of GPT-3 on a custom dataset that updates the models weights in a way that makes more likely to give answers consistent with the task you have in mind. For example, we may want the model to give medically specific and accurate answers to [questions related to depression](https://betterprogramming.pub/how-to-finetune-gpt-3-finetuning-our-virtual-mental-health-assistant-641c1f3b1ef3) (although this might be quite legally/ethically risk!). This can be done with a new collection of prompt/completion data points:

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

After we've created a fine-tuned model we should be able to ask our model questions like: "*You often say that economists have a sense of humor because they use decimal points, why is that? Russ Roberts responds:*", with the prompt suffix "reminding" our model the sort of dialogue we expect to receive back. 


<a id="baseline-results"></a>
### Baseline results

Before spending money to fine-tuning models, I subjectively evaluated how well ChatGPT does "out of the box" (i.e. zero-shot) on 12 quintessential Russ'isms and EconTalk facts (inspired by the [EconTalk drinking game](http://www.salanwoodbine.com/econtalkdrinkinggame/)):

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


As Table 1 shows below that out of the box ChatGPT does well for half of the questions 1, 2, 6, 7, 8, and 10. I would give this model a score of 6/12 (50%). Questions 5 and 11 shows that the model is hesitant to give answers on behalf of specific people, and has a surprisingly hard time remembering the Adam Smith's quote. The model also provides inaccurate answers for question 12, providing some real EconTalk guests, but not necessarily those who were there in 2014 (for example Ben Bernanke has not been a guest, and Tyler Cowen was on EconTalk in 2013/2017 but not 2014), and strangely adding that Russ was a guest on his own podcast. All this suggests that there is room for improvement with a fine-tuned model to give more accurate and Russ-like answers. 

<br>

<p align="center"> Table 1: Completions to baseline prompts </p>
<table align="center">
  <tr>
    <td><img src="/figures/q1_econtalk.png" alt="q1" width=480px></td>
    <td><img src="/figures/q2_econtalk.png" alt="q2" width=480px></td>
  </tr>
  <tr>
    <td><img src="/figures/q3_econtalk.png" alt="q3" width=480px></td>
    <td><img src="/figures/q4_econtalk.png" alt="q4" width=480px></td>
  </tr>
  <tr>
    <td><img src="/figures/q5_econtalk.png" alt="q5" width=480px></td>
    <td><img src="/figures/q6_econtalk.png" alt="q6" width=480px></td>
  </tr>
  <tr>
    <td><img src="/figures/q7_econtalk.png" alt="q7" width=480px></td>
    <td><img src="/figures/q8_econtalk.png" alt="q8" width=480px></td>
  </tr>
  <tr>
    <td><img src="/figures/q9_econtalk.png" alt="q9" width=480px></td>
    <td><img src="/figures/q10_econtalk.png" alt="q10" width=480px></td>
  </tr>
  <tr>
    <td><img src="/figures/q11_econtalk.png" alt="q11" width=480px></td>
    <td><img src="/figures/q12_econtalk.png" alt="q12" width=480px></td>
  </tr>
</table>


<br>

## Data and processing

The first step in any data science project is gather and clean a dataset and then properly format it. A full list of EconTalk episodes was obtained from the [xml file](http://www.econlib.org/library/EconTalk.xml) hosted on EconLib. A [simple web-scraper](https://github.com/ErikinBC/EconChattR/blob/main/1_scrape.R) was then used to extract the XPath as identified by the [SelectorGadget](https://chrome.google.com/webstore/detail/selectorgadget/mhjhnkcfbdhnjickkkdbjoemdmbfginb?hl=en) tool available as a chrome plugin (see example below).

<br>
<p align="center"><img src="/figures/selectorgadget.png" width="40%"></p>
<p align="center"><i>EconTalk episodes transcripts can be obtained from the episode page</i></p>
<br>

After the raw transcripts were downloaded, it become clear that the earlier episodes did not have verbatim transcripts and hence could not be used for the prompt/completion style formatting. The data was therefore subset to the oldest episode that had a full episode transcript which was from early 2012. A total of 573 episode transcripts remained which ranged from [Jan 11th 2012](https://www.econtalk.org/david-rose-on-the-moral-foundations-of-economic-behavior/) to [December 20th 2022](https://www.econtalk.org/hannah-ritchie-on-eating-local/) (as of January 4th 2023). 

The second [step of the pipeline](https://github.com/ErikinBC/EconChattR/blob/main/2_process_transcripts.py) was to ensure that the dialogue was aggregated to alternate text segments between Russ and his guest (see example below), which would later form the basis of the prompt/completion.

<br>
<p align="center"><img src="/figures/russ_guest.png" width="50%"></p>
<br>

The [next step](https://github.com/ErikinBC/EconChattR/blob/main/3_prepare_training.py) of the pipeline carried out some basic string cleaning procedures including:

1. Removing repeated words
2. Making sure there was right number of spaces around punctuation
3. Removing transcript artifacts like "\[? \]" and timestamps (e.g. "12:03")
4. Removing excess white space
5. Removing special characters

At this point there was a total of 31046 unique Guest/Russ dialogue pairs. However not all ~31K prompt/completion values were informative. For example, many dialogue exchanges amounted to a handful of words which had no information without the broader context as the two examples show below.

```json
{"prompt":"Okay.","completion":"But, Mike might be untouchable. We'll see."}
{"prompt":"Yeah, I always hated that.","completion":"Which part?"}
```

For a prompt/completion pair to be kept, I imposed the following token requirements:

1. Russ' statements had to have at least 20 tokens
2. The guest's statements had to have at least 60 tokens
3. The combined token length of Russ and the guest must be less than 2000 tokens

This led to a reduced dataset of 10807 prompt/completions.


<br>

## Model training and cost

### Training set to match budget 

OpenAI has [model-specific costs](https://openai.com/api/pricing/) for both training and inference. For example the simplest model (`text-ada-001`) costs $0.0004 per 1000 tokens whilst the most sophisticated (`text-davinci-003`) costs 75x more at $0.03 per 1000 tokens. While this seems like a small price to pay, consider that the ~10K cleaned prompt/completions have a total of 4,818,101 tokens. This means that single epoch on all data prompt/completion pairs would cost $145 USD for the most expensive option. While this would be a trivial amount for a commercial endeavor, `EconChatR` is a research project on a shoe string budget. 

Instead, for each model I calculated the number of tokens that would support a $10 USD training cost for 4 epochs. This amounted to a data reduction of 15%, 83%, and 98% for the `babbage`, `curie`, and `davinci` models (with the `ada` model being cheap enough to train for <$10 without a data reduction). For example, as Table 2 shows below, the `davinci` model could only be trained on 83K tokens.

<p align="center"> <i> Table 2: Data reduction needed for $10 cost </i> </p>
<p align="center"> <img src="/figures/data_reduction.png" width="25%"> </p>

### Curated data choices

Because I knew the prompts I would be giving in the experiment phase, I wanted to look for dialogue pairs that had key words I thought would most closely align with a Russ-like responses. I came up with 12 different string matches to find:

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

Each string match was associated with list of conversation pairs, and these were then ordered by the number of token from smallest largest. For each model, I then iteratively selected the maximum number of prompt/completion pairings that had the largest number of each of the 12 string match types (see example below). This also ensured that larger models used a data subset of smaller models. In other words, all data points found in `davinci` are found in `curie`, and all data points found in `curie` were also in the training set of `babbage`, etc.

<p align="center"> <img src="/figures/ordering_prompt_completions.png" width="25%"> </p>
<p align="center"> <i> Training samples were chosen to maximize coverage for 12 string match types </i> </p>

Overall, this left a total of 10807 (Ada), 9186 (Babbage), 1514 (Curie), and 204 (Davinci) prompt/completion pairs for each of the models to be trained on for 4 epochs.[[^4]]

### Fine-tuning API

Training the custom GPT-3 model was [fairly simple](https://github.com/ErikinBC/EconChattR/blob/main/4_tune_models.py):

1. First, the model-specific dataset was uploaded to OpenAI using the `openai.File.create` command.
2. Second, a fine-tuning call was made for each of the models and their associated dataset using the `openai.FineTune.create` command.

As expected the cost of each model was less than $10. Furthermore, the queue to train the `davinci` and `curie` models was very short and training was done in less than an hour for each of them. The `ada` and `babbabe` models took longer both because of a longer queue and a larger training set (although the total time was less than 2 hours for each).


<br>

## Fine-tuning results

After the four models were done training, prompts could be submitted the custom models in the OpenAI playground by selecting the associated model (alternatively the python API could have been used). Two things quickly became clear:

1. The prompt had to be re-engineered from the baseline format to the style in which we trained it. For example "*Why is "skin in the game" a deep insight?*" became "*Russ, what do you think Nassim Taleb's idea of skin in the game? Russ Roberts responds:*". Some questions required up to 10 tries to get the exact wording right.
2. Using a temperature of 0.6 and max tokens of 250, it was clear that there was substantial variation in the types of answers the model would provide. This suggests a value in generating dozens of responses, and selecting the one that most closely aligns with the EconTalk corpus (for example, see [this approach](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) on using embeddings). 

<br>
<p align="center"> <img src="/figures/playground.png" width="20%"> </p>
<p align="center"> <i> After fine-tuning, your models will appear on the OpenAI playground </i> </p>
<br>

For the remainder of this section I will review the model-specific answers to the questions. A csv-version of this can be [found here](https://github.com/ErikinBC/EconChattR/blob/main/output/results_finetune.csv).


<a id="davinci-examples"></a>
### davinci

Starting with the largest model, we see it answers questions with a "transcript" like feel to it; using incomplete words, double dashes, and starting sentences with conjunctions. The first two questions are about how often and why Russ brings up the [Bootlegger and Baptist](https://en.wikipedia.org/wiki/Bootleggers_and_Baptists) concept. While the fine-tuned model does a great job at explaining the concept in a way Russ might, it does not answer the question directly. 

For the third question, the model does a very good job at explaining Russ' skepticism of empirical research. However, the last few sentences after John Ioannidis (misspelled in answer) are a little confused. Russ' answer to the "data can speak for itself" is also very good, although the last sentence is a bit confusing. On the fourth/fifth questions, the model is able to accurately remember the Adam Smith and FA Hayek quotes. Huzzah!

I love Russ' answer to the sixth question about "skin in the game" where it gives both a "chatty" answer, but also calls out EconTalk's favorite cast of characters including investment bankers. Russ' answers to the seventh and eighth question about the "prairie" are also very good, although the latter answer runs on a bit at the end. Unfortunately Russ' answer to the tenth question about the Chesterton fence was quite poor. While it does mention "unintended consequences," it is quite far off the mark, and is the only question that is materially worse than the baseline ChatGPT.

Overall, I'd give a zero marks to questions 10 & 12, and part marks to question 2, giving the model a (subjective) score of 9.5/12 (~80%). The model also deserves bonus points for giving a convincing "conversational" feel to the answers in a way that would be hard to distinguish between a real EconTalk transcript.  


<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">model</th><th title="Field #2">prompt</th><th title="Field #3">completion</th></tr></thead>
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
<td>Russ, how skeptical are you of empirical research published in academic journals? Russ Roberts responds:</td>
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

### curie

The `curie` model performed [poorly](https://github.com/ErikinBC/EconChattR/blob/main/output/results_baseline.csv) in baseline experiments. However, because the model is about 1/10th of the cost of the `davinci`, it was given significantly more data during its training (830K vs 83K tokens). Unfortunately this extra data did translate to better answers. As the table shows below, the fined-tuned version of the model does give convincing sounding sentences, but the content is itself quite poor. The only questions the model was able to do well at was completely the Adam Smith quote.

<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">model</th>
<th title="Field #2">prompt</th>
<th title="Field #3">completion</th>
</tr></thead>
<tbody><tr>
<td>curie</td>
<td>Do you mention the &quot;Bootlegger and Baptist&quot; theory often? Russ Roberts responds:</td>
<td>I don&#39;t think so. I used it in the book. It&#39;s a good example of the way that economics and social science and history have a lot in common. It&#39;s often said that the Bootlegger and Baptist theory is the attempt to justify what the politicians are doing. And I think it&#39;s the other way around. I think it&#39;s the politicians who justify what they want to do by appealing to the Bootlegger and Baptist theory. In a way, it&#39;s a way to make yourself feel good about what you&#39;re doing. And it&#39;s really hard to fight against it. It&#39;s hard to say, &#39;Well, I don&#39;t want to do this because it&#39;s bad for the poor. I think that&#39;s a really good argument. But most of us don&#39;t think that way. Most of us think, &#39;Well, it&#39;s not so bad on the poor. We&#39;ll take care of them. We&#39;ll give them a handout. We&#39;ll have a tax cut for them. We&#39;ll help them. We&#39;ll do this. And the politicians, of course, are aware of that. They appeal to that. And I think it&#39;s a very deep and powerful phenomenon.</td>
</tr>
<tr>
<td>curie</td>
<td>As a proponent of free markets, are you attracted to the &quot;Bootlegger and Baptist&quot; theory? Russ Roberts responds:</td>
<td>Well, I&#39;m not a big fan of the Bootlegger and Baptist theory, but I&#39;m not sure it&#39;s a good theory either. I think the empirical evidence that--I&#39;m not sure it&#39;s true. I think it&#39;s true that some people are more likely to give money to a politician that they think is going to help their own self-interested, ideological, ideological self. I think it&#39;s true that there are people out there who are crony capitalists. I think it&#39;s certainly true that there are people out there who are ideological, who are more likely to support politicians that are ideologically like them, and so on. I think there&#39;s some truth to the idea that some people are more likely to support politicians that are ideologically like them. And I think it&#39;s true that there are people out there who are crony capitalists. I think it&#39;s certainly true that there are people out there who are ideological, who are more likely to support politicians that are ideologically like them, and so on. I think there are real issues there. I&#39;m not sure it&#39;s the right theory, but I think it&#39;s true.</td>
</tr>
<tr>
<td>curie</td>
<td>How skeptical are you of empirical research published in academic journals? Russ Roberts responds:</td>
<td>I&#39;m skeptical of most of it. I think there&#39;s some really good work that gets published in those journals. But I think there&#39;s a lot of work that&#39;s just not very good. I think the peer review process is very imperfect. I think that&#39;s true in economics as well as other fields. I think that&#39;s true in psychology as well; I think there&#39;s a lot of psychological research that gets published that is not of very much value. I find it a little bit of an insult to my intelligence to be honest. I can&#39;t imagine sitting in the armchair and feeling, &#39;Well, this was a good study, this was a bad study. I think a lot of the studies are flawed. I think it&#39;s very hard to tease out the true effect of a policy change in, say, a randomized control trial, that&#39;s where I think a lot of the work gets published. I think the sample is not large enough, I don&#39;t think the controls are good enough. I think there&#39;s a lot of mistakes in the analysis and the cherry picking of the data. I think it&#39;s a very imperfect science. I think the peer review process is not good. I think the journals are not good publications--except</td>
</tr>
<tr>
<td>curie</td>
<td>Do you believe that &quot;data can speak for itself&quot; as a general principle? Russ Roberts responds:</td>
<td>Well, I&#39;m skeptical about that. I think the question is: Is it true? So, you know, the question is: Do we have enough evidence to demonstrate that one intervention works better than another? And I think we need to be honest about that. And I think we need to be honest about the fact that there are going to be biases in our data sets. We need to be careful about how we try to control for those. So, I have some--I&#39;m a big fan of experimental economics. And I think experimental economics is very important. I don&#39;t think randomized control trials are the be-all and end-all. I think we need to be honest about the fact that what works in one place is not necessarily going to work in another place. So, I think there&#39;s a lot of issues that we&#39;ve got to be honest about. I&#39;m not a big fan of the phrase &#39;data can speak for itself. That&#39;s a phrase that sometimes comes up in the literature. And I understand the appeal of it. I think it&#39;s true--that if you have a large enough sample, you can make strong claims about the average impact of an intervention. But the phrase itself--I think it&#39;s over</td>
</tr>
<tr>
<td>curie</td>
<td>How would Russ Roberts finish the following sentence from Adam Smith: &quot;Man desires not only to be loved</td>
<td>but to be lovely. &quot;</td>
</tr>
<tr>
<td>curie</td>
<td>Finish the following sentence by FA Hayek: &quot;The curious task of economics</td>
<td>is to demonstrate to men how little they can be relied upon to know about what they imagine they know. &quot;</td>
</tr>
<tr>
<td>curie</td>
<td>Why do you think Nassim Taleb&#39;s &quot;skin in the game&quot; is a deep thought? Russ Roberts responds:</td>
<td>Well, I don&#39;t know about that, but it&#39;s a nice thought. I think it goes back to the issue of self-interest versus public interest. The idea of living in a system where I&#39;m not insulated from the consequences of my actions and I have to take a little bit of risk--it&#39;s a very deep idea that I think is very powerful. And I think it&#39;s very difficult to implement. I think it&#39;s a nice thought--but I don&#39;t know if it&#39;s a good idea. It&#39;s hard to know.</td>
</tr>
<tr>
<td>curie</td>
<td>In what way does understanding the challenge around &quot;creating a prairie&quot; align with F A Hayek&#39;s famous maxim about the curious task of economics? Russ Roberts responds:</td>
<td>Well, I think there&#39;s a couple of lessons in there for us. I think the first is that a lot of what we think of as our knowledge, our understanding, is not just false but of limited use. So, when we think we have a grasp on some particular issue, we often are just fooling ourselves. And I think that&#39;s a very important lesson. I think the second lesson is that we have to be humble enough to realize that we don&#39;t have the answers, and we have to be open enough to realize that we may be wrong. And I think that&#39;s very important. But the third part of that is, I think, the most important part. And that is that we have to be willing to change our mind. And I think one of the hallmarks of a good economist is that you are open to that. I think most economists I know are very skeptical about their own knowledge and the knowledge of their colleagues. And I think that&#39;s a good thing. But I think the reason you have to be open to that is because the world changes. And what you thought was true yesterday may not be true today. And it&#39;s good to be open to that.</td>
</tr>
<tr>
<td>curie</td>
<td>Why is it harder to &quot;create a prairie&quot; then it seems? Russ Roberts responds:</td>
<td>Well, that&#39;s a really interesting question. And, I think part of the answer is that it is harder to create a prairie. It&#39;s hard to convert grassland into prairie. It&#39;s not that hard to convert prairie into a field again. But, that&#39;s not what we&#39;re talking about here. The question is, why is it hard to create a prairie? Why is it hard to create a prairie that has a different--a different ecological function? And, I think the answer to that is that, as you said, it&#39;s not just the grasses and the prairie. It&#39;s the soil, it&#39;s the insects, it&#39;s the wildlife. It&#39;s a whole ecosystem that comes together to make a prairie. And, the fact that it&#39;s more complicated is really the reason that the prairie has survived for so long.</td>
</tr>
<tr>
<td>curie</td>
<td>As a free market economist, why are you drawn to the idea of the &quot;Chesterton Fence&quot;? Russ Roberts responds:</td>
<td>Well, there are two parts to that. One is that, as an economist, I&#39;m skeptical of the ability of government to solve problems. So, I&#39;m very much in favor of limited government. And the other part of that is that I&#39;m skeptical of what I call &#39;redistribution. So, the idea that government would take money from one group of people and give it to another group of people is deeply, deeply troubling to me. So, the question is: How do we justify that? How do we justify the fact that we have a safety net? How do we justify that the government has the power to take money from people who are rich and give it to people who are poor? And I think the answer--it&#39;s a very good question, and I think the answer is that it&#39;s not just the poor who are poor. It&#39;s the rich too. It&#39;s the middle class. And they suffer the same challenges. And so, the idea that we would do that, I think, is immoral. It&#39;s a bad idea. And I think it&#39;s deeply immoral. So, that&#39;s the first part of the story. The second part of the story is that, as I said, I think the</td>
</tr>
<tr>
<td>curie</td>
<td>Do macroeconomists have a sense of humor? Russ Roberts responds:</td>
<td>Well, I&#39;m an economist, so I have a sense of humor. But I&#39;m not the only one. I&#39;m not the only one. I think the profession is rather grim. I think we have a lot of good empirical work being done. I think we&#39;re much more aware of the data than we were a decade ago. I think we have a much better understanding of what&#39;s going on around the world. But I think the macro side of economics is still stuck in a time warp. We&#39;re still romanticizing the gold standard, interest rates, Say&#39;s Law. And we have a hard time understanding the role of monetary policy in the current environment. I think we have a long way to go.</td>
</tr>
<tr>
<td>curie</td>
<td>Who are some guests from EconTalk from the year 2014? Russ Roberts responds:</td>
<td>Well, I think my favorite guest of the year is probably Jim Otteson, who is a professor at the University of Georgia. And, he and I have had a number of conversations on the EconTalk site, and he&#39;s a very interesting person. And, I think the other person I&#39;d mention is my friend and co-host, Dave Feldman. Dave is a very interesting economist. We&#39;ve had some conversations over the years, and he&#39;s a great guest. I think my guests from the year 2014 are Jim Otteson and Dave Feldman. Russ Roberts responds: I&#39;m going to add one more. I&#39;m going to add one more guest, which is Michael Munger. Michael Munger is a very, very, very smart person. He&#39;s a philosopher, but he also has a deep understanding of economics. And, our conversation on his book, What&#39;s So Great About Christianity, is a classic.</td>
</tr>
</tbody></table>


<br>


### Ada/Babbage

The two smallest models were trained on most/all of the corpus with laughably bad results. The models will latch on to one or two words from the prompt and then provide a bizarre riff that sounds nothing like Russ or a conversation on the podcast. I've included a handful of fun, but bad, examples below.

<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">model</th>
<th title="Field #2">prompt</th>
<th title="Field #3">completion</th>
</tr></thead>
<tbody><tr>
<td>ada</td>
<td>Do you mention the &quot;Bootlegger and Baptist&quot; theory often? Russ Roberts responds:</td>
<td>I do. And I think it&#39;s an interesting point. The Baptist view was that the invitation to the Baptist Church was to help the poor; to help the poor; to help the poor. And then when the Lord showed the way, they disappeared. The Bootleggers, who were the &quot;strong-minded sort, &quot; said: Well, no, it&#39;s okay. It&#39;s okay. It&#39;s okay. They can just go and help the poor. And they did. But then when the Lord showed the way, they didn&#39;t. And that&#39;s because they were not ready for it.</td>
</tr>
<tr>
<td>babbage</td>
<td>As a proponent of free markets, are you attracted to the &quot;Bootlegger and Baptist&quot; theory? Russ Roberts responds:</td>
<td>Well, I think there are many people who are attracted to the Bootlegger and Baptist theory, for reasons that we&#39;ve talked about before. And I think that&#39;s a great explanation for why there&#39;s so much religious activity in the United States. And I think it&#39;s a beautiful theory. I think it&#39;s true. But when I look at the world in which we live, I think what&#39;s true is that there&#39;s a lot of religious activity in the world, but it&#39;s not all religious activity. And I think it&#39;s important to think about what&#39;s going on. I think the religious activity that&#39;s going on, which is so much of it, is not religious activity of the sort that you&#39;re talking about. It&#39;s not a religious revival. It&#39;s not the Pentecostal phenomenon. It&#39;s not the Buddhist phenomenon. It&#39;s not the Hare Krishna phenomenon. It&#39;s not the Scientology phenomenon. It&#39;s just a lot of people coming together who have a desire to be close to God, who have a desire to be close to other people, and to have a community of people that they can be close to. And that&#39;s a wonderful thing. It&#39;s not a bad thing. It&#39;s not a</td>
</tr>
<tr>
<td>ada</td>
<td>Is Russ Roberts skeptical of empirical research published in academic journals?</td>
<td>Yes. I think it&#39;s important to be skeptical of the published results of empirical work, and not just accept them as the result of randomized control trials, where we choose a group and then we test it in a way that we know of, but it&#39;s an experimental trial. So, I think, empirical work, especially in economics, it&#39;s very difficult in economics to take a side. And, I think it&#39;s a little bit of a race. It&#39;s a little bit of a tug of war. And, I think it&#39;s a little bit of a tug of war because there&#39;s this intellectual tug of war between the empirical study that&#39;s published in the journal of economics or in the academic journals--we don&#39;t know the truth about how this works, the empirical study--and then the rational choice economists, who then publish that result in the journal or in the academic journal. And, it&#39;s not just a tug of battle. It&#39;s a tug of war. And, as a result, I think it&#39;s an incredibly difficult thing to know. So, I&#39;m not as skeptical as I might be of empirical work and I think it&#39;s a little bit of a tug of war. I think we have a very weak grasp of what&#39;s going</td>
</tr>
<tr>
<td>babbage</td>
<td>Do you believe that &quot;data can speak for itself&quot; as a general principle? Russ Roberts responds:</td>
<td>My answer to that is: There&#39;s a lot of data in the world. And, I think there&#39;s a lot of things that--I think of it as a way to be skeptical about the data. I think the challenge is that, the data is complicated. It&#39;s not just, &#39;I saw a spike in unemployment and therefore the recession is over. It&#39;s not just, &#39;I saw a decline in inflation and therefore the economy has recovered. It&#39;s complicated. And, I think that&#39;s one of the things we&#39;re trying to understand. And, I think it&#39;s a very challenging thing to do. I&#39;m a big fan of the idea that, &#39;Well, we don&#39;t know everything that we need to know. And, &#39;We don&#39;t know everything that we can know. And, that&#39;s okay. And, I think that&#39;s not just a philosophical thing. I think it&#39;s very practical. I think it&#39;s very practical for a number of reasons. But, I think, even more than that, the way I think about it is, it&#39;s very likely that there are things we don&#39;t know, and things that we don&#39;t know are things that we don&#39;t know we don&#39;t know</td>
</tr>
<tr>
<td>ada</td>
<td>How would Russ Roberts finish the following sentence from Adam Smith: &quot;Man desires not only to be loved</td>
<td>How would Russ Roberts finish the following sentence from Adam Smith: &quot;Man desires not only to be loved, but to be lovely. &quot;So, this is the point. Smith is suggesting that the reason we like to be loved is not because we&#39;ve got a flaw. It&#39;s not because we&#39;re ugly. It&#39;s because we&#39;re lovely. Smith is suggesting that loving is a bad thing, not a good thing. The reason we like to be loved is that it&#39;s a pleasure. Smith is suggesting that the reason we like to be loved is a bad thing. That is: It&#39;s not a good thing to be loved. It&#39;s a bad thing to be unloved. The reason we like to be unloved is because we&#39;re ugly. We&#39;re ugly because we do not like to be unloved. We&#39;re ugly because we like to be loved. The reason we like to be loved is not a bad thing. The reason we like to be unloved is a good thing. The reason we like to be loved is not a bad thing to be loved. The reason we like to be unloved is not a bad thing to be unloved. The reason we like to be unloved is not a bad thing to be unloved. </td>
</tr>
<tr>
</tbody></table>


<br>

## Conclusion 

LLMs like OpenAI's `text-davinci-003` are remarkably capable at sounding like public personalities after training on a small number of samples.  One could image all sorts of fun, useful, or even malicious ways this could be done to create personality-specific sounding bots from the White House Press Secretary to [Joe Rogan](https://mobile.twitter.com/ashwingop/status/1610290425524539394). Keep in mind that my experiment used only 2% of the dataset I put together (i.e. 83K tokens and 204 conversation pairs). Training on the entire dataset (~10K conversation pairs) would no doubt produce even more impressive results. Furthermore, the way the prompt/completions were specified could likely be improved to provide more context, and a way of further refining the trained model output through embeddings could also be useful.

However, there are still many drawbacks. The smaller models (`ada`/`babbage`) do not appear to learn anything useful from fine-tuning, and the `curie` model is not fully able to understand the key idea found in the prompt. This suggests a scaling law in favour of model size over dataset size, which aligns with OpenAI's [earlier paper](https://arxiv.org/pdf/2001.08361v1.pdf), rather than DeepMind's [*Chinchilla* experiments](https://arxiv.org/pdf/2203.15556.pdf). However, the aforementioned result is not exactly comparable since the "budget" I'm referring to is the training cost OpenAI charges, rather than the number of flops. 


Overall, I was impressed by the ease of using the OpenAI API, and found myself often going between the command line and the GUI playground found on their website. My main critique of OpenAI's tools is that have not provided any seeding functionality, meaning these experiments are not reproducible by other researchers (or even the same researcher on a different day!). Interested readers can create their own versions of [`EconChatR`](https://github.com/ErikinBC/EconChattR) by cloning this repo and running the main pipeline. 


<br>
<p align="center"> <img src="/figures/openai_cost.png" width="40%"> </p>
<p align="center"> <i> A set of fun experiments for $40 USD </i> </p>
<br>

<br>

***

<br>

### Footnotes

[^1]: Here are some fairly emblematic headlines that have come out on social and traditional media platforms: "[Google is done. Here’s why OpenAI’s ChatGPT Will Be a Game Changer](https://medium.com/@lucapetriconi/google-is-done-heres-why-openai-s-chatgpt-will-be-a-game-changer-98ae591ad747)", "[ChatGPT is a game changer for artificial intelligence](https://www.ctvnews.ca/video?clipId=2585069)", "[ChatGPT is a GAME CHANGER!](https://www.reddit.com/r/algotrading/comments/zkdi3e/chatgpt_is_a_game_changer/)".

[^2]: The model has ~170B parameters and was trained on about 300 billion tokens (where 5 tokens averages about 4 words).

[^3]: It would [likely take](https://twitter.com/tomgoldsteincs/status/1600196981955100694?lang=en) 8 high-end GPUs to run a model of this size on a local machine (each GPU would cost ~$15K). Furthermore, it has been [estimated](https://lambdalabs.com/blog/demystifying-gpt-3) that GPT-3 would have cost $5 million USD  to train and one [popular estimate](https://twitter.com/tomgoldsteincs/status/1600196995389366274) has pegged the cost of running ChatGPT at $3 million USD a month.

[^4]: Note that these numbers are slightly different than what is seen above for the reason that the number of pairs (`n_pairs`) column is based on an estimate rather than an actual value, since some pairs will have more/fewer tokens than others. The final number quoted is based on the model-specific training set that gets generated by rank-ordering the 12 different string match types from the smallest to the largest number of tokens.
