---
title: 'RMarkdown+Jekyll'
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---



My recent experience in [blogging](http://bioeconometrician.ghost.io/) with the Ghost platform has been fairly satisfactory. When it comes to writing book reviews, or making short posts, writing up a post in Markdown/html is a fairly efficient process. However as a fledging Bioeconometrician, many of the topics I want to post about relate to data science [related posts](http://bioeconometrician.ghost.io/tag/r/), and actually assembling them for Ghost is somewhat cumbersome. As an avid user of the R language, and a recent convert in using RMarkdown for the effective communication of applied statistics, I have often wanted a platform where I could write up a post in RMarkdown and post it directly to a website without worrying about embedded image locations and other such complications.

Luckily R bloggers can combine [Jekyll](https://jekyllrb.com/), a static site generator suited for Github pages, with a few extra hacks to create an excellent RMarkdown to Github page pipeline. The first thing to do is to go and fork Barry Clark's excellent [Jekyll Now](https://github.com/barryclark/jekyll-now) as a template. Next, I used [Andy South's](http://andysouth.github.io/blog-setup/) `rmd2md()` function to create the necessary images and links for a `.Rmd` file. `LaTeX` support can be provided with the MathJax's highlighter by adding the appropriate code to the  `_layouts/default.html` file so one can show in-line math statements like $a^2 + b^2 = c^2$ or equations such as:

$$\int_{-\infty}^{\infty} \Big(\frac{1}{2\pi\sigma^2}\Big)^{\frac{1}{2}} e^{-\frac{1}{2}(x-\mu)^2}dx=1$$

The beauty of RMarkdown is that we can easily show snippets of our data:


{% highlight r %}
iris %>% head(3)
{% endhighlight %}



{% highlight text %}
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
{% endhighlight %}

Or plots:



{% highlight r %}
ggplot(dat,aes(x=val,y=..density..,color='black')) +
  geom_density(aes(fill=Species),show.legend = F,alpha=0.65) +
  facet_grid(Flower~Type,scales='free') +
  labs(x='Value',y='Density',subtitle='Iris data set')
{% endhighlight %}

![plot of chunk iris_chunk](/figures/iris_chunk-1.png)

```python
from pyensembl import EnsemblRelease

# release 77 uses human reference genome GRCh38
data = EnsemblRelease(77)

# will return ['HLA-A']
gene_names = data.gene_names_at_locus(contig=6, position=29945884)
```

I'm looking forward to posting any future work I engage in on this site. Feel free to copy the template of this website from [here](https://github.com/erikdrysdale/erikdrysdale.github.io).
