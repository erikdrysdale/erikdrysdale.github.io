---
title: "Matching methods are poorly suited to analysing harm reduction policies: a critique of Lee et. al (2021)"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

### Executive summary

In the recently published paper in *JAMA Network Open*, *Systematic Evaluation of State Policy Interventions Targeting
the US Opioid Epidemic, 2007-2018*, [Lee et. al (2021)](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2776301) (hereafter Lee) use an econometric method which claims to find a deleterious association between the adoption of harm reduction policies at the state level and overdose deaths.[[^1]] Their method estimates that both Naloxone and Good Samaritan laws *increase* overdose deaths for up to 12 quarters after policy adoption. I provide three compelling reasons why their analysis with regards to Naloxone and Good Samaritan laws and their relation to overdose deaths is flawed from a statistical perspective.

1. Their test statistic does not have a proper distribution under a null hypothesis of no effect. Randomizing the date of policy adoption for Naloxone laws and using their method leads to an average test-statistic z-score of 1.3 (it should be zero). **Under random policy adoption dates, their method would conclude that Naloxone and Good Samaritan laws increases overdose deaths 41% and 25% of the time, respectively**.
2. The matching procedure is failing to provide control states which look "similar" to the treated states. Specifically, the rate of overdose deaths before policy adoption is growing faster in the treated states, which violates the parallel trends assumption needed for inference.
3. The covariates used to generate the propensity score probabilities have a weak relationship to policy adoption.

Since the date of policy adoption determines the coefficient used to assess whether these law increase overdose deaths, the Naloxone and Good Samaritan results are unreliable since a completely random date assignment also leads to a disproportionate number of inferences which suggest the same results. The statistical reason for the inflated type-I error rate is because the standard errors are too small (the null distribution is not a standard normal), and (in the case of Naloxone laws) there is a non-zero mean.

I also found the matching methods unsatisfactory since the covariates used to generate the scores have a weak relationship to the date of policy adoption. This means that propensity scores used to find matches are noisy. These poor matches in turn lead to a violation of the parallel trends assumption as the gap in opioid deaths between the treatment and control states is growing before as well as after policy adoption. Since the average treatment effect coefficient they used relies on this assumption, the results are confounded. 

### Background

The United States and Canada are experiencing an [opioid crisis](https://en.wikipedia.org/wiki/Opioid_epidemic). The origins of this public health emergency are complex, but "innovations" in prescription opioids (OxyContin and Vicodin), the emergence of new street drugs like black tar heroin and fentanyl, as well as drug criminalization have been posited as primary causes. The annual numbers of opioid overdose deaths in North America are now staggering (see Figure 1).[[^2]] To put these numbers into perspective, the number of fatal motor vehicle crashes and firearms suicides combined (mortality rates of 11 and 8 per 100K persons, respectively) are equivalent to opioid overdose deaths in the US.[[^3]] Although Canada has fewer deaths by firearms and motor vehicles, our country is experiencing the overdoses with no less severitys than our southern neighbours.

<center><h3>Figure 1: US and CAD opioid deaths </h3></center>
<p align="center"><img src="/figures/gg_deaths_comp.png" width="60%"></p>

[Harm reduction](https://en.wikipedia.org/wiki/Harm_reduction) is a public health approach to drug use which stresses the need to reduce the risk of using drugs rather than reducing demand through criminalization or social stigma. "Safe sex" and "drink responsibly" are examples of harm reduction approaches to other public health challenges. Recent research has called into question the efficacy of harm reduction policies in response to the opioid crisis. For example [Lee et. al (2021)](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2776301) have carried out econometric analysis which claims that Naloxone and Good Samaritan policies *increase* overdose deaths in the long run. Specifically, the authors use a [matching method](https://imai.fas.harvard.edu/research/tscs.html) designed to estimate causal effects from policies in time-series cross sectional data. This method gives positive coefficient estimates for total overdose deaths up to 12 quarters after policy adoption. The paper defines these harm reduction policies as follows:

1. Good Samaritan laws provide immunity or other legal protection for those who call for help during overdose events.
2. Naloxone access laws provide civil or criminal immunity to licensed health care clinicians or lay responders for administration of opioid antagonists, such as naloxone hydrochloride, to reverse overdose.

What explanation could be given for finding these deleterious effects? The authors do not speculate too much, but they do cite [other research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3135264) that suggests moral hazard may be at work.

> It is possible that the prospect of getting access to overdose-reversing treatment may instead induce moral hazards by encouraging people to use opioids and other drugs in riskier ways than they would have without the safety net of Naloxone.
>
> These unintended consequences may occur if the fundamental causes of demand for opioids are not addressed and if the ability to reverse overdose is expanded without increasing treatment of opioid overdose.

I was sceptical of these findings for two reasons. First, the claim of moral hazard only makes sense for explaining an increase of non-fatal opioid overdoses. By definition, in order for naloxone to be a moral hazard, it has to reduce the risk of dying from opioids. It would be strange that something which induces more use because of increased personal safety also killed more people. Second, having been an applied statistician and economist for almost an decade, I knew that these econometric estimates are highly sensitive to minor changes in the analysis pipeline. I was curious weather these estimates would be robust to any sort of statistical probing.

In the rest of this post I will provide evidence that the econometric results for Naloxone or Good Samaritan laws from Lee et. al (2021) are flawed. Though I think this paper needs revisions to address these problems, I applaud the authors for providing reproducible code and data. I also welcome a critique of any of my claims.

### Data overview

The authors do an excellent job at making their [code and data](https://dataverse.harvard.edu/dataverse/bk) accessible for researchers. Some of the data sources are confidential, but they provide aggregated measures that can be used in the downstream analysis. This paper should serve as a model for how data-driven social science research is conducted. My analysis in turn was based on code that can be found [here](https://github.com/ErikinBC/us_opioid). The data for the paper comes from several sources including the Optum Clinformatics Data Mart Database, NCHS Multiple Cause of Death file, and Centers for Disease Control and Prevention's CDC Wonder database.

I focused exclusively on two policy treatments (Naloxone and Good Samaritan laws) and one outcome (total overdose deaths). Though the paper looked at multiple policies and outcomes, I choose to focus on the two harm reduction policies and overdose deaths as I felt these were the most consequential for informing the harm reduction community. Figure 2 shows that the underlying dynamics in the different types of overdose deaths in the US, with synthetic opioids (like fentanyl) sky-rocketing even as methadone deaths fall.

<center><h3>Figure 2: US Opioid deaths over time by category </h3></center>
<p align="center"><img src="/figures/gg_dpc.png" width="60%"></p>

Figure 3 below shows the cumulative percentage of US states that adopted a harm reduction or supply-controlling policy. By the end of the time series, all states (and DC) had adopted a Naloxone laws. The fact makes the finding the counter factual claim that states should not have adopted Naloxone laws all the more paradoxical as every state eventually joined the treated group. Though the time-series cross-sectional statistical methods do not need there to be a permanently untreated state *per se*, the implication is that all states, eventually, made a policy mistake. 

<center><h3>Figure 3: Cumulative state policy adoption </h3></center>
<p align="center"><img src="/figures/gg_pct_policy.png" width="60%"></p>

TALK ABOUT THIS

<center><h3>Figure 4: Difference on overdose deaths between adopted versus un-adopted states </h3></center>
<p align="center"><img src="/figures/gg_diff_policy.png" width="60%"></p>

### The statistical estimator

The authors rely on a panel matching method implemented through the [`PanelMatch`](https://cran.r-project.org/web/packages/PanelMatch/index.html) package in `R`. Formally, the method is trying to estimate difference in the change of the outcome for up to $F$ periods ahead conditioning on $L$-lags of of the control covariate:

$$
\begin{align*}
\delta(F,L) &= \frac{1}{\sum_{i=1}^N \sum_{t=L+1}^{T-F} D_{it} \Bigg\{ (Y_{i,t+F} - Y_{i,t-1}) - \sum_{i'\in M_{it} w_{it}^{i'} (Y_{i',t+F} - Y_{i',t-1}) }}  \Bigg\} },
\end{align*}
$$

Where \((D_{it}\\) is an indicator of whether state \((i\\) changed its policy at time \((t\\), \((M_{it}\\) is the set of control states, \((Y_{it}\\) is the outcome, and \((w_{it}\\) is the weight for the control state. At every time point, the sum of weights for a given treated state must be one. This average treatment among the treated (ATT) estimator is sensitive to the parallel trends assumption since it compares the difference in the difference in levels between the treated and control states (hence it is a type of difference-in-difference estimator).

### Failure of the parallel trends assumption

Unfortunately when I use Lee's code to replicate their analysis, it was clear that the growth rates in overdose deaths between the treated and control states was not equal. 

<center><h4>Figure 5: Overdose death growth rates between control groups</h4></center>
<table>
    <tr>
        <td> <p align="left">(5A) Observed distribution </p> <img src="/figures/gg_pct_lag.png" style="width: 50%;"/>  </td>
        <td> <p align="left">(5B) Bootstrapped difference </p> <img src="/figures/gg_pct_dist.png" style="width: 50%;"/> </td>
    </tr>
</table>

### Matching and propensity scores

I did not spend too much time analyzing what wrong wit

<center><h3>Figure 6: Control state weights </h3></center>
<p align="center"><img src="/figures/gg_csum_weights.png" width="60%"></p>

### Randomizing policy adoption dates


<center><h4>Figure X: </h4></center>
<table>
    <tr>
        <td> <p align="left">(XA) </p> <img src="/figures/gg_tab_zscore.png" style="width: 50%;"/>  </td>
        <td> <p align="left">(XB) </p> <img src="/figures/gg_se_tab.png" style="width: 50%;"/> </td>
    </tr>
</table>


<center><h3>Figure X: </h3></center>
<p align="center"><img src="/figures/gg_coef_zscore.png" width="60%"></p>

<center><h3>Figure X: </h3></center>
<p align="center"><img src="/figures/gg_se_coef.png" width="60%"></p>


* * *

### Notes

[^1]: This results of this study was picked up by the media and made a bit of a splash. See [here](https://www.eurekalert.org/pub_releases/2021-02/iu-isf021721.php), [here](https://www.cato.org/blog/more-evidence-prescription-drug-monitoring-programs-might-increase-overdose-deaths), [here](https://www.news-medical.net/news/20210217/Opioid-supply-controlling-policies-may-have-unintended-consequences-finds-study.aspx), and [here](https://www.sciencedaily.com/releases/2021/02/210217134802.htm). 

[^2]: This death rate is likely a conservative estimate because it is difficult to separate overdose deaths from opioid-specific deaths. US data comes from [here](https://www.drugabuse.gov/sites/default/files/Overdose_data_1999-2019.xlsx), with 2020 estimate based on [preliminary estimates](https://www.cdc.gov/nchs/nvss/vsrr/drug-overdose-data.htm) of a 27% year-on-year increase. [Canadian data](https://health-infobase.canada.ca/substance-related-harms/opioids-stimulants/graphs?index=395) comes from [here](https://health-infobase.canada.ca/src/doc/HealthInfobase-SubstanceHarmsData.zip), with 202 estimate based on annualized estimates from January to September data.  

[^3]: Motor vehicle deaths [source](https://www.iihs.org/topics/fatality-statistics/detail/state-by-state), and firearms death [source](https://health.ucdavis.edu/what-you-can-do/facts.html).