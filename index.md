[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

![Header](images/ea9464db-32b5-4693-a424-46dd7409bef5.jpeg)

## Contents

* [Credit Risk Modeling](#credit-risk-modeling)
* [Graph Machine Learning for Credit Risk](#graph-machine-learning-for-credit-risk)
* [Model Risk Management for AI/ML Models](#model-risk-management-for-aiml-models)

## Credit Risk Modeling

- [Machine Learning-Driven Credit Risk: A Systemic Review](https://link.springer.com/article/10.1007/s00521-022-07472-2)

> This paper systematically reviews a series of major research contributions (76 papers) over the past eight years using statistical, machine learning and deep learning techniques to address the problems of credit risk.

- [Machine Learning in Retail Credit Risk: Algorithms, Infrastructure, and Alternative Data — Past, Present, and Future | NVIDIA](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)
> A lecture on how machine learning (ML) is reshaping credit risk models: learn about new methods to build transparent ML in highly regulated environments, how deep learning is surfacing alternative financial data and making it more useful, and how on-premises GPU computing is accelerating model development.

- [The Use of Machine Learning for Credit Underwriting | FinRegLab](https://finreglab.org/ai-machine-learning/explainability-and-fairness-of-machine-learning-in-credit-underwriting/the-use-of-ml-for-credit-underwriting-market-data-science-context/)

> The report discusses the efficacy of novel techniques for managing machine learning underwriting models. This research proposes a framework that will help all stakeholders — model developers, risk and compliance personnel, and regulators — assess the accuracy and utility of accessible information about a machine learning underwriting model’s decision-making.

- [Machine learning in Credit Risk Modeling: Efficiency should not come at the expense of Explainability | James - Credit Risk AI](https://www.slideshare.net/YvanDeMunck/machine-learning-in-credit-risk-modeling-a-james-white-paper)

> This whitepaper offers an overview of machine learning applications in the field of Credit Risk Modeling. A primer on Linear models, Decision Trees, and Ensemble methods (Random Forest and Gradient Boosting) in the context of Credit Risk applications.

- [Machine Learning approach for Credit Scoring](https://arxiv.org/abs/2008.01687)

> This paper proposes an end-to-end corporate rating model development methodology using machine learning. A core model architecture (design) is made up of a concatenation of a Light-GBM classifier (risk differentiation), a probability calibrator (risk quantification) and a rating attribution (assignment) system. A must-read for everyone who wants to understand the potential of ML in Credit Risk Modeling.

- [PSD2 Explainable AI Model for Credit Scoring](https://arxiv.org/abs/2011.10367)
> This paper presents an explainable credit risk model design based solely on open-data account transactions (PSD2). The paper also provides detailed guidance on feature engineering of raw transactional risk drivers for probability of default (PD) models.

- [Deep Neural Networks for Behavioral Credit Rating](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7824729/)

> The paper presents a deep neural network model for behavioral credit risk assessment and invites to reconsider the regulatory requirements for model explainability to allow the usage of non-linear models for credit risk assessment purposes. One novelty of this paper is the quantification of the difference in the calibration accuracy of models for each class via the Brier score.

- [An End-to-End Deep Learning Approach to Credit Scoring using CNN + XGBoost on Transaction Data](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4168935)

> This paper argues that the more detailed transactional information the models are fed and are able to utilize, the greater the discriminatory power that can be achieved. Using machine learning algorithms to engineer features based on raw transaction data when developing application models might help reduce the performance gap vis-a-vis behavioral models.

- [Automating Interpretable Machine Learning Scorecards | Moody's Analytics](https://www.moodysanalytics.com/-/media/article/2020/Automating-Interpretable-Scorecards.pdf)

> This paper compares a logistic regression with supervised binning to other popular machine learning (ML) methods and shows that performance of the modified logistic regression is similar to challenger ML models, albeit without losses in economic interpretability.

- [Building Credit Scorecards using SAS and Python | The SAS Data Science Blog](https://blogs.sas.com/content/subconsciousmusings/2019/01/18/building-credit-scorecards-using-statistical-methods-and-business-logic/)

> This blog post presents a step-by-step guide to building a Credit Scorecard using Weight-of-Evidence (WOE) logistic regression approach using SAS and Python.

<details>
  <summary>Other sources | Under review </summary>

  <a href="https://doi.org/10.1214/aos/1016218223">Jerome Friedman. Trevor Hastie. Robert Tibshirani. "Additive logistic regression: a statistical view of boosting (With discussion and a rejoinder by the authors)." Ann. Statist. 28 (2) 337 - 407, April 2000.</a><br>

  <a href="https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html">Why model calibration matters and how to achieve it | The Unofficial Google Data Science Blog</a><br>

  <a href="https://amueller.github.io/aml/04-model-evaluation/11-calibration.html">Calibration, Imbalanced Data |  Applied Machine Learning in Python</a><br>

  <a href="https://www.fast.ai/posts/2017-11-13-validation-sets.html">How (and why) to create a good validation set</a><br>

</details>

## Graph Machine Learning for Credit Risk

- [Graph Neural Networks for Credit Modeling | Katana Graph](https://blog.katanagraph.com/graph-neural-networks-for-credit-modeling)

> Graph Neural Networks excel at encoding high-dimensional data, enabling the machine to capture complex, shifting relationships over time and identify nuances within massive, disparate data. These techniques enable enterprises to learn which features and consumer relationships are most important for credit decisioning.

- [Network Based Credit Risk Models](https://www.tandfonline.com/doi/full/10.1080/08982112.2019.1655159)

> This paper proposes an augmented logistic regression model that incorporates centrality measures derived from similarity networks among borrowers, deduced from their financial ratios. Inclusion of topological variables describing institutions centrality in similarity networks increases the predictive performance of the credit rating model.

- [Temporal-Aware Graph Neural Network for Credit Risk Prediction](https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.79?mobileUi=0)

> The authors build the dynamic graphs to predict defaults by collecting multiple lending events of users and ordering the events by the lending time. The proposed model incorporates static, temporal  and structural features within a dynamic graph to predict the user's credit risk profile.

- [Loan Default Analysis with Multiplex Graph Learning](https://dl.acm.org/doi/10.1145/3340531.3412724)

> In this paper, the authors analyze Transfers and Social relations between users to define the number of defaulted neighbors for each user and then split users into three distinct groups. Both social and transaction relations achieve good performance since people with similar credit risk tend to gather together and such a pattern can be naturally modeled via graph model.

- [Every Corporation Owns Its Structure: Corporate Credit Ratings via Graph Neural Networks](https://arxiv.org/abs/2012.01933)

> This paper offers a new method named corporation-to-graph to explore the relations between features for corporate rating models. In this model, each corporation is represented as an individual graph from which feature level interactions can be learned.

<details>
  <summary>Other sources | Under review </summary>

  <a href="https://www.youtube.com/watch?v=rLCLIUmd9SE&ab_channel=MLOpsLearners">Graph Neural Networks: Theory, Problem, and Approaches | MLOps Learners</a><br>

  <a href="https://huggingface.co/blog/intro-graphml">Introduction to Graph Machine Learning | Hugging Face</a><br>

</details>

## Model Risk Management for AI/ML Models

- [Machine Learning Explainability in Finance: an Application to Default Risk Analysis | Bank of England](https://www.bankofengland.co.uk/working-paper/2019/machine-learning-explainability-in-finance-an-application-to-default-risk-analysis)

> This paper proposes a framework for addressing the ‘black box’ problem present in some machine learning (ML) applications. The paper's goal is to develop a systematic analytical framework that could be used for approaching explainability questions in real-world ﬁnancial applications.

- [Explainable Machine Learning Models of Consumer Credit Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4006840)

> This study demonstrates that the functionality of black-box machine learning (ML) models can be explained to a range of different stakeholders using the right tools. The approach is aimed at unlocking the future potential of applying AI to improve credit risk models' performance.

- [Understanding the Performance of Machine Learning Models to Predict Credit Default: A Novel Approach for Supervisory Evaluation | Bank of Spain](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774075)

> A regulatory perspective from the Bank of Spain on the use of machine learning (ML) methods for probability of default (PD) models, including assessments of discriminatory power, calibration accuracy, and regulatory capital savings.

- [Machine Learning in Credit Risk: Measuring the Dilemma Between Prediction and Supervisory Cost | Bank of Spain](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3724374)

> Bank of Spain's research on the trade-off between accuracy gains and supervisory costs for various model uses of credit risk models (scoring, pricing, provisioning, capital). The paper identifies up to 13 factors that might constitute a supervisory cost and proposes a methodology for evaluating these costs.

- [Accuracy of Explanations of Machine Learning Models for Credit Decisions | Bank of Spain](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4144780)

> A credit risk governance and regulation framework is proposed to evaluate the relevance of the input variables on a credit risk model’s prediction using post hoc explanation techniques such as SHAP and permutation feature importance. The paper also highlights the potential of synthetic datasets for evaluating model interpretability tools.

- [Explaining and Accelerating Machine Learning for Loan Delinquencies | NVIDIA](https://developer.nvidia.com/blog/explaining-and-accelerating-machine-learning-for-loan-delinquencies/)

> In this post, the authors discuss how to use RAPIDS to GPU-accelerate the complete default analytics workflow: load and merge data, train a model to predict new results, and explain predictions of a financial credit risk problem using Shapley values.

- [Accelerating Trustworthy AI for Credit Risk Management | NVIDIA](https://developer.nvidia.com/blog/accelerating-trustworthy-ai-for-credit-risk-management/)

> In this post, the authors describe some use case scenarios for SHAP Clustering. The target of the proposed explainability model are functions for risk management, assessment and scoring of credit portfolios in traditional banks as well as in ‘fintech’ platforms for P2P lending/crowdfunding.

- [Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/abs/1811.10154)

> This paper challenges the notion of a trade-off between accuracy and interpretability outlining several key reasons why explainable black boxes should be avoided in high-stakes decisions. Black-box algorithms can still be useful in high-stakes decisions as part of the knowledge discovery process (for instance, to obtain baseline levels of performance), but they are not generally the final goal of knowledge discovery.

- [We Didn’t Explain the Black Box – We Replaced it with an Interpretable Model | FICO](https://community.fico.com/s/blog-post/a5Q2E0000001czyUAA/fico1670)

> This guest blog was written by the winners of the FICO Recognition Award in the 2018 Explainable Machine Learning Challenge. The team's interpretable Two-Layer Additive Risk Model turned out to be just as accurate (~74% accuracy) as the best black box models.

<details>
  <summary>Other sources | Under review</summary>

  <a href="https://theeffectbook.net/index.html">The Effect: An Introduction to Research Design and Causality</a><br>

</details>
