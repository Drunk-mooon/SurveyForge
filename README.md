<div align="center">
<h1>SurveyForge: On the Outline Heuristics, Memory-Driven Generation, and Multi-dimensional Evaluation for Automated Survey Writing</h1>

[[ Paper 📓 ]](https://arxiv.org/abs/xxxx.xxxxx) [[ Website 🏠 ]](https://huggingface.co/papers/xxxx.xxxxx) [[ SurveyBench Benchmark 🤗 ]](https://huggingface.co/datasets/U4R/SurveyBench) 

<i>
🤩 Tired of chaotic structures and inaccurate references in AI-generated survey paper? <b>SurveyForge</b> is here to revolutionize your research experience!
</i>
</div>


## Introduction

<p align="center">
  <img src="./assets/framework_surveyforge.png" width="99%">
  <!-- <div>The overview of SURVEYFORGE. The framework consists of two main stages: Outline Generation and Content Writing. In the Outline Generation stage, SURVEYFORGE utilizes heuristic learning to generate well-structured outlines by leveraging topic-relevant literature and structural patterns from existing surveys. In the Content Writing stage, a memory-driven Scholar Navigation Agent (SANA) retrieves high-quality literature for each subsection and LLM generates the content of each subsection. Finally, the content is synthesized and refined into a coherent and comprehensive survey.
</div> -->
</p>

Survey papers are vital in scientific research, especially with the rapid increase in research publications. Recently, researchers have started using LLMs to automate survey creation for improved efficiency. However, LLM-generated surveys often fall short compared to human-written ones, particularly in outline quality and citation accuracy. To address this, we introduce **SurveyForge**, which first creates an outline by analyzing the structure of human-written outlines and consulting domain-related articles. Then, using high-quality papers retrieved by our scholar navigation agent, **SurveyForge** can automatically generate and refine the content of the survey.

Moreover, to achieve a comprehensive evaluation, we construct **SurveyBench**, which includes 100 human-written survey papers for win-rate comparison and assesses AI-generated survey papers across three dimensions: reference, outline, and content quality.

## 🤔How to try out SurveyForge?

Due to the current limitations on API call frequency, **please kindly send us an email or open an issue** in the repository to inform us of the **survey topic** you intend to generate.

⏱️Surveyforge only takes about **10 minutes** to generate a survey paper. There may be a wait time as the number of users increases, so submit your topic early!

🌟Don’t forget to click the STAR to track if your survey is ready!

At this stage, this is the best way for us to showcase the capabilities of SurveyForge. Moving forward, we are actively working to enhance our API capacity and aim to make SurveyForge publicly accessible in the near future. Thank you for your understanding and support!

## 📝Examples

| Topics                                                   | Links |
| -------------------------------------------------------- | :---: |
| Multimodal Large Language Models                         | [Comprehensive Survey on Multimodal Large Language Models Advances, Challenges, and Future Directions.pdf](demo_papers/Comprehensive%20Survey%20on%20Multimodal%20Large%20Language%20Models%20Advances,%20Challenges,%20and%20Future%20Directions.pdf) |
| 3D Object Detection in Autonomous Driving                | [Comprehensive Survey on 3D Object Detection in Autonomous Driving.pdf](demo_papers/Comprehensive%20Survey%20on%203D%20Object%20Detection%20in%20Autonomous%20Driving.pdf) |
| Vision Transformers                                      | [A Comprehensive Survey on Vision Transformers Foundations, Advances, Applications, and Future Directions.pdf](demo_papers/A%20Comprehensive%20Survey%20on%20Vision%20Transformers%20Foundations,%20Advances,%20Applications,%20and%20Future%20Directions.pdf) |
| Generative Diffusion Models                              | [Comprehensive Survey on Generative Diffusion Models Foundations, Innovations, and Applications.pdf](demo_papers/Comprehensive%20Survey%20on%20Generative%20Diffusion%20Models%20Foundations,%20Innovations,%20and%20Applications.pdf) |
| LLM-based Multi-Agent                                    | [Comprehensive Survey of Large Language Model-Based Multi-Agent Systems.pdf](demo_papers/Comprehensive%20Survey%20of%20Large%20Language%20Model-Based%20Multi-Agent%20Systems.pdf) |
| Self-Supervised Learning in Computer Vision              | [A Comprehensive Survey on Self-Supervised Learning in Computer Vision.pdf](demo_papers/A%20Comprehensive%20Survey%20on%20Self-Supervised%20Learning%20in%20Computer%20Vision.pdf) |
| Embodied Artificial Intelligence                         | [A Comprehensive Survey on Embodied Artificial Intelligence Foundations, Advances, and Future Directions.pdf](demo_papers/A%20Comprehensive%20Survey%20on%20Embodied%20Artificial%20Intelligence%20Foundations,%20Advances,%20and%20Future%20Directions.pdf) |
| Vector Database Management Systems                       | [A Comprehensive Survey of Vector Database Management Systems Foundations, Architectures, and Future Directions.pdf](demo_papers/A%20Comprehensive%20Survey%20of%20Vector%20Database%20Management%20Systems%20Foundations,%20Architectures,%20and%20Future%20Directions.pdf) |
| Gradient Descent and Its Expanding Frontier              | [Comprehensive Survey of Gradient Descent and Its Expanding Frontier.pdf](demo_papers/Comprehensive%20Survey%20of%20Gradient%20Descent%20and%20Its%20Expanding%20Frontier.pdf) |
| Formal Verification of Neural Networks                   | [Comprehensive Survey on Formal Verification of Neural Networks Foundations, Methods, and Future Directions.pdf](demo_papers/Comprehensive%20Survey%20on%20Formal%20Verification%20of%20Neural%20Networks%20Foundations,%20Methods,%20and%20Future%20Directions.pdf) |
| Edge Computing Paradigms and Technologies                | [A Survey on Edge Computing Paradigms and Technologies.pdf](demo_papers/A%20Survey%20on%20Edge%20Computing%20Paradigms%20and%20Technologies.pdf) |
| Automated Machine Learning                               | [Automated Machine Learning Foundations, Advancements, Applications, and Future Directions.pdf](demo_papers/Automated%20Machine%20Learning%20Foundations,%20Advancements,%20Applications,%20and%20Future%20Directions.pdf) |
| AI in Facial Recognition                                 | [Applications of Artificial Intelligence in Facial Recognition Techniques, Challenges, and Future Directions.pdf](demo_papers/Applications%20of%20Artificial%20Intelligence%20in%20Facial%20Recognition%20Techniques,%20Challenges,%20and%20Future%20Directions.pdf) |
| Natural Language Processing                              | [Advancements in Natural Language Processing Developments, Trends, and Future Directions.pdf](demo_papers/Advancements%20in%20Natural%20Language%20Processing%20Developments,%20Trends,%20and%20Future%20Directions.pdf) |
| Adversarial Machine Learning                             | [Adversarial Machine Learning Attack Methods and Defense Mechanisms.pdf](demo_papers/Adversarial%20Machine%20Learning%20Attack%20Methods%20and%20Defense%20Mechanisms.pdf) |
| Federated Learning                                       | [Federated Learning Privacy-Preserving Collaborative Machine Learning.pdf](demo_papers/Federated%20Learning%20Privacy-Preserving%20Collaborative%20Machine%20Learning.pdf) |
| Human-Computer Intelligent Interaction                   | [Human-Computer Intelligent Interaction Foundations, Technologies, and Future Perspectives.pdf](demo_papers/Human-Computer%20Intelligent%20Interaction%20Foundations,%20Technologies,%20and%20Future%20Perspectives.pdf) |
| AI-Powered Autonomous Scientific Discovery               | [AI-Powered Autonomous Scientific Discovery Challenges, Innovations, and Future Directions.pdf](demo_papers/AI-Powered%20Autonomous%20Scientific%20Discovery%20Challenges,%20Innovations,%20and%20Future%20Directions.pdf) |
| LLMs in Mental Health Services                           | [Applications of Large Language Models in Mental Health Services Capabilities, Challenges, and Future Directions.pdf](demo_papers/Applications%20of%20Large%20Language%20Models%20in%20Mental%20Health%20Services%20Capabilities,%20Challenges,%20and%20Future%20Directions.pdf) |
| Quantum Computing Algorithms                             | [Quantum Computing Algorithms Foundations, Advancements, and Frontier Perspectives.pdf](demo_papers/Quantum%20Computing%20Algorithms%20Foundations,%20Advancements,%20and%20Frontier%20Perspectives.pdf) |

## 🕵️‍♂️How to evaluate the quality of the survey paper?

We offer **SurveyBench**, a benchmark for **academic research** and **evaluating the quality of AI-generated surveys.**

[SurveyBench Download](https://huggingface.co/datasets/U4R/SurveyBench)

Currently , SurveyBench consists of approximately 100 human-written survey papers across 10 distinct topics, carefully curated by doctoral-level researchers to ensure thematic consistency and academic rigor. The supported topics and the core references corresponding to each topic are as follows:

| Topics                                                   | # Reference |
| -------------------------------------------------------- | :---------: |
| Multimodal Large Language Models                         |     912     |
| Evaluation of Large Language Models                      |     714     |
| 3D Object Detection in Autonomous Driving                |     441     |
| Vision Transformers                                      |     563     |
| Hallucination in Large Language Models                   |     500     |
| Generative Diffusion Models                              |     994     |
| 3D Gaussian Splatting                                    |     330     |
| LLM-based Multi-Agent                                    |     823     |
| Graph Neural Networks                                    |     670     |
| Retrieval-Augmented Generation for Large Language Models |     608     |

More support topics coming soon!

### 🧑‍💻You can evaluate the survey by:

```
cd SurveyBench && python test.py --is_human_eval
```

Note set `is_human_eval` True for human survey evaluation, False for generated surveys.

If you want to evaluate your method on SurveyBench, please follow the format:

```
generated_surveys
|-- 3D Gaussian Splatting
    |-- exp_1
        |-- ref.json
    |-- exp_2
        |-- ref.json
...
|-- Graph Neural Networks
...
```

## Citations

```
{}
```

