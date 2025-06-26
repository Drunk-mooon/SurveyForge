ROUGH_OUTLINE_WITH_SURVEY_PROMPT = '''
You are an AI assistant tasked with creating a concise, high-level, comprehensive and original academic survey outline for [TOPIC]. This is a critical task, as the outline you generate will serve as the foundation for a high-quality, comprehensive academic survey paper suitable for submission to top-tier journals in the field. The structure and content you provide will directly influence the quality and coherence of the final paper, so your work here is of utmost importance.

You are provided with two sets of information:

1. Titles, abstracts, top-level outlines and publication dates of human-written surveys that may be related to [TOPIC]. Use these to understand the logical structure, style, and academic phrasing typical in this field.
   ---
   [SURVEY LIST]
   ---

2. Titles, abstracts and publication dates of academic papers that may be related to [TOPIC]. Use these to identify key themes, recent developments, and research trends in the field.
   ---
   [PAPER LIST]
   ---

**Important Notes:**
- Concise Top-level Outline: The first-level outline should be highly condensed and abstract, representing high-level categories.
- Utilize Information Sources: Draw inspiration from the structure and style of the human-written surveys while incorporating the content and trends from the academic papers to create a comprehensive and up-to-date outline.
- Critical Importance of Descriptions: The descriptions you provide for each section are crucial. They will be used directly for retrieving relevant academic papers and generating more detailed second-level outlines in subsequent stages. Structure these descriptions as bullet points, each representing a key aspect or sub-domain to be explored.

**Main Task:**
Create a comprehensive, original, and academically rigorous survey outline for [TOPIC]. The outline should include a title and approximately [SECTION NUM] high-level, abstract sections that encapsulate broad categories of the topic, following the structure typical of academic surveys in this field. Ensure your outline is both well-structured and current.

**Important Academic Guidelines:**
1. **Relevance and Focus**: Ensure every section directly relates to [TOPIC], covering its breadth including recent developments and historical context.
2. **Originality and Critical Analysis**: Provide new perspectives that go beyond summarizing existing work. Draw inspiration from provided sources without directly copying their structure.
3. **Logical Structure**: Follow a typical academic survey progression: introduction, main body sections, and conclusion with future directions.

**Additional Guidelines:**
1. **Comprehensive Coverage and Gap Identification**: While being aware that the provided paper list is only a subset of the entire field, strive to cover all important aspects of [TOPIC]. At the end of your outline, include a section titled 'Potential Gaps and Future Research Directions' to address areas that might be underrepresented but could be important to the field.

**Your outline should contain a title and approximately [SECTION NUM] sections.**

Each section should be followed by:
- A **Description** consisting of a informative description and **2-4** bullet points, the number of points per subsection does not need to be consistent, you can consider adding or reducing points. Each bullet point should represent a key aspect or sub-domain of the section, followed by a Informative description.
- It is important to note that there is no overlap between bullet points, which represent different aspects of the section
- **Do not use abbreviations or acronyms in the descriptions**. Always write out full terms to ensure clarity and aid in future retrieval processes
- Ensure all terms are fully written out without abbreviations or acronyms.
**Return in the format:**
<format>
Title: [TITLE OF THE SURVEY ON TOPIC]
Section 1: [NAME OF SECTION 1]
Description 1: [INFORMATIVE DESCRIPTION OF Section 1]
1. [Informative description of key aspect or sub-domain 1 of Section 1]
2. [Informative description of key aspect or sub-domain 2 of Section 1]
...

Section K: [NAME OF SECTION K]
Description K: [INFORMATIVE DESCRIPTION OF Section K]
1. [informative description of key aspect or sub-domain 1 of Section K]
2. [Informative description of key aspect or sub-domain 2 of Section K]
...
N. [Informative description of key aspect or sub-domain N of Section K]

Potential Gaps and Future Research Directions:
- [Potential gap or future direction 1]
...
- [Potential gap or future direction ...]
</format>

The outline:
'''


MERGING_OUTLINE_WITH_SURVEY_PROMPT = '''
You are an AI assistant tasked with creating a concise, high-level, comprehensive and original academic survey outline for [TOPIC]. This is a critical task, as the outline you generate will serve as the foundation for a high-quality, comprehensive academic survey paper suitable for submission to top-tier journals in the field. The structure and content you provide will directly influence the quality and coherence of the final paper, so your work here is of utmost importance.


You are provided with two sets of information:

1. Titles, abstracts, top-level outlines and publication dates of human-written surveys that may be related to [TOPIC]. Use these to understand the logical structure, style, and academic phrasing typical of academic survey papers written by humans.
  
   ---
   [SURVEY LIST]
   ---

2. AI-generated outlines from subsets of papers related to [TOPIC], each containing:
   - Title
   - Sections 
   - Descriptions for each section (a informative description and some bullet points)
   - "Potential Gaps and Future Research Directions" section

   ---
   [OUTLINE LIST]
   ---

**Main Task:**
Generate a final, cohesive top-level outline for [TOPIC] by merging and refining the provided AI-generated outlines. Focus on creating an accurate structure with precise, informative section descriptions that will help the subsequent RAG (Retrieval-Augmented Generation) and second-level outline generation. Remember, this outline will be the backbone of a high-quality academic survey paper, so maintain the highest standards of academic rigor and comprehensiveness.
You need to generate a final outline based on these provided outlines to make the final outline show comprehensive insights of the topic and more logical.

**Instructions:**
1. Structure:
   - Start with an **Introduction** section
   - Include approximately **[SECTION NUM] main body sections**
   - End with a **Conclusion** section
   - Ensure logical flow and comprehensive coverage of [TOPIC]

2. Content:
   - Prioritize and consolidate key themes from AI-generated outlines
   - Eliminate redundancies and overlaps between sections
   - Ensure each section has a distinct focus and purpose

3. Descriptions:
   - In general, for each section, provide a brief yet informative description and **several sub-domain points with informative sub-description**, the number of points per subsection does not need to be consistent. You can add or subtract according to the actual scope of the section
   - Each bullet point should represent a key aspect or sub-domain of the section, followed by a Informative description
   - Ensure descriptions are broad enough to cover the section topic but specific enough. It is important to avoid duplication in the description of sub-domains in each section
   - Avoid repetition of concepts across different sections


4. Precision and Academic Standards:
   - Ensure each section and description relates directly to [TOPIC]
   - Maintain a logical progression of ideas throughout the outline
   - Use academically appropriate terminology and phrasing
   - Avoid detailed content generation - focus on accurate structural representation and guiding descriptions
   - **Do not use abbreviations or acronyms in the descriptions**. Always write out full terms to ensure clarity and aid in future retrieval processes

5. Consistency and Coherence:
   - Create clear distinctions between sections to avoid overlap. It is important to note that there is no overlap between bullet points, which represent different aspects of the section
   - If a topic appears in more than one section or the descriptions of sub-domains, clearly distinguish the context and relevance of each section. If necessary, you can choose to remove or merge
   
**Format your final outline as follows:**

<format>
Title: [TITLE OF THE SURVEY ON TOPIC]

Section 1: Introduction
Description 1: [INFORMATIVE DESCRIPTION OF Introduction]
1. [Informative description of key aspect or sub-domain 1 of Introduction]
2. ...

Section 2: [NAME OF SECTION 2]
Description 2: [INFORMATIVE DESCRIPTION OF Section 2]
1. [Informative description of key aspect or sub-domain 1 of Section 2]
2. ...

Section ...

Section K-1: [NAME OF SECTION K-1]
Description K-1: [INFORMATIVE DESCRIPTION OF Section K-1]
1. [Informative description of key aspect or sub-domain 1 of Section K-1]
...
N. [Informative description of key aspect or sub-domain N of Section K-1]

Section K: Conclusion
Description K: [INFORMATIVE DESCRIPTION OF Conclusion]
1. [Informative description of key aspect or sub-domain 1 of Conclusion]
2. ...

</format>

Remember, the quality of this outline is crucial as it will guide the entire writing process of a high-impact academic survey paper. Ensure that your outline is comprehensive, logically structured, and reflective of the current state and future directions of research in [TOPIC]. Your work here will significantly influence the final paper's potential for publication in top-tier academic journals.

Only the final outline is returned, note that the final outline contains only the Section and Description parts and does not provide any other information. 
Ensure all terms are fully written out without abbreviations or acronyms.
'''


SUBSECTION_OUTLINE_WITH_SURVEY_PROMPT = '''
You are an expert in artificial intelligence writing a comprehensive outline of the survey about **[TOPIC]**.

You have created the following overall outline:
---
[OVERALL OUTLINE]
---

You need to enrich the section **[SECTION NAME]**, described as: **[SECTION DESCRIPTION]**

**Main task:**
Generate a comprehensive framework for **[SECTION NAME]** by creating an appropriate number of subsections (typically 3-6, but adjust based on content importance and complexity). Each subsection should focus on a specific aspect and be followed by a Informative description.

**Resources provided:**

1. **A list of [RAG NUM] relevant papers with titles, abstracts, publication dates for this section:**
   ---
   [PAPER LIST]
   ---

2. **Titles, abstracts, top-second outlines and publication dates of human-written surveys** that may be related to [TOPIC].
   ---
   [SURVEY LIST]
   ---

   *Note:* These surveys may not be directly about **[TOPIC]**. Only use these to understand the logical structure, style, and academic phrasing typical of academic survey papers written by humans.

**How to use the provided resources:**
- Use the relevant papers to identify key themes, recent developments, and important concepts within **[SECTION NAME]**.
- Refer to the human-written surveys to understand typical structures and academic phrasing, but ensure your outline is original and specifically tailored to **[TOPIC]** and **[SECTION NAME]**.
- Synthesize information from both sources to create a comprehensive and up-to-date framework for the section.
- Prioritize recent developments and emerging trends when creating your outline, while also acknowledging foundational concepts.

**Guidelines:**
1. **Relevance:** Each subsection must be related to **[SECTION NAME]** and align with its description.
2. **Originality:** Learn from the human-written surveys to inform your structure, but be careful to avoid plagiarism.
3. **Logical Flow:** Arrange subsections in a logical order that builds upon previous ones, ensuring a coherent progression of ideas. It is important to note that there is no overlap between subsection and its bullet points, which represent different aspects of the section.
4. **Flexibility:** The number of subsections should be determined by the content requirements of **[SECTION NAME]**. While 3-6 subsections are typical, prioritize comprehensive coverage over adhering to a strict number.
5. **Separability:** Each subsection should have **an informative description** and **2-4 sub-domain points with informative sub-description**, which do not duplicate and fit the subsection**, the number of points per subsection does not need to be consistent. You can add or subtract according to the actual scope of the section. Each bullet point should represent a key aspect or sub-domain of the section, followed by a informative description.

** Output format: **
<format>
Subsection 1: [NAME OF SUBSECTION 1]
Description 1: [INFORMATIVE DESCRIPTION OF SUBSECTION 1]
1. [Informative description of Key aspect or sub-domain 1 of SUBSECTION 1]
2. ...

Subsection 2: [NAME OF SUBSECTION 2]
Description 2: [INFORMATIVE DESCRIPTION OF SUBSECTION 2]
1. [Informative description of Key aspect or sub-domain 1 of SUBSECTION 2]
...
N. [Informative description of Key aspect or sub-domain N of SUBSECTION 2]

...

Subsection K: [NAME OF SUBSECTION K]
Description K: [INFORMATIVE DESCRIPTION OF SUBSECTION K]
1. [Informative description of Key aspect or sub-domain 1 of SUBSECTION K]
2. ...

</format>

Note: The number of subsections (K) should be appropriate for the content of **[SECTION NAME]**. Ensure descriptions are specific, contain key terminology, and provide clear guidance for detailed content creation.
Only return the outline without any other informations:
'''


EDIT_FINAL_OUTLINE_PROMPT_NEW = '''
You are an expert in artificial intelligence tasked with refining a comprehensive survey outline about [TOPIC].

You have created a draft outline as follows:
---
[OVERALL OUTLINE]
---

This outline contains a title, several main sections, and subsections under each main section. Each section and subsection is accompanied by a brief description.

Your task is to refine this outline, making it comprehensive, logically coherent, and free of repetitions. While doing so, carefully consider the descriptions provided for each section and subsection to ensure the refined outline accurately reflects the intended content and scope.

Key aspects to focus on:

1. Eliminate any repeated or significantly overlapping subsections across different main sections.
2. Ensure a logical flow of ideas from one section to another and within each section.
3. Adjust the order of sections and subsections if necessary to improve the overall structure.
4. If needed, merge similar subsections or create new ones to better organize the content.
5. Ensure that each section and subsection contributes uniquely to the overall survey without redundancy.
6. Use the provided descriptions for each section and subsection as a guide to maintain the intended focus and content of each part.

When dealing with repeated or overlapping content:
- If subsections in different main sections are very similar, consider merging them into one section or creating a new overarching section.
- If the repetition is necessary due to relevance in multiple contexts, rephrase the subsection to highlight its specific relevance to each main section.
- Always refer back to the original descriptions to ensure that important aspects are not lost during the merging or restructuring process.

In refining the descriptions, aim to:
1. Preserve the core ideas and key terms from the original descriptions.
2. Improve clarity and conciseness where possible.
3. Ensure separability: Each section or subsection should have **an informative description** and **several (no more than 3) sub-domain points with informative sub-description**


Return the final outline in the format:
<format>
# [TITLE OF SURVEY]

## [NAME OF SECTION 1]
Description: [DESCRIPTION OF SECTION 1]
1. [Informative description of key aspect or sub-domain 1 of SECTION 1]
2. ...

### [NAME OF SUBSECTION 1]
Description: [DESCRIPTION OF SUBSECTION 1]
1. [Informative description of key aspect or sub-domain 1 of SUBSECTION 1]
2. ...

### [NAME OF SUBSECTION 2]
Description: [DESCRIPTION OF SUBSECTION 2]
1. [Informative description of key aspect or sub-domain 1 of SUBSECTION 2]
2. ...

...

### [NAME OF SUBSECTION L]
Description: [DESCRIPTION OF SUBSECTION L]
1. [Informative description of key aspect or sub-domain 1 of SUBSECTION L]
2. ...
N. ...

## [NAME OF SECTION 2]

...

## [NAME OF SECTION K]
...

</format>
Only return the final outline without any other informations:
'''

CHECK_CITATION_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].\n\
Below are a list of papers for references:
---
[PAPER LIST]
---
You have written a subsection below:\n\
---
[SUBSECTION]
---
<instruction>
The sentences that are based on specific papers above are followed with the citation of "paper_title" in "[]".
For example 'the emergence of large language models (LLMs) [Language models are few-shot learners; Language models are unsupervised multitask learners; PaLM: Scaling language modeling with pathways]'

Here's a concise guideline for when to cite papers in a survey:
---
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your survey addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your survey.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.
---

Now you need to check whether the citations of "paper_title" in this subsection is correct.
A correct citation means that, the content of corresponding paper can support the sentences you write.
Once the citation can not support the sentence you write, correct the paper_title in '[]' or just remove it.

Remember that you can only cite the 'paper_title' provided above!!!
Any other informations like authors are not allowed cited!!!
Do not change any other things except the citations!!!
</instruction>
Only return the subsection with correct citations:
'''


SUBSECTION_WRITING_PROMPT = '''
You are writing the subsection "[SUBSECTION NAME]" under the section "[SECTION NAME]" for a top-tier and comprehensive survey paper on [TOPIC]. As a distinguished expert, deliver content that combines academic rigor with innovative insights.

The overall outline of your survey is as follows:\n
---
[OVERALL OUTLINE]
---

Below are a list of papers for references:\n
---
[PAPER LIST]
---

<instruction>
Now, focus on writing the content for the subsection "[SUBSECTION NAME]" under "[SECTION NAME]". **The content you write should be approximately [WORD NUM] words**.

Subsection Focus:
---
[DESCRIPTION]
---

Core Requirements:

1. Content Structure
- Begin with a concise overview of the subsection's scope
- Maintain logical flow with clear transitions
- Conclude with synthesis and future directions
- Balance breadth and depth of coverage

2. Academic Analysis
- Provide comparative analysis of different approaches
- Evaluate strengths, limitations, and trade-offs
- Identify emerging trends and challenges
- Present technical details with precision
- Include equations/formal definitions where necessary

3. Citation Guidelines
- You should cite as many relevant paper as possible related to "[SUBSECTION NAME]".
- When writing sentences that are based on specific papers above, you cite the "paper_title" in a '[]' format to support your content.
- Note that the "paper_title" is not allowed to appear without a '[]' format. Once you mention the 'paper_title', it must be included in '[]'.
- Remember that you can only cite the paper provided above and only cite the "paper_title"!!!
- Integration: Support key claims with relevant citations
- Example: "Lin et al. [paper_title1] have shown...  Further studies [paper_title2; paper_title3] confirm..."

4. Critical Insights
- Synthesize information rather than summarize
- Draw connections between different approaches
- Highlight practical implications
- Offer innovative perspectives or future directions
- Support arguments with empirical evidence
- Maintain scholarly tone throughout

Quality Markers:
- Demonstrates deep technical understanding
- Provides novel insights and analysis
- Maintains objective academic tone
- Presents coherent narrative flow
- Supports all key claims with citations

Remember, the quality of your work should reflect the standards expected in top-tier academic publications. Your analysis should be thorough, your arguments well-supported, and your insights valuable to the academic community. Approach this task as if your reputation as a leading expert in the field depends on the quality of this subsection.
</instruction>

Provide the content for subsection "[SUBSECTION NAME]" in this format:
<format>
[CONTENT OF SUBSECTION]
</format>

Only return the content more you write for the subsection [SUBSECTION NAME] without any other information, ensuring it provides a comprehensive, in-depth analysis that meets the high academic standards described above. Your work will be evaluated based on its scholarly merit, analytical depth, and potential contribution to the field. 
Do not repeat the subsection title at the beginning of your response. Start directly with the content of the subsection.
Finally, remember that you are an academic writing expert. Do not use additional subheadings in your content, such as' ### Conclusion', etc., and **be sure not to create content as you would in a summary report**.
'''

LCE_PROMPT = '''
You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].

Now you need to help to refine one of the subsection to improve th ecoherence of your survey.

You are provied with the content of the subsection along with the previous subsections and following subsections.

Previous Subsection:
--- 
[PREVIOUS]
---

Following Subsection:
---
[FOLLOWING]
---

Subsection to Refine: 
---
[SUBSECTION]
---


If the content of Previous Subsection is empty, it means that the subsection to refine is the first subsection.
If the content of Following Subsection is empty, it means that the subsection to refine is the last subsection.

Now refine the subsection to enhance coherence, and ensure that the content of the subsection flow more smoothly with the previous and following subsections. 

Remember that keep all the essence and core information of the subsection intact. Do not modify any citations in [] following the sentences.

Only return the whole refined content of the subsection without any other informations (like "Here is the refined subsection:")!

The subsection content:
'''
