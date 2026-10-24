"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "entity_extraction"
] = """-目标-
给定一段视频内容的文本描述和一组实体类型，从文本中识别出所有属于这些类型的实体，以及所有已识别实体之间的关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，使用原始形式（保留原文中的大小写和语言）
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: 对实体属性和活动的详细描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1中识别的实体中，找出所有*明确相关*的实体对 (source_entity, target_entity)。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称（与步骤1中一致）
- target_entity: 目标实体名称（与步骤1中一致）
- relationship_description: 解释为什么认为源实体和目标实体之间存在关系
- relationship_strength: 表示关系强度的数值评分（1-10）
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 将步骤1和步骤2中识别的所有实体和关系作为一个列表返回。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成时，输出 {completion_delimiter}

-重要规则-
- 实体命名：优先使用具体名称（如"张教授"），避免使用泛称（如"一个人"、"女性"、"演讲者"）。如果文本中提到了人名，必须使用人名而非泛称。
- 关系描述：必须描述具体的动作或联系（如"教导"、"批评"、"发明了"），禁止使用"相关"、"有关系"、"出现在同一场景"等模糊描述。
- 实体去重：如果同一实体在文本中有多种称呼（如"张教授"和"张老师"），统一使用最具体的名称，不要创建多个实体。

######################
-示例-
######################
示例 1:

Entity_types: [person, location, object, event, action]
Text:
李明站在实验室中央，手中紧握着那块发光的晶体。这是他在西藏考察时从冰川深处发现的。张教授走过来，仔细端详着晶体表面的纹路，若有所思地说："这种结构我在二十年前的论文里见过，当时我们以为只是理论推测。"门外传来急促的脚步声，是王助理带着最新的光谱分析报告赶来了。
################
Output:
("entity"{tuple_delimiter}"李明"{tuple_delimiter}"person"{tuple_delimiter}"李明是一名研究人员，在西藏考察时发现了发光晶体，目前在实验室中进行研究。"){record_delimiter}
("entity"{tuple_delimiter}"张教授"{tuple_delimiter}"person"{tuple_delimiter}"张教授是一位资深学者，二十年前曾在论文中预测过类似的晶体结构。"){record_delimiter}
("entity"{tuple_delimiter}"王助理"{tuple_delimiter}"person"{tuple_delimiter}"王助理负责光谱分析工作，带来了最新的分析报告。"){record_delimiter}
("entity"{tuple_delimiter}"发光晶体"{tuple_delimiter}"object"{tuple_delimiter}"一块从西藏冰川深处发现的发光晶体，表面有特殊纹路，具有独特的结构。"){record_delimiter}
("entity"{tuple_delimiter}"实验室"{tuple_delimiter}"location"{tuple_delimiter}"研究人员进行晶体分析的工作场所。"){record_delimiter}
("entity"{tuple_delimiter}"西藏冰川"{tuple_delimiter}"location"{tuple_delimiter}"李明发现发光晶体的地点，位于西藏的冰川深处。"){record_delimiter}
("entity"{tuple_delimiter}"光谱分析"{tuple_delimiter}"action"{tuple_delimiter}"对晶体进行的科学分析方法，用于研究其成分和结构。"){record_delimiter}
("relationship"{tuple_delimiter}"李明"{tuple_delimiter}"发光晶体"{tuple_delimiter}"李明在西藏考察时发现了这块发光晶体，是晶体的发现者。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"张教授"{tuple_delimiter}"发光晶体"{tuple_delimiter}"张教授对晶体进行观察分析，并发现其结构与自己二十年前的理论预测吻合。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"李明"{tuple_delimiter}"西藏冰川"{tuple_delimiter}"李明在西藏冰川进行考察时发现了晶体。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"王助理"{tuple_delimiter}"光谱分析"{tuple_delimiter}"王助理负责执行光谱分析并带来报告。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"李明"{tuple_delimiter}"张教授"{tuple_delimiter}"李明和张教授在实验室中共同研究晶体。"{tuple_delimiter}6){completion_delimiter}
#############################
示例 2:

Entity_types: [person, location, object, event, concept]
Text:
纪录片的画面切换到了一片茂密的热带雨林。解说员介绍道，这里是亚马逊流域最后一片未被开发的原始森林。镜头缓缓推进，一群金刚鹦鹉从树冠层飞过，它们鲜艳的羽毛在阳光下闪烁。地面上，一只美洲豹正悄无声息地穿过灌木丛，跟踪着一群水豚。这种捕猎行为在雨季尤为常见，因为猎物会集中在水源附近。
################
Output:
("entity"{tuple_delimiter}"亚马逊雨林"{tuple_delimiter}"location"{tuple_delimiter}"亚马逊流域最后一片未被开发的原始热带雨林，拥有丰富的生物多样性。"){record_delimiter}
("entity"{tuple_delimiter}"金刚鹦鹉"{tuple_delimiter}"object"{tuple_delimiter}"一群拥有鲜艳羽毛的鸟类，从树冠层飞过。"){record_delimiter}
("entity"{tuple_delimiter}"美洲豹"{tuple_delimiter}"object"{tuple_delimiter}"一只在灌木丛中悄无声息移动的大型猫科动物，正在跟踪猎物。"){record_delimiter}
("entity"{tuple_delimiter}"水豚"{tuple_delimiter}"object"{tuple_delimiter}"一群被美洲豹跟踪的草食动物。"){record_delimiter}
("entity"{tuple_delimiter}"捕猎行为"{tuple_delimiter}"event"{tuple_delimiter}"美洲豹跟踪水豚的捕猎活动，在雨季尤为常见。"){record_delimiter}
("entity"{tuple_delimiter}"雨季"{tuple_delimiter}"concept"{tuple_delimiter}"雨季期间猎物会集中在水源附近，影响捕猎行为的频率。"){record_delimiter}
("relationship"{tuple_delimiter}"美洲豹"{tuple_delimiter}"水豚"{tuple_delimiter}"美洲豹正在跟踪水豚，二者是捕食者与猎物的关系。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"金刚鹦鹉"{tuple_delimiter}"亚马逊雨林"{tuple_delimiter}"金刚鹦鹉栖息在亚马逊雨林的树冠层。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"捕猎行为"{tuple_delimiter}"雨季"{tuple_delimiter}"捕猎行为在雨季更为频繁，因为猎物集中在水源附近。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"美洲豹"{tuple_delimiter}"捕猎行为"{tuple_delimiter}"美洲豹是捕猎行为的执行者。"{tuple_delimiter}9){completion_delimiter}
#############################
示例 3:

Entity_types: [person, organization, location, event, object, action, concept, time]
Text:
2023年春节期间，故宫博物院举办了"盛世华章"大型文物展览。展览展出了来自全国各地博物馆的三百余件珍贵文物，包括唐代的三彩骆驼俑和宋代的汝窑天青釉碗。策展人陈教授介绍说，这次展览旨在展现中华文明五千年的辉煌历程。开幕式当天，超过五万名观众涌入故宫，创下了近年来的参观记录。
#############
Output:
("entity"{tuple_delimiter}"故宫博物院"{tuple_delimiter}"organization"{tuple_delimiter}"故宫博物院是举办"盛世华章"文物展览的机构，春节期间吸引了大量观众。"){record_delimiter}
("entity"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"event"{tuple_delimiter}"一场展出三百余件珍贵文物的大型展览，旨在展现中华文明五千年的辉煌历程。"){record_delimiter}
("entity"{tuple_delimiter}"陈教授"{tuple_delimiter}"person"{tuple_delimiter}"盛世华章展览的策展人，介绍了展览的主旨和意义。"){record_delimiter}
("entity"{tuple_delimiter}"三彩骆驼俑"{tuple_delimiter}"object"{tuple_delimiter}"唐代文物，是展览中展出的珍贵文物之一。"){record_delimiter}
("entity"{tuple_delimiter}"汝窑天青釉碗"{tuple_delimiter}"object"{tuple_delimiter}"宋代文物，是展览中展出的珍贵文物之一。"){record_delimiter}
("entity"{tuple_delimiter}"2023年春节"{tuple_delimiter}"time"{tuple_delimiter}"展览举办的时间。"){record_delimiter}
("entity"{tuple_delimiter}"中华文明"{tuple_delimiter}"concept"{tuple_delimiter}"展览旨在展现的主题，涵盖五千年的辉煌历程。"){record_delimiter}
("relationship"{tuple_delimiter}"故宫博物院"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"故宫博物院是展览的主办方和举办地点。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"陈教授"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"陈教授担任展览的策展人，负责策划和介绍。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"三彩骆驼俑"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"三彩骆驼俑是展览中展出的代表性文物。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"汝窑天青釉碗"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"汝窑天青釉碗是展览中展出的代表性文物。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"盛世华章展览"{tuple_delimiter}"中华文明"{tuple_delimiter}"展览以展现中华文明的辉煌历程为主旨。"{tuple_delimiter}9){completion_delimiter}
#############################
-真实数据-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个专业的信息整合助手，负责将以下数据生成一段全面的摘要描述。
给定一个或两个实体，以及一组相关描述，它们都与同一个实体或实体组有关。
请将所有描述合并为一段完整、全面的描述。确保包含所有描述中的关键信息。
如果描述之间存在矛盾，请解决矛盾并提供一致的摘要。
使用第三人称书写，并包含实体名称以提供完整上下文。

#######
-数据-
实体: {entity_name}
描述列表: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """上一轮提取遗漏了很多实体。请使用相同的格式继续补充：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎仍有一些实体被遗漏了。请回答 YES 或 NO，是否还有需要补充的实体。
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "person",        # 人物/角色
    "organization",  # 组织/机构/团体
    "location",      # 地点/场景/场所
    "event",         # 事件/活动
    "object",        # 物体/道具/工具
    "action",        # 动作/行为/活动
    "concept",       # 抽象概念/主题
    "time",          # 时间/时期/日期
]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]


PROMPTS[
    "naive_rag_response"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the input data tables, suitable for the specified response length and format.
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{content_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the input data tables appropriate for the response length and format.
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""


PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """-Goal-
For a given query, generate a declarative sentence to serve as a query for retrieving relevant knowledge.

######################
-Examples-
######################

Question: What are the main characters? \n(A) Alice\n(B) Bob\n(C) Charlie\n(D) Dana
################
Output:
The main characters. (Maybe Alice, Bob, Charlie or Dana)

Question: What locations are shown in the video?
################
Output:
The locations shown in the video.

Question: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
################
Output:
The animals that appear in the wildlife footage. (Maybe Lions, Elephants or Zebras)

#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """-Goal-
Given a question that may include scene-related information, generate a declarative sentence to serve as a query for retrieving relevant video segments.

######################
-Examples-
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Maybe Sunny, Rainy, Snowy or Windy)

#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
    "keywords_extraction"
] = """- Goal -
Given a query, extract the relevant keywords that can help answer the query. Please list the keywords separated by commas.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest, scene

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
color, car, chases, main character, city

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
weather, opening scene, film, Sunny, Rainy, Snowy, Windy

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
    "filtering_segment"
] = """---Role---

You are a helpful assistant to determine whether the video may contain information relevant to the knowledge based on its rough caption.
Please note that this is a rough caption of the video segments, which means it may not directly contain the answer but may indicate that the video segment is likely to contain information relevant to answering the question. 

---Video Caption---

{caption}

---Knowledge We Need---

{knowledge}

---Answer---
Please provide an answer that begins with "yes" or "no," followed by a brief step-by-step explanation.
Answer:
"""



PROMPTS[
    "videorag_response"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
Reference relevant video segments within the answers, specifying the video name and start & end timestamps. Use the following reference format:

---Example of Reference---

In one segment, the film highlights the devastating effects of deforestation on wildlife habitats [1]. Another part illustrates successful conservation efforts that have helped endangered species recover [2].

#### Reference:
[1] video_name_1, 05:30, 08:00  
[2] video_name_2, 25:00, 28:00 

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""

PROMPTS[
    "videorag_response_wo_reference"
] = """---Role---

You are a helpful assistant responding to a query with retrieved knowledge.

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a response of the target length and format that responds to the user's question with relevant general knowledge.
Summarize useful and relevant information from the retrieved text chunks and the information retrieved from videos, suitable for the specified response length and format.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Notice---
Please add sections and commentary as appropriate for the length and format if necessary. Format the response in Markdown.
"""

PROMPTS[
    "videorag_response_for_multiple_choice_question"
] = """---Role---

You are a helpful assistant responding to a multiple-choice question with retrieved knowledge.

---Goal---

Generate a concise response that addresses the user's question by summarizing relevant information derived from the retrieved text and video content. Ensure the response aligns with the specified format and length.
Please note that there is only one choice is correct.

---Target response length and format---

{response_type}

---Retrieved Information From Videos---

{video_data}

---Retrieved Text Chunks---

{chunk_data}

---Goal---

Generate a concise response that addresses the user's question by summarizing relevant information derived from the retrieved text and video content. Ensure the response aligns with the specified format and length.
Please note that there is only one choice is correct.

---Notice---
Please provide your answer in JSON format as follows:
{{
    "Answer": "The label of the answer, like A/B/C/D or 1/2/3/4 or others, depending on the given query"
    "Explanation": "Provide explanations for your choice. Use sections and commentary as needed to ensure clarity and depth. Format the response in Markdown."
}}
Key points:
1. Ensure that the "Answer" reflects the correct label format.
2. Structure the "Explanation" for clarity, using Markdown for any necessary formatting.
"""