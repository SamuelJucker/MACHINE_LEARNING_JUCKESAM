
Trained with A100 on Google Colab.
Hardware Req.
Runs with cheap trash i 7


What is the Goal of this project.




 what are knowledgraphs?
The Day to day life of financial professionals is usually very stressful and busy. Yet it is crucial to stay up to date with current events and news. Available Market overviews only offer a human-biased insight in the general events of the day. With this Project we can summarize each article by hand and therefore know that there is no human interference. In the end we get, what may seem trivial at first, but in the end is the decisive factor for success in the financial world, time.

The Training is done here: custom_finnews_summarization.ipynb

Saving time meticulously and therefore striving for efficiency is, not only, but especially in the financial world one of the core Values.

Download model from my drive...........https://drive.google.com/drive/folders/1nE1GgVeeQ9vO1RqB09HyiuhTv1Ru13MH?usp=sharing

https://arxiv.org/abs/2203.02155
https://arxiv.org/pdf/2210.12467v2 / ECTSum: A New Benchmark Dataset For Bullet Point Summarization of
Long Earnings Call Transcripts : download from: https://github.com/rajdeep345/ECTSum/blob/main/README.md


dolly: https://huggingface.co/datasets/databricks/databricks-dolly-15k : https://arxiv.org/abs/2203.02155
kaggle alpaca lora// https://www.kaggle.com/code/gbhacker23/wealth-alpaca-lora/notebook // acc- to kaggle: https://www.kaggle.com/code/gbhacker23/wealth-alpaca-lora?scriptVersionId=125008762&cellId=8 // analyis: inference notebooks etc. https://github.com/gaurangbharti1/wealth-alpaca // // cleansed: https://huggingface.co/datasets/gbharti/wealth-alpaca_lora   // csv: https://huggingface.co/datasets/gbharti/finance-alpaca-csv


maybe financial summarization pegasus: (maybe benchmark) https://huggingface.co/human-centered-summarization/financial-summarization-pegasus // https://arxiv.org/pdf/1912.08777.pdf // It is based on the PEGASUS model and in particular PEGASUS fine-tuned on the Extreme Summarization (XSum) dataset: google/pegasus-xsum model. PEGASUS was originally proposed by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu in PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. //  ROUGE scores (similarity to a human-generated summary) compared to our base model. Moreover, our advanced model also offers several convenient plans tailored to different use cases and workloads, ensuring a seamless experience for both personal and enterprise access.

// one sentence summar: https://huggingface.co/datasets/EdinburghNLP/xsum // https://huggingface.co/datasets/EdinburghNLP/xsum // https://arxiv.org/abs/1808.08745 // paper.Document to one sentence...

// xsum:https://paperswithcode.com/dataset/xsum



Top 5 Key Takeaways
Instruction Tuning is Key: For zero-shot summarization tasks, instruction tuning significantly outperforms mere model scaling.
High-Quality References: The presence of high-quality reference summaries is vital for both training and evaluating summarization models effectively.
Human-Level Performance: Some instruction-tuned LLMs are approaching human-level performance in summarization tasks.
Stylistic Differences: There are notable stylistic differences between human and machine-generated summaries, with humans typically producing more abstractive summaries.
Evaluation Metrics: The effectiveness of automatic evaluation metrics depends heavily on the quality of reference summaries, indicating a need for better quality control in evaluation datasets.

edtsum: https://huggingface.co/datasets/TheFinAI/flare-edtsum


// what is padding etc??


info: 
### Datasets for Financial News Summarization

To effectively summarize financial news, you should use datasets that are diverse and comprehensive, covering various aspects of financial reporting. Here are some recommended datasets:

1. **Financial News Articles**:
   - **Reuters News Dataset**: Provides a large collection of financial and business news articles.
   - **Bloomberg News**: Subscription-based access to financial news with a comprehensive archive.
   - **Yahoo Finance News**: Articles covering market news, stock analysis, and financial reports.
   
2. **Specialized Financial Datasets**:
   - **FinBERT Dataset**: Specifically designed for financial sentiment analysis, which can be useful for understanding the tone and implications of financial news.
   - **FNS (Financial News Summary) Dataset**: Contains summaries of financial news, which can be used for training and evaluating summarization models.
   
3. **General News Summarization Datasets**:
   - **CNN/DailyMail**: Although not specific to finance, it contains a wealth of news articles and summaries that can be useful for general summarization training.
   - **XSUM**: Another general news summarization dataset with high-quality reference summaries, useful for evaluating summarization models.

4. **Academic and Research Datasets**:
   - **DUC (Document Understanding Conferences) datasets**: Often used in academic research for summarization tasks, though not specifically financial, they can provide robust benchmarking data.

### Differences Between Extractive and Abstractive Summaries

Understanding the difference between extractive and abstractive summaries is crucial for tailoring the summarization approach to your needs:

1. **Extractive Summarization**:
   - **Definition**: Extractive summarization involves selecting and extracting key sentences or phrases directly from the source text to form a summary.
   - **Characteristics**:
     - **Accuracy**: Maintains high factual accuracy as it uses exact sentences from the original text.
     - **Style**: May result in summaries that are disjointed or lack natural flow since the extracted sentences are taken out of their original context.
     - **Complexity**: Typically simpler to implement as it requires identifying and ranking important sentences.
   - **Use Cases**: Suitable for applications where factual correctness is paramount, such as technical documentation or legal documents.
   
2. **Abstractive Summarization**:
   - **Definition**: Abstractive summarization involves generating new sentences that capture the essence of the original text, using novel words and phrases.
   - **Characteristics**:
     - **Creativity**: Produces more coherent and fluent summaries that can paraphrase and generalize the content.
     - **Risk of Error**: Higher risk of introducing factual inaccuracies as it involves generating new text.
     - **Complexity**: More complex to develop and train as it requires understanding and rephrasing the content.
   - **Use Cases**: Ideal for applications requiring natural language understanding and generation, such as news summarization, storytelling, or customer service responses.

### Application in Financial News Summarization

- **Extractive Summarization**: Can be used for quickly generating summaries of financial reports or earnings calls where precision and factual correctness are crucial.
- **Abstractive Summarization**: Useful for creating engaging and readable summaries of financial news articles, market analysis, and expert opinions, where the readability and coherence of the summary are important.

By leveraging the appropriate datasets and understanding the differences between summarization techniques, you can better tailor your approach to effectively summarize financial news, balancing accuracy and readability based on the context.


(Some non-default generation parameters are set in the model config. These should go into a GenerationConfig file (https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model) instead. This warning will be raised to an exception in v4.41.
Non-default generation parameters: {'max_length': 256, 'num_beams': 8, 'length_penalty': 0.8, 'forced_eos_token_id': 1}
('./results/tokenizer_config.json',
 './results/special_tokens_map.json',
 './results/spiece.model',
 './results/added_tokens.json',
 './results/tokenizer.json'))
Instruction Tuning is Key: For zero-shot summarization tasks, instruction tuning significantly outperforms mere model scaling.

errors: https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model