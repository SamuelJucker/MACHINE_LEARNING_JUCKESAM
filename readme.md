
Trained with A100 on Google Colab.

What is the Goal of this project:

The Day to day life of financial professionals is usually very stressful and busy. Yet it is crucial to stay up to date with current events and news. Available Market overviews only offer a human-biased insight in the general events of the day. With this Project we can summarize each article by hand and therefore know that there is no human interference. In the end we get, what may seem trivial at first, but in the end is the decisive factor for success in the financial world, time.



Saving time meticulously and therefore striving for efficiency is, not only, but especially in the financial world one of the core Values.

Download model from my drive...........https://drive.google.com/drive/folders/1nE1GgVeeQ9vO1RqB09HyiuhTv1Ru13MH?usp=sharing Extract the model into the Models Folder

The Training and Validation is done here: custom_finnews_summarization.ipynb, Running the model as well as Data Download and Preprocessing should be done locally.  In Google Colab one should consider to reduce the amount of epochs and use a subset of the Data to reduce time. This should especially be done if a free tier is used. An Example Summary which would be adjustable is also done in the colab Notebook. However there is also a frontend to run locally.


For Data Collection the following Dataset was chosen: ECTSum: A New Benchmark Dataset For Bullet Point Summarization of
Long Earnings Call Transcripts : download from: https://github.com/rajdeep345/ECTSum/blob/main/README.md

I also tried enriching Data with the following Data Set: dolly: https://huggingface.co/datasets/databricks/databricks-dolly-15k : https://arxiv.org/abs/2203.02155 . but due to the differen formats decided against it in final format.

Key Takeaways: there are barely any naturally summarized domainspecific Datasets publicly available. Synthetic Dataset may be interesting, but if generated with for example Chat GPT would generate almost certainly inferior models.




