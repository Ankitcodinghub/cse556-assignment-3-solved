# cse556-assignment-3-solved
**TO GET THIS SOLUTION VISIT:** [CSE556 Assignment 3 Solved](https://www.ankitcodinghub.com/product/cse-556-natural-language-processing-assignment-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;127099&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE556 Assignment 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
General Instructions:

● Every assignment has to be attempted by four people. At least one subtask has to be done by one team member. All members need to have a working understanding of the entire code and assignment.

● Create separate .ipynb or .py files for each part. The file name should follow the format: “A3_&lt;Part number&gt;.ipynb/.py”

● Carefully read the deliverables for all tasks. Along with the code files, submit all the other files mentioned for each task, strictly following the naming convention instructed.

● Only one person has to submit the zip file containing all the mentioned files and the report PDF. It will be named “A3_&lt;Group No&gt;.zip”. The person with the alphabetically smallest name should submit it.

● You are required to submit your trained models. You must also retain all your checkpoints and load and run them during the demo.

● Your report must include the details of each group member’s contribution.

Task Definition – Given two sentences, calculate the similarity between these two sentences.

The similarity is given as a score ranging from 0 to 5.

Train datapoint examples –

score sentence1 sentence2

2.400 A woman is playing the guitar. A man is playing guitar.

For this task, you are required to implement three setups:

● Setup 1A – You are required to train a BERT model (google-bert/bert-base-uncased · Hugging Face) using HuggingFace for the task of Text Similarity. You are required to obtain BERT embeddings while making use of a special token used by BERT for separating multiple sentences in an input text and an appropriate linear layer or setting of BertForSequenceClassification (BERT) framework for a float output. Choose a suitable loss function. Report the required evaluation metric on the validation set.

● Setup 1B – You are required to make use of the Sentence-BERT model

(https://arxiv.org/pdf/1908.10084.pdf) and the SentenceTransformers framework (Sentence-Transformers). For this setup, make use of the Sentence-BERT model to encode the sentences and determine the cosine similarity between these embeddings for the validation set. Report the required evaluation metric on the validation set.

You must save and submit your model checkpoints for 1A and 1C in an appropriate format.

scale of 1 for training the sentence transformers.

Evaluation Metrics – Pearson Correlation

● Generate the following plots for Setup 1A and 1C:

a) Loss Plot: Training Loss and Validation Loss V/s Epochs

b) Analyse and Explain the plots obtained as well

● Provide a brief comparison and explanation for the performance differences between the three setups in the report.

● Provide all evaluation metrics for all the setups in your report pdf.

The dataset is attached as a file in the assignment post. It contains the following files –

● A training data file of the name – ‘train.csv’

● A validation data file of the name – ‘dev.csv’

● a sample test file with the name – ‘sample_test.csv’

● A sample of the CSV file to be generated during the demo – ‘sample_demo.csv’

To download the dataset, make use of HuggingFace datasets – Load a Dataset – Hugging Face For downloading the training dataset, use the command datasets.load_dataset(“wmt16″,”de-en”, split=”train[:50000]”)

Each dataset sample consists of a piece of text in German and its translation in English. Use this data to train German-English translation models.

You are required to implement the following setups –

● Setup 2A – Train an encoder-decoder transformer model using a deep learning library like PyTorch. (Transformer — PyTorch 2.2 documentation). Tutorial: Language Translation with nn.Transformer and torchtext — PyTorch Tutorials 2.2.1+cu121 documentation. You must train the sequence-to-sequence model from scratch for German-English translation and report the evaluation metrics on the validation and test datasets.

● Setup 2B – Perform zero-shot evaluation of the t5-small model

(https://huggingface.co/google-t5/t5-small) for the task of machine translation from German to English. Zero-shot evaluation refers to testing of a language model without explicitly training or fine-tuning it for the given task. The t5-small model allows for this setup by prepending a prefix to the input sentence. This prefix is available through carefully reading the model documentation for the T5 model available at T5. Utilise this to generate the translations for the validation and testing sets and report the required

You must save and submit your model checkpoints for 2A and 2C in an appropriate format.

Evaluation Metrics – i) String-based metrics – BLEU (BLEU – a Hugging Face Space by evaluate-metric) and METEOR (METEOR – a Hugging Face Space by evaluate-metric) ii) Machine Learning Based Metric – BERTScore (BERT Score – a Hugging Face Space by

evaluate-metric)

● Generate the following plots for Setup 2A and 2C:

a) Loss Plot: Training Loss and Validation Loss V/s Epochs

b) Analyse and Explain the plots obtained as well

● Provide a brief comparison and explanation for the performance differences between the three setups in the report.

● Provide all evaluation metrics for all the setups in your report pdf.
