# Medical Specialty Classifier
<tbd>

## Usage
This code uses VertexAI via LiteLLM integration.
Read these [pre-requisites](https://docs.litellm.ai/docs/providers/vertex#pre-requisites)
So to setup the creds, hit
```bash
gcloud auth application-default login  
```

And ensure u add your respective `project` and `location` in `.env` file which gets loaded up at init.

```bash
(med-classifier) ➜  med-classifier git:(main) ✗ uv run main.py
Hello from med-classifier!
['Okay, this is a test Leaguer.\n']
Classifying wall of text: A healthy brain helps us feel good in all aspects of life. In order for it to work well, we need to nourish it with healthy foods, thoughts, and activities while also reducing exposure to the stuff that damages it. We collaborated with the Centre for Applied Neuroscience to help you learn more about your brain, how to keep it healthy, and how to feel better. For the next three weeks, you’ll focus on goals like understanding how to support your brain with nutrition and supplements, thinking patterns, and lifestyle adjustments. You’ll also focus on reducing exposure to negative influences.
response: Prediction(
    reasoning='The text discusses brain health, healthy habits, and lifestyle adjustments to improve well-being. This aligns with the focus of neurology and psychiatry. Given the emphasis on mental well-being and thinking patterns, psychiatry is the more relevant specialty.',
    specialty='psychiatry',
    confidence=0.95
)




[2025-04-29T13:41:33.281700]

System message:

Your input fields are:
1. `wall_of_text` (str)
Your output fields are:
1. `reasoning` (str)
2. `specialty` (Literal['dermatology', 'orthopedics', 'cardiology', 'neurology', 'psychiatry', 'pediatrics', 'internal medicine', 'family medicine', 'emergency medicine', 'radiology'])
3. `confidence` (float)
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## wall_of_text ## ]]
{wall_of_text}

[[ ## reasoning ## ]]
{reasoning}

[[ ## specialty ## ]]
{specialty}        # note: the value you produce must exactly match (no extra characters) one of: dermatology; orthopedics; cardiology; neurology; psychiatry; pediatrics; internal medicine; family medicine; emergency medicine; radiology

[[ ## confidence ## ]]
{confidence}        # note: the value you produce must be a single float value

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Classify a wall of text to an appropriate medical specialty and confidence score. If nothing matches, return 'internal medicine'


User message:

[[ ## wall_of_text ## ]]
A healthy brain helps us feel good in all aspects of life. In order for it to work well, we need to nourish it with healthy foods, thoughts, and activities while also reducing exposure to the stuff that damages it. We collaborated with the Centre for Applied Neuroscience to help you learn more about your brain, how to keep it healthy, and how to feel better. For the next three weeks, you’ll focus on goals like understanding how to support your brain with nutrition and supplements, thinking patterns, and lifestyle adjustments. You’ll also focus on reducing exposure to negative influences.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## specialty ## ]]` (must be formatted as a valid Python Literal['dermatology', 'orthopedics', 'cardiology', 'neurology', 'psychiatry', 'pediatrics', 'internal medicine', 'family medicine', 'emergency medicine', 'radiology']), then `[[ ## confidence ## ]]` (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
The text discusses brain health, healthy habits, and lifestyle adjustments to improve well-being. This aligns with the focus of neurology and psychiatry. Given the emphasis on mental well-being and thinking patterns, psychiatry is the more relevant specialty.

[[ ## specialty ## ]]
psychiatry

[[ ## confidence ## ]]
0.95

[[ ## completed ## ]]





(med-classifier) ➜  med-classifier git:(main) ✗ 

```