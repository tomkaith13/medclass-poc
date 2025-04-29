import dspy
import os
from dotenv import load_dotenv
from medical_classification.classify import specialty_classify


load_dotenv()

lm = dspy.LM(
    "vertex_ai/gemini-2.0-flash-lite",
    vertex_project=os.getenv("PROJECT_ID"),
    vertex_location=os.getenv("LOCATION"),
    temperature=0.1,
    cache=True,
)
dspy.configure(lm=lm)
# dspy.settings.configure(track_usage=True)


def main():
    print("Hello from med-classifier!")
    resp = lm("say this is a test Leaguer", temperature=0.5)
    print(resp)

    # wall_of_text = "I have a rash on my arm and it's really itchy. What should I do?"
    # wall_of_text = 'You probably make time for other people. Maybe it’s a work meeting, your kids’ after-school program, or a friend that needs your help. What if you set aside time for yourself just like you do when someone else needs you? Help yourself get to 5,000 steps by making a walking appointment. Think of it as a break from all the stressors going on. A breather just for you.'
    # wall_of_text = "Kids out of school, working from home, and social distancing have become our new reality. What does parenting look like during a global pandemic? Is it really that important to have social-media-worthy, colour-coded schedules? Ban all screen time? Become the perfect home-school teacher? All while still having a home-cooked meal on the table? Probably not. No one was prepared for #pandemicparenting but following along with this 7-day program can arm you with some ideas and tips to make the most of this unprecedented time."
    wall_of_text = 'Running nose, sore throat, and fever'
    # wall_of_text = 'bla bla bal'
    # wall_of_text = "A healthy brain helps us feel good in all aspects of life. In order for it to work well, we need to nourish it with healthy foods, thoughts, and activities while also reducing exposure to the stuff that damages it. We collaborated with the Centre for Applied Neuroscience to help you learn more about your brain, how to keep it healthy, and how to feel better. For the next three weeks, you’ll focus on goals like understanding how to support your brain with nutrition and supplements, thinking patterns, and lifestyle adjustments. You’ll also focus on reducing exposure to negative influences."
    print(f"Classifying wall of text: {wall_of_text}")
    classification_resp = specialty_classify(
        wall_of_text=wall_of_text,
    )
    print(f"response: {classification_resp}")
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    main()
