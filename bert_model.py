from transformers import pipeline


# LOAD AI MODEL
classifier = pipeline(
    "text-classification",
    model="jy46604790/Fake-News-Bert-Detect"
)

def predict_news(text):


    result = classifier(text)[0]


    label = result["label"]
    score = round(result["score"] * 100, 2)


    # LABEL FIX
    if label.upper() == "LABEL_1":
        final_label = "REAL"


        explanation = (
            "AI found this news similar to trusted and verified reporting."
        )


    else:
        final_label = "FAKE"


        explanation = (
            "AI detected misleading or suspicious fake-news patterns."
        )


    return {
        "label": final_label,
        "confidence": score,
        "explanation": explanation
    }